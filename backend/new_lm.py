"""Bayesian backend runtime with websocket update synchronization."""

from __future__ import annotations

import atexit
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import bayesian
import numpy as np
import starlette.websockets
import torch
from fastapi import FastAPI, WebSocket
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EXPECTED_FILTERED_TOKEN_COUNT = 17_235
HF_SPACE_MARKER = "▁"
ROOT_MARKER = "^"
PIDFILE_PATH = Path("/tmp/dotter_new_lm.pid")
START_TS = time.monotonic()


def _log(msg: str) -> None:
    elapsed = time.monotonic() - START_TS
    print(f"[new_lm +{elapsed:8.3f}s] {msg}", file=sys.stderr, flush=True)


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _pick_device() -> str:
    # CUDA-only by design: never add CPU fallback here.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for backend runtime; CPU fallback is forbidden.")
    return "cuda"


def _initial_context_token_id(tokenizer: AutoTokenizer) -> int:
    if tokenizer.bos_token_id is not None:
        return int(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    raise ValueError("tokenizer must expose either bos_token_id or eos_token_id")


def _internal_token_to_hf_token(token: str) -> str:
    return token.replace("_", HF_SPACE_MARKER)


def _bayes_string_to_model_text(full_string: str) -> str:
    if not full_string.startswith(ROOT_MARKER):
        raise ValueError(f"expected requested prior to start with {ROOT_MARKER!r}: {full_string!r}")
    surface = full_string.removeprefix(ROOT_MARKER)
    if surface.startswith("_"):
        surface = surface[1:]
    return surface.replace("_", " ")


def _kill_process(pid: int) -> None:
    _log(f"terminating previous backend pid={pid}")
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except OSError:
            _log(f"previous backend pid={pid} exited after SIGTERM")
            return
        time.sleep(0.1)
    _log(f"previous backend pid={pid} still alive; sending SIGKILL")
    os.kill(pid, signal.SIGKILL)


def _ensure_singleton_backend_process() -> None:
    current_pid = os.getpid()
    _log(f"singleton check start pid={current_pid}")
    if PIDFILE_PATH.exists():
        existing_pid_text = PIDFILE_PATH.read_text(encoding="utf-8").strip()
        if existing_pid_text:
            existing_pid = int(existing_pid_text)
            if existing_pid != current_pid:
                try:
                    _kill_process(existing_pid)
                except OSError:
                    pass

    PIDFILE_PATH.write_text(f"{current_pid}\n", encoding="utf-8")
    _log(f"singleton pidfile written path={PIDFILE_PATH}")

    def _cleanup_pidfile() -> None:
        if PIDFILE_PATH.exists():
            try:
                pid_text = PIDFILE_PATH.read_text(encoding="utf-8").strip()
                if pid_text == str(current_pid):
                    PIDFILE_PATH.unlink()
            except OSError:
                pass

    atexit.register(_cleanup_pidfile)


@dataclass(frozen=True)
class RequestedPrior:
    full_string: str
    last_token_lexindex: int


@dataclass
class CacheTrieNode:
    children: dict[int, "CacheTrieNode"] = field(default_factory=dict)
    past_key_values: Any | None = None
    last_logits: np.ndarray | None = None


class PrefixCacheTrie:
    """Token-prefix cache with trie structure and KV payloads."""

    def __init__(self) -> None:
        self.root = CacheTrieNode()

    def reset(self) -> None:
        self.root = CacheTrieNode()

    def _longest_cached_prefix(self, token_ids: list[int]) -> tuple[int, CacheTrieNode]:
        node = self.root
        best_len = 0
        best_node = self.root
        for i, token in enumerate(token_ids):
            child = node.children.get(token)
            if child is None:
                break
            node = child
            if node.past_key_values is not None:
                best_len = i + 1
                best_node = node
        return best_len, best_node

    def _ensure_path(self, token_ids: list[int]) -> CacheTrieNode:
        node = self.root
        for token in token_ids:
            if token not in node.children:
                node.children[token] = CacheTrieNode()
            node = node.children[token]
        return node

    def infer_next_logits(
        self,
        token_ids: list[int],
        model: AutoModelForCausalLM,
        device: str,
    ) -> np.ndarray:
        target_node = self._ensure_path(token_ids)
        if target_node.last_logits is not None:
            return target_node.last_logits

        prefix_len, prefix_node = self._longest_cached_prefix(token_ids)
        suffix = token_ids[prefix_len:]
        if prefix_len == 0:
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            output = model(input_ids=input_ids, use_cache=True)
        else:
            input_ids = torch.tensor([suffix], dtype=torch.long, device=device)
            attention_mask = torch.ones(
                (1, prefix_len + len(suffix)),
                dtype=torch.long,
                device=device,
            )
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=prefix_node.past_key_values,
                use_cache=True,
            )

        logits = output.logits[0, -1, :].detach().cpu().numpy().astype(np.float64)
        target_node.past_key_values = output.past_key_values
        target_node.last_logits = logits
        return logits


class PriorModel:
    """HF prior model + tokenizer mapping to Rust follower-logit order."""

    def __init__(self, lexicographic_tokens: list[str]) -> None:
        _log("PriorModel init start")
        self.device = _pick_device()
        _log("loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _log("loading model weights")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(self.device)
        _log("model loaded and moved to CUDA")
        _log("calling model.eval()")
        self.model.eval()
        _log("model.eval() complete")
        self.cache = PrefixCacheTrie()
        self.initial_token_id = _initial_context_token_id(self.tokenizer)

        _log("building lex-index to tokenizer-id mapping")
        vocab = self.tokenizer.get_vocab()
        self.clean_ids = np.array(
            [vocab[_internal_token_to_hf_token(token)] for token in lexicographic_tokens],
            dtype=np.int64,
        )
        _log("lex-index mapping complete")
        if len(self.clean_ids) != EXPECTED_FILTERED_TOKEN_COUNT:
            raise ValueError(
                f"filtered token count mismatch: expected {EXPECTED_FILTERED_TOKEN_COUNT}, got {len(self.clean_ids)}"
            )

        _log("PriorModel init complete")

    def reset_cache(self) -> None:
        self.cache.reset()

    def _encode_context(self, model_text: str) -> list[int]:
        encoded = self.tokenizer.encode(model_text, add_special_tokens=False)
        if encoded:
            return [int(x) for x in encoded]
        return [self.initial_token_id]

    def prior_update_json_for_request(self, requested_prior_json: str) -> str:
        requested_prior = RequestedPrior(**json.loads(requested_prior_json))
        model_text = _bayes_string_to_model_text(requested_prior.full_string)
        token_ids = self._encode_context(model_text)
        with torch.no_grad():
            last_logits = self.cache.infer_next_logits(token_ids, self.model, self.device)

        follower_logits = last_logits[self.clean_ids]
        payload = {
            "full_string": requested_prior.full_string,
            "final_token_lexindex": requested_prior.last_token_lexindex,
            "follower_logits": follower_logits.tolist(),
        }
        return _json_dumps(payload)


class BackendRuntime:
    def __init__(self) -> None:
        _log("BackendRuntime init start")
        self.session = bayesian.BayesianSession()
        _log("BayesianSession constructed")
        lexicographic_tokens = json.loads(self.session.lexicographic_tokens_json())
        _log(f"loaded lexicographic tokens count={len(lexicographic_tokens)}")
        self.prior_model = PriorModel(lexicographic_tokens)
        self.lock = asyncio.Lock()
        _log("BackendRuntime init complete")

    def reset(self) -> None:
        self.session.reset()
        self.prior_model.reset_cache()

    async def reset_and_emit_prior(self, websocket: WebSocket) -> None:
        async with self.lock:
            self.reset()
            requested_prior_json = self.session.next_requested_prior()
            prior_json = self.prior_model.prior_update_json_for_request(requested_prior_json)
            self.session.receive_prior_update(prior_json)
            self.session.apply_updates()
        await websocket.send_text(_json_dumps({"type": "reset_ack"}))
        await websocket.send_text(_json_dumps({"type": "prior_update", "content_json": prior_json}))

    async def emit_next_prior(self, websocket: WebSocket) -> None:
        async with self.lock:
            requested_prior_json = self.session.next_requested_prior()
            prior_json = self.prior_model.prior_update_json_for_request(requested_prior_json)
            self.session.receive_prior_update(prior_json)
            self.session.apply_updates()
        await websocket.send_text(_json_dumps({"type": "prior_update", "content_json": prior_json}))

    async def apply_likelihood_and_emit_prior(
        self,
        websocket: WebSocket,
        likelihood_json: str,
    ) -> None:
        async with self.lock:
            self.session.receive_likelihood_update(likelihood_json)
            self.session.apply_updates()
            requested_prior_json = self.session.next_requested_prior()
            prior_json = self.prior_model.prior_update_json_for_request(requested_prior_json)
            self.session.receive_prior_update(prior_json)
            self.session.apply_updates()
        await websocket.send_text(_json_dumps({"type": "prior_update", "content_json": prior_json}))


_ensure_singleton_backend_process()

app = FastAPI()
runtime = BackendRuntime()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            message = json.loads(data)
        except starlette.websockets.WebSocketDisconnect:
            break

        msg_type = message.get("type")
        if msg_type == "reset":
            prompt = message.get("content", {}).get("prompt")
            if prompt not in (None, ""):
                _log("received legacy reset prompt; ignoring because priors are trie-driven")
            await runtime.reset_and_emit_prior(websocket)
            continue

        if msg_type == "likelihood_update":
            likelihood_json = message.get("content_json")
            if not isinstance(likelihood_json, str):
                raise TypeError("likelihood_update requires content_json")
            await runtime.apply_likelihood_and_emit_prior(websocket, likelihood_json)
            continue

        if msg_type == "ping":
            await websocket.send_text(
                _json_dumps({"type": "pong", "content": {"pingTime": message.get("pingTime")}})
            )
            continue

        await websocket.send_text(
            _json_dumps(
                {
                    "type": "error",
                    "content": {"message": f"unknown message type: {msg_type}"},
                }
            )
        )


if __name__ == "__main__":
    import uvicorn

    _log("starting uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
