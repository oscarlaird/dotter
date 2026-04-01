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
PIDFILE_PATH = Path("/tmp/dotter_new_lm.pid")
START_TS = time.monotonic()


def _log(msg: str) -> None:
    elapsed = time.monotonic() - START_TS
    print(f"[new_lm +{elapsed:8.3f}s] {msg}", file=sys.stderr, flush=True)


def _pick_device() -> str:
    # CUDA-only by design: never add CPU fallback here.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for backend runtime; CPU fallback is forbidden.")
    return "cuda"


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
        _log("PrefixCacheTrie initialized")

        self.clean_tokens = lexicographic_tokens
        _log("building clean_ids mapping")
        vocab = self.tokenizer.get_vocab()
        self.clean_ids = np.array(
            [vocab[token] for token in self.clean_tokens],
            dtype=np.int64,
        )
        _log("clean_ids mapping complete")
        if len(self.clean_ids) != EXPECTED_FILTERED_TOKEN_COUNT:
            raise ValueError(
                f"filtered token count mismatch: expected {EXPECTED_FILTERED_TOKEN_COUNT}, got {len(self.clean_ids)}"
            )

        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer eos_token_id is required for stop logit extraction")
        self.stop_token_id = int(self.tokenizer.eos_token_id)
        _log("PriorModel init complete")

    def reset_cache(self) -> None:
        self.cache.reset()

    def _encode(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        if len(encoded) == 0:
            raise ValueError("tokenizer produced an empty token sequence")
        return [int(x) for x in encoded]

    def prior_update_for_string(
        self,
        full_string: str,
    ) -> tuple[str | None, str, np.ndarray, float]:
        token_ids = self._encode(full_string)
        with torch.no_grad():
            last_logits = self.cache.infer_next_logits(token_ids, self.model, self.device)

        follower_logits = last_logits[self.clean_ids]
        stop_logit = float(last_logits[self.stop_token_id])

        # For now we let Rust infer canonical support from full string.
        final_token = None
        return final_token, full_string, follower_logits, stop_logit


def _symbol_to_char(symbol: str) -> str:
    if symbol == "Space":
        return " "
    if symbol == "Stop":
        return "$"
    if symbol == "Start":
        return ""
    if len(symbol) == 1 and "A" <= symbol <= "Z":
        return symbol.lower()
    return ""


def top_snapshot_strings(snapshot_json: str, max_items: int) -> list[str]:
    snapshot = json.loads(snapshot_json)
    nodes: list[dict[str, Any]] = snapshot["nodes"]
    root = int(snapshot["root"])
    stack: list[tuple[int, str]] = [(root, "")]
    scored: list[tuple[float, str]] = []

    while stack:
        node_index, prefix = stack.pop()
        node = nodes[node_index]
        symbol = _symbol_to_char(node["symbol"])
        value = prefix + symbol
        if "$" not in value and symbol != "":
            scored.append((float(node["z"]), value))
        for _child_symbol, child_idx in node["children"]:
            stack.append((int(child_idx), value))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for _, text in scored:
        if text in seen:
            continue
        out.append(text)
        seen.add(text)
        if len(out) >= max_items:
            break
    return out


class BackendRuntime:
    def __init__(self) -> None:
        _log("BackendRuntime init start")
        self.prompt = ""
        self.session = bayesian.BayesianSession()
        _log("BayesianSession constructed")
        lexicographic_tokens = json.loads(self.session.lexicographic_tokens_json())
        _log(f"loaded lexicographic tokens count={len(lexicographic_tokens)}")
        self.prior_model = PriorModel(lexicographic_tokens)
        self.lock = asyncio.Lock()
        _log("BackendRuntime init complete")

    def reset(self, prompt: str = "") -> None:
        self.prompt = prompt
        self.session.reset()
        self.prior_model.reset_cache()

    async def apply_likelihood_and_emit_priors(
        self,
        websocket: WebSocket,
        snapshot_json: str,
    ) -> None:
        async with self.lock:
            self.session.apply_likelihood_update(snapshot_json)
            if not self.prompt:
                raise ValueError("reset prompt must be non-empty before likelihood updates")
            final_token, full_string, follower_logits, stop_logit = (
                self.prior_model.prior_update_for_string(self.prompt)
            )
            content = {
                "final_token": final_token,
                "full_string": full_string,
                "follower_logits": follower_logits.tolist(),
                "stop_logit": stop_logit,
            }
            self.session.apply_prior_update(json.dumps(content))
            payload = {"type": "prior_update", "content": content}
            await websocket.send_text(json.dumps(payload))


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
            prompt = str(message.get("content", {}).get("prompt", ""))
            runtime.reset(prompt)
            await websocket.send_text(json.dumps({"type": "reset_ack"}))
            continue

        if msg_type == "likelihood_update":
            content = message.get("content", {})
            snapshot_json = content.get("snapshot_json")
            if not isinstance(snapshot_json, str):
                raise TypeError("likelihood_update requires snapshot_json")
            await runtime.apply_likelihood_and_emit_priors(websocket, snapshot_json)
            continue

        if msg_type == "ping":
            await websocket.send_text(
                json.dumps({"type": "pong", "content": {"pingTime": message.get("pingTime")}})
            )
            continue

        await websocket.send_text(
            json.dumps(
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
