#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

DEFAULT_PHRASES_FILE = Path("src/lib/phrases.txt")


def detect_device(device_arg: str) -> str:
    import torch

    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_phrases(path: Path, limit: int | None) -> list[str]:
    phrases = [line.rstrip("\n") for line in path.read_text().splitlines()]
    if limit is not None:
        phrases = phrases[:limit]
    return phrases


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text()
    return args.prompt


def maybe_add_newline(text: str, append_newline: bool) -> str:
    return text + ("\n" if append_newline else "")


def load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda":
        torch_dtype = torch.float16
    elif device == "mps":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def encode_text(tokenizer, text: str, add_special_tokens: bool) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=add_special_tokens)


def get_initial_context_token_id(tokenizer) -> int | None:
    if tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return None


def score_continuation_bits(
    model,
    tokenizer,
    device: str,
    prompt: str,
    continuation: str,
    *,
    add_special_tokens: bool,
) -> tuple[float, int]:
    import torch

    full_ids = encode_text(tokenizer, prompt + continuation, add_special_tokens)
    prompt_ids = encode_text(tokenizer, prompt, add_special_tokens)

    continuation_token_count = len(full_ids) - len(prompt_ids)
    if continuation_token_count <= 0:
        raise ValueError("Continuation did not produce any tokens to score.")

    if len(prompt_ids) == 0:
        initial_token_id = get_initial_context_token_id(tokenizer)
        if initial_token_id is None:
            raise ValueError(
                "Cannot score the first token without context because this tokenizer "
                "has neither a BOS nor EOS token. Try passing a prompt or enabling "
                "special tokens."
            )
        full_ids = [initial_token_id] + full_ids
        prompt_ids = [initial_token_id]

    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]

    target_start = len(prompt_ids)
    target_ids = input_ids[0, target_start:]
    target_logits = logits[target_start - 1 : -1]
    log_probs = torch.log_softmax(target_logits, dim=-1)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    total_bits = float((-token_log_probs.sum() / math.log(2)).item())
    return total_bits, continuation_token_count


def benchmark_phrases(
    model,
    tokenizer,
    device: str,
    phrases: Sequence[str],
    *,
    prompt: str,
    append_newline: bool,
    add_special_tokens: bool,
    verbose: bool,
) -> dict[str, float | int | str]:
    total_bits = 0.0
    total_chars = 0
    total_tokens = 0

    for index, phrase in enumerate(phrases, start=1):
        continuation = maybe_add_newline(phrase, append_newline)
        phrase_bits, phrase_tokens = score_continuation_bits(
            model,
            tokenizer,
            device,
            prompt,
            continuation,
            add_special_tokens=add_special_tokens,
        )
        total_bits += phrase_bits
        total_chars += len(continuation)
        total_tokens += phrase_tokens

        if verbose and (index == 1 or index % 50 == 0 or index == len(phrases)):
            print(
                f"Scored {index}/{len(phrases)} phrases "
                f"({total_bits / max(total_chars, 1):.4f} bits/char so far)"
            )

    return {
        "phrase_count": len(phrases),
        "prompt_chars": len(prompt),
        "total_bits": total_bits,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "bits_per_character": total_bits / total_chars,
        "bits_per_token": total_bits / total_tokens,
        "avg_bits_per_phrase": total_bits / len(phrases),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a causal language model on the repo phrase set."
    )
    parser.add_argument("model", help="Hugging Face model id or local model path")
    parser.add_argument(
        "--phrases-file",
        default=str(DEFAULT_PHRASES_FILE),
        help=f"Phrase file to score (default: {DEFAULT_PHRASES_FILE})",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional prompt prepended before each phrase",
    )
    parser.add_argument(
        "--prompt-file",
        help="Read the prompt from a file instead of --prompt",
    )
    parser.add_argument(
        "--append-newline",
        action="store_true",
        help="Append a newline to each phrase before scoring it",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only score the first N phrases",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Encode with tokenizer special tokens enabled",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print periodic progress updates",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.prompt and args.prompt_file:
        parser.error("Pass either --prompt or --prompt-file, not both.")

    phrases_path = Path(args.phrases_file)
    if not phrases_path.exists():
        parser.error(f"Phrases file not found: {phrases_path}")

    prompt = load_prompt(args)
    phrases = read_phrases(phrases_path, args.limit)
    if not phrases:
        parser.error("No phrases found to score.")

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        parser.error(
            "This script requires `torch` and `transformers`. "
            "Install the repo requirements first, for example: "
            "`pip install -r requirements.txt`."
        )

    device = detect_device(args.device)
    print(f"Loading model: {args.model}")
    print(f"Using device: {device}")
    tokenizer, model = load_model(args.model, device)

    results = benchmark_phrases(
        model,
        tokenizer,
        device,
        phrases,
        prompt=prompt,
        append_newline=args.append_newline,
        add_special_tokens=args.add_special_tokens,
        verbose=args.verbose,
    )

    print()
    print("Benchmark results")
    print(f"model: {args.model}")
    print(f"phrases_file: {phrases_path}")
    print(f"phrase_count: {results['phrase_count']}")
    print(f"prompt_chars: {results['prompt_chars']}")
    print(f"append_newline: {args.append_newline}")
    print(f"add_special_tokens: {args.add_special_tokens}")
    print(f"total_chars: {results['total_chars']}")
    print(f"total_tokens: {results['total_tokens']}")
    print(f"total_bits: {results['total_bits']:.4f}")
    print(f"bits_per_character: {results['bits_per_character']:.6f}")
    print(f"bits_per_token: {results['bits_per_token']:.6f}")
    print(f"avg_bits_per_phrase: {results['avg_bits_per_phrase']:.6f}")


if __name__ == "__main__":
    main()
