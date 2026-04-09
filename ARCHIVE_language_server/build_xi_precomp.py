#!/usr/bin/env python3
"""Build a precomputed xi lookup table for clean TinyLlama tokens.

Outputs three files in the chosen directory:

- tokens.txt   : lexicographically ordered clean tokens (one per line)
- prefixes.txt : lexicographically ordered token prefixes (one per line)
- xi.bits      : row-major packed bits for xi(token, prefix)

The clean-token construction matches ``lm.py``:
- special tokens removed
- only lowercase letters plus SentencePiece ``▁``
- ``▁`` normalized to a literal space in stored strings
"""

from __future__ import annotations

import argparse
import bisect
import random
import string
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from transformers import AutoTokenizer

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "precomp" / "xi"


def build_clean_vocab(tokenizer) -> dict[str, tuple[str, int]]:
    clean_vocab: dict[str, tuple[str, int]] = {}
    for raw_token, token_id in tokenizer.vocab.items():
        if raw_token in tokenizer.all_special_tokens:
            continue
        if not all(c in string.ascii_lowercase or c == "▁" for c in raw_token):
            continue
        clean_token = raw_token.replace("▁", " ")
        prev = clean_vocab.get(clean_token)
        if prev is not None and prev != (raw_token, token_id):
            raise ValueError(
                f"clean token collision for {clean_token!r}: "
                f"{prev!r} vs {(raw_token, token_id)!r}"
            )
        clean_vocab[clean_token] = (raw_token, token_id)
    return clean_vocab


def sorted_clean_tokens_raw_and_ids(tokenizer) -> tuple[list[str], list[str], list[int]]:
    clean_vocab = build_clean_vocab(tokenizer)
    items = sorted(clean_vocab.items())
    clean_tokens = [clean for clean, _ in items]
    raw_tokens = [raw for _, (raw, _) in items]
    clean_ids = [token_id for _, (_, token_id) in items]
    return clean_tokens, raw_tokens, clean_ids


def sorted_prefixes(tokens: list[str]) -> list[str]:
    return sorted({token[:i] for token in tokens for i in range(len(token) + 1)})


def get_prefix_range(prefix: str, tokens: list[str]) -> tuple[int, int]:
    if prefix == "":
        return 0, len(tokens)
    next_prefix = prefix[:-1] + chr(ord(prefix[-1]) + 1)
    return (
        bisect.bisect_left(tokens, prefix),
        bisect.bisect_left(tokens, next_prefix),
    )


def build_token_prefix_masks(tokens: list[str], prefixes: list[str]) -> np.ndarray:
    n_prefixes = len(prefixes)
    row_nbytes = (n_prefixes + 7) // 8
    prefix_to_col = {prefix: col for col, prefix in enumerate(prefixes)}
    masks = np.zeros((len(tokens), row_nbytes), dtype=np.uint8)

    for token_index, token in enumerate(tokens):
        row = masks[token_index]
        for i in range(len(token) + 1):
            col = prefix_to_col[token[:i]]
            row[col // 8] |= 1 << (col % 8)

    return masks


def compute_canonical_flags(
    raw_tokenizer: Tokenizer,
    left_raw_token: str,
    left_id: int,
    raw_tokens: list[str],
    token_ids: list[int],
    chunk_size: int,
) -> np.ndarray:
    flags = np.zeros(len(raw_tokens), dtype=bool)

    for start in range(0, len(raw_tokens), chunk_size):
        stop = min(start + chunk_size, len(raw_tokens))
        chunk_raw_tokens = raw_tokens[start:stop]
        texts = [left_raw_token + right_raw_token for right_raw_token in chunk_raw_tokens]
        encoded = raw_tokenizer.encode_batch(texts)

        for offset, enc in enumerate(encoded):
            ids = enc.ids
            if len(ids) != 2:
                continue
            if ids[0] == left_id and ids[1] == token_ids[start + offset]:
                flags[start + offset] = True

    return flags


def direct_xi(
    raw_tokenizer: Tokenizer,
    left_raw_token: str,
    left_id: int,
    prefix: str,
    tokens: list[str],
    raw_tokens: list[str],
    token_ids: list[int],
    chunk_size: int,
) -> bool:
    start, stop = get_prefix_range(prefix, tokens)
    if start == stop:
        return False
    flags = compute_canonical_flags(
        raw_tokenizer,
        left_raw_token,
        left_id,
        raw_tokens[start:stop],
        token_ids[start:stop],
        chunk_size,
    )
    return bool(flags.any())


def write_lines(path: Path, items: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(item)
            f.write("\n")


def build_xi_precomp(
    *,
    model_name: str,
    output_dir: Path,
    limit_tokens: int | None,
    chunk_size: int,
    validate_samples: int,
    progress_every: int,
) -> None:
    t0 = time.time()
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_tokenizer = Tokenizer(hf_tokenizer.backend_tokenizer.model)
    clean_tokens, raw_tokens, clean_ids = sorted_clean_tokens_raw_and_ids(hf_tokenizer)

    if limit_tokens is not None:
        clean_tokens = clean_tokens[:limit_tokens]
        raw_tokens = raw_tokens[:limit_tokens]
        clean_ids = clean_ids[:limit_tokens]

    prefixes = sorted_prefixes(clean_tokens)
    token_prefix_masks = build_token_prefix_masks(clean_tokens, prefixes)
    row_nbytes = token_prefix_masks.shape[1]
    total_bits = len(clean_tokens) * len(prefixes)
    total_bytes = len(clean_tokens) * row_nbytes
    total_mib = total_bytes / (1024 * 1024)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / "tokens.txt"
    prefixes_path = output_dir / "prefixes.txt"
    bits_path = output_dir / "xi.bits"

    write_lines(tokens_path, clean_tokens)
    write_lines(prefixes_path, prefixes)

    print("Starting xi precompute")
    print(f"model: {model_name}")
    print(f"token_count: {len(clean_tokens)}")
    print(f"prefix_count: {len(prefixes)}")
    print(f"matrix_bits: {total_bits}")
    print(f"matrix_bytes: {total_bytes}")
    print(f"matrix_mib: {total_mib:.2f}")
    print(f"chunk_size: {chunk_size}")
    print(f"progress_every: {progress_every}")
    print(f"output_dir: {output_dir}")
    print(flush=True)

    sample_row_indices: list[int] = []
    sample_prefix_indices: list[int] = []
    if validate_samples > 0 and clean_tokens and prefixes:
        rng = random.Random(0)
        sample_row_indices = sorted(
            set(
                [0, len(clean_tokens) // 2, len(clean_tokens) - 1]
                + [rng.randrange(len(clean_tokens)) for _ in range(validate_samples)]
            )
        )
        sample_prefix_indices = sorted(
            set(
                [0, len(prefixes) // 2, len(prefixes) - 1]
                + [rng.randrange(len(prefixes)) for _ in range(validate_samples)]
            )
        )

    sampled_rows: dict[int, bytes] = {}

    with bits_path.open("wb") as f:
        for row_index, (left_raw_token, left_id) in enumerate(zip(raw_tokens, clean_ids, strict=True)):
            flags = compute_canonical_flags(
                raw_tokenizer,
                left_raw_token,
                left_id,
                raw_tokens,
                clean_ids,
                chunk_size,
            )
            if flags.any():
                row_bytes = np.bitwise_or.reduce(token_prefix_masks[flags], axis=0)
            else:
                row_bytes = np.zeros(row_nbytes, dtype=np.uint8)
            row_blob = row_bytes.tobytes()
            f.write(row_blob)

            if row_index in sample_row_indices:
                sampled_rows[row_index] = row_blob

            rows_done = row_index + 1
            if row_index == 0 or rows_done % progress_every == 0 or rows_done == len(clean_tokens):
                elapsed = time.time() - t0
                rate = rows_done / elapsed if elapsed > 0 else float("inf")
                remaining_rows = len(clean_tokens) - rows_done
                eta_sec = remaining_rows / rate if rate > 0 else float("inf")
                pct = 100.0 * rows_done / len(clean_tokens)
                print(
                    f"[{rows_done}/{len(clean_tokens)}] "
                    f"{pct:5.1f}% "
                    f"elapsed={elapsed:.1f}s "
                    f"rate={rate:.1f} rows/s "
                    f"eta={eta_sec:.1f}s",
                    flush=True,
                )

    if validate_samples > 0 and sampled_rows:
        for row_index in sample_row_indices:
            left_raw_token = raw_tokens[row_index]
            left_id = clean_ids[row_index]
            row_blob = sampled_rows[row_index]
            for prefix_index in sample_prefix_indices:
                prefix = prefixes[prefix_index]
                matrix_value = bool((row_blob[prefix_index // 8] >> (prefix_index % 8)) & 1)
                direct_value = direct_xi(
                    raw_tokenizer,
                    left_raw_token,
                    left_id,
                    prefix,
                    clean_tokens,
                    raw_tokens,
                    clean_ids,
                    chunk_size,
                )
                assert matrix_value == direct_value, (
                    f"validation failed for token={left_token!r}, prefix={prefix!r}: "
                    f"matrix={matrix_value}, direct={direct_value}"
                )

    elapsed = time.time() - t0
    print()
    print("Xi precompute summary")
    print(f"model: {model_name}")
    print(f"token_count: {len(clean_tokens)}")
    print(f"prefix_count: {len(prefixes)}")
    print(f"matrix_bits: {total_bits}")
    print(f"matrix_bytes: {total_bytes}")
    print(f"row_bytes: {row_nbytes}")
    print(f"output_dir: {output_dir}")
    print(f"tokens_path: {tokens_path}")
    print(f"prefixes_path: {prefixes_path}")
    print(f"bits_path: {bits_path}")
    print(f"elapsed_sec: {elapsed:.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the lexicographically ordered xi precompute files."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id or local tokenizer path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for tokens.txt / prefixes.txt / xi.bits (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--limit-tokens",
        type=int,
        help="Only use the first N lexicographically ordered clean tokens (useful for smoke tests).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Batch size for tokenizer calls when checking canonical pairs.",
    )
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=8,
        help="Number of random sample rows/prefixes to validate against direct xi evaluation.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print a progress update every N completed rows.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_xi_precomp(
        model_name=args.model,
        output_dir=args.output_dir,
        limit_tokens=args.limit_tokens,
        chunk_size=args.chunk_size,
        validate_samples=args.validate_samples,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
