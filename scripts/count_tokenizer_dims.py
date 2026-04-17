#!/usr/bin/env python3
"""Print NUM_TOKENS (BPE piece table size) and NUM_PREFIXES for a Hugging Face tokenizer.json.

Matches Rust `bpe::BpeMerges::from_tokenizer_json_str` (vocab intern order + merge walk) and
`word_tokenizer::token_prefixes` over lex-sorted internal vocab strings (`hf_token_to_internal`
is identity for the filtered tokenizer).

Usage:
  python3 scripts/count_tokenizer_dims.py
  python3 scripts/count_tokenizer_dims.py /path/to/tokenizer.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = REPO_ROOT / "bayesian" / "tokenizers" / "tinyllamaalpha" / "tokenizer.json"


def hf_token_to_internal(s: str) -> str:
    return s


def _char_indices(s: str):
    i = 0
    for ch in s:
        yield i, ch
        i += len(ch.encode("utf-8"))


def token_prefixes_rust(token: str) -> list[str]:
    """Same as Rust `token_prefixes`: `char_indices().skip(1)` then full token if non-empty."""
    prefixes: list[str] = [""]
    it = _char_indices(token)
    next(it, None)  # skip first (matches `.skip(1)`)
    for idx, _ in it:
        prefixes.append(token[:idx])
    if token:
        prefixes.append(token)
    return prefixes


def parse_merge_line(line: str, line_no: int) -> tuple[str, str]:
    if " " not in line:
        raise ValueError(f"bad merge line {line_no}: {line!r}")
    left, right = line.split(" ", 1)
    if " " in right:
        raise ValueError(f"bad merge line {line_no}: {line!r}")
    return left, right


def piece_table_len(content: str) -> int:
    v = json.loads(content)
    model = v["model"]
    piece_to_id: dict[str, int] = {}
    pieces: list[str] = []

    def intern_piece(piece: str) -> int:
        if piece in piece_to_id:
            return piece_to_id[piece]
        i = len(pieces)
        piece_to_id[piece] = i
        pieces.append(piece)
        return i

    def intern_owned_piece(piece: str) -> int:
        return intern_piece(piece)

    vocab = model["vocab"]
    for token in vocab.keys():
        intern_piece(hf_token_to_internal(token))

    merges = model["merges"]
    for idx, item in enumerate(merges):
        line_no = idx + 1
        line = item if isinstance(item, str) else str(item)
        left_s, right_s = parse_merge_line(line, line_no)
        left = hf_token_to_internal(left_s)
        right = hf_token_to_internal(right_s)
        intern_piece(left)
        intern_piece(right)
        merged = left + right
        intern_owned_piece(merged)

    return len(pieces)


def prefix_count(content: str) -> int:
    v = json.loads(content)
    model = v["model"]
    vocab = model["vocab"]
    lex_tokens = sorted(hf_token_to_internal(k) for k in vocab.keys())
    lex_prefixes: list[str] = []
    for token in lex_tokens:
        lex_prefixes.extend(token_prefixes_rust(token))
    lex_prefixes.sort()
    out = []
    for p in lex_prefixes:
        if not out or out[-1] != p:
            out.append(p)
    return len(out)


def _parse_rust_usize_const(src: str, name: str) -> int | None:
    m = re.search(rf"pub const {name}: usize = ([0-9_]+);", src)
    if not m:
        return None
    return int(m.group(1).replace("_", ""))


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    text = path.read_text(encoding="utf-8")
    v = json.loads(text)
    vocab_n = len(v["model"]["vocab"])
    n_tok = piece_table_len(text)
    n_pre = prefix_count(text)
    print(f"path:             {path}")
    print(f"model.vocab keys: {vocab_n}  (piece table is >= this after merges add new strings)")
    print(f"NUM_TOKENS:       {n_tok}  (BpeMerges piece table size)")
    print(f"NUM_PREFIXES:     {n_pre}")
    if n_tok < vocab_n:
        print(
            "error: piece table smaller than vocab — script bug or wrong JSON shape.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg_path = REPO_ROOT / "bayesian" / "crates" / "bpe" / "src" / "tokenizer_config.rs"
    if cfg_path.is_file():
        cfg = cfg_path.read_text(encoding="utf-8")
        rt = _parse_rust_usize_const(cfg, "NUM_TOKENS")
        rp = _parse_rust_usize_const(cfg, "NUM_PREFIXES")
        if rt is not None and rt != n_tok:
            print(
                f"note: tokenizer_config.rs NUM_TOKENS={rt} differs from computed {n_tok} — update the const.",
                file=sys.stderr,
            )
        if rp is not None and rp != n_pre:
            print(
                f"note: tokenizer_config.rs NUM_PREFIXES={rp} differs from computed {n_pre}.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
