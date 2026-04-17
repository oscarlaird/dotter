#!/usr/bin/env python3
"""Filter the TinyLlama BPE tokenizer down to the dotter trie alphabet.

Reads the raw HF tokenizer.json (downloaded or cached), keeps only tokens whose
surface form (after sentinel replacement) matches the trie alphabet, reindexes
the vocabulary to a dense 0..n-1 range, and writes the filtered tokenizer.json
that the Rust `bpe` crate embeds at compile time.

Sentinel replacement (applied to every token string and merge piece):
  - SentencePiece `▁` (U+2581) → `S`  (trie word-boundary)

The trie stop symbol `Z` is **not** derived from the HF `$` token (that was the old wire convention).
We append a synthetic `Z` entry to the filtered vocabulary so the Rust trie and backend can agree on
`STOP_MARKER == "Z"` while leaving `$` free for future real punctuation in the payload alphabet.

Run from the repo root:
    python3 scripts/filter_bpe_tokenizer.py
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HF_SOURCE_TOKENIZER = Path("/tmp/tinyllama_tokenizer/tokenizer.json")

OUTPUT_DIR = REPO_ROOT / "bayesian" / "tokenizers" / "tinyllamaalpha"

HF_SPACE = "\u2581"
TRIE_SPACE = "S"
TRIE_STOP = "Z"

TOKEN_PATTERN = re.compile(r"[a-z,.' ]+")
MAX_SPACES = 1


def sentinel_replace(s: str) -> str:
    return s.replace(HF_SPACE, TRIE_SPACE)


def normalize_for_match(token: str) -> str:
    return token.replace(TRIE_SPACE, " ")


def count_spaces(token: str) -> int:
    return normalize_for_match(token).count(" ")


def load_source(path: Path) -> tuple[dict, dict[str, int], list[tuple[str, str]]]:
    source_json = json.loads(path.read_text(encoding="utf-8"))
    model = source_json["model"]
    vocab: dict[str, int] = {str(k): int(v) for k, v in model["vocab"].items()}
    raw_merges = model["merges"]
    merges: list[tuple[str, str]] = []
    for entry in raw_merges:
        if isinstance(entry, list):
            merges.append((str(entry[0]), str(entry[1])))
        elif isinstance(entry, str):
            i = entry.index(" ")
            merges.append((entry[:i], entry[i + 1 :]))
        else:
            raise ValueError(f"unexpected merge entry type: {type(entry)}")
    return source_json, vocab, merges


def filter_and_reindex(
    vocab: dict[str, int],
    merges: list[tuple[str, str]],
) -> tuple[dict[str, int], list[str]]:
    mapped_vocab = {sentinel_replace(tok): orig_id for tok, orig_id in vocab.items()}
    mapped_merges = [(sentinel_replace(l), sentinel_replace(r)) for l, r in merges]

    legal = {
        tok
        for tok in mapped_vocab
        if TOKEN_PATTERN.fullmatch(normalize_for_match(tok)) is not None
        and count_spaces(tok) <= MAX_SPACES
    }

    reachable = {tok for tok in legal if len(tok) == 1}
    kept_merges: list[str] = []
    seen_merge_pairs: set[tuple[str, str]] = set()
    for left, right in mapped_merges:
        merged = left + right
        if (left, right) in seen_merge_pairs:
            continue
        if left in reachable and right in reachable and merged in legal:
            kept_merges.append(f"{left} {right}")
            seen_merge_pairs.add((left, right))
            reachable.add(merged)

    id_ordered = [
        tok
        for tok, _ in sorted(mapped_vocab.items(), key=lambda kv: kv[1])
        if tok in reachable
    ]
    final_vocab = {tok: idx for idx, tok in enumerate(id_ordered)}
    if TRIE_STOP in final_vocab:
        raise SystemExit(
            f"synthetic trie stop {TRIE_STOP!r} already present in filtered vocab "
            "(unexpected collision with HF-derived tokens)"
        )
    final_vocab[TRIE_STOP] = len(final_vocab)
    return final_vocab, kept_merges


def build_output_json(source_json: dict, vocab: dict[str, int], merges: list[str]) -> dict:
    model = source_json.get("model", {})
    return {
        "version": source_json.get("version", "1.0"),
        "truncation": copy.deepcopy(source_json.get("truncation")),
        "padding": copy.deepcopy(source_json.get("padding")),
        "added_tokens": [],
        "normalizer": copy.deepcopy(source_json.get("normalizer")),
        "pre_tokenizer": copy.deepcopy(source_json.get("pre_tokenizer")),
        "post_processor": None,
        "decoder": copy.deepcopy(source_json.get("decoder")),
        "model": {
            "type": model.get("type", "BPE"),
            "dropout": copy.deepcopy(model.get("dropout")),
            "unk_token": None,
            "continuing_subword_prefix": copy.deepcopy(model.get("continuing_subword_prefix")),
            "end_of_word_suffix": copy.deepcopy(model.get("end_of_word_suffix")),
            "fuse_unk": bool(model.get("fuse_unk")),
            "byte_fallback": bool(model.get("byte_fallback")),
            "vocab": vocab,
            "merges": merges,
        },
    }


def main() -> None:
    if not HF_SOURCE_TOKENIZER.exists():
        raise SystemExit(
            f"Source tokenizer not found at {HF_SOURCE_TOKENIZER}\n"
            "Download it first:\n"
            "  python3 -c \"\n"
            "from transformers import AutoTokenizer\n"
            "t = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')\n"
            "t.save_pretrained('/tmp/tinyllama_tokenizer')\n"
            "\""
        )

    source_json, vocab, merges = load_source(HF_SOURCE_TOKENIZER)
    final_vocab, final_merges = filter_and_reindex(vocab, merges)

    if not final_vocab:
        raise SystemExit("filter removed every token")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_json = build_output_json(source_json, final_vocab, final_merges)
    tokenizer_path = OUTPUT_DIR / "tokenizer.json"
    with tokenizer_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
        f.write("\n")

    summary = {
        "token_pattern": TOKEN_PATTERN.pattern,
        "sentinel_replace": {HF_SPACE: TRIE_SPACE},
        "synthetic_stop_token": TRIE_STOP,
        "max_spaces": MAX_SPACES,
        "source_vocab_size": len(vocab),
        "source_merge_count": len(merges),
        "final_vocab_size": len(final_vocab),
        "final_merge_count": len(final_merges),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"source_vocab_size={len(vocab)}")
    print(f"source_merge_count={len(merges)}")
    print(f"final_vocab_size={len(final_vocab)}")
    print(f"final_merge_count={len(final_merges)}")
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
