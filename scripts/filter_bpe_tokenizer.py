#!/usr/bin/env python3
"""Filter a Hugging Face BPE tokenizer.json down to a legal token subset.

It keeps only tokens whose full surface form matches a supplied regex
(``[a-z]+`` by default). When requested, a designated source symbol such as
SentencePiece ``▁`` can be treated as a literal space for matching while still
being written back unchanged in the output tokenizer files.

The utility retains only merge rules that can still be built from the surviving
single-character pieces, reindexes the filtered vocabulary to a dense ``0..n-1``
range, and writes a filtered ``tokenizer.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceTokenizer:
    source_json: dict
    vocab: dict[str, int]
    merges: list[str]


@dataclass(frozen=True)
class FilteredTokenizer:
    vocab: dict[str, int]
    merges: list[str]
    kept_token_count: int
    kept_merge_count: int
    reachable_token_count: int
    dropped_token_count: int
    dropped_merge_count: int


def normalize_for_match(token: str, space_symbol: str | None) -> str:
    if not space_symbol:
        return token
    return token.replace(space_symbol, " ")


def count_spaces_for_filter(token: str, space_symbol: str | None) -> int:
    return normalize_for_match(token, space_symbol).count(" ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a Hugging Face BPE tokenizer.json to a regex-defined legal "
            "token set and emit a new tokenizer.json."
        )
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        required=True,
        help="Input Hugging Face tokenizer.json path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write tokenizer.json and optional summary.json.",
    )
    parser.add_argument(
        "--token-pattern",
        default=r"[a-z]+",
        help="Regex that every retained token must match exactly. Default: [a-z]+",
    )
    parser.add_argument(
        "--space-symbol",
        default=None,
        help=(
            "Optional source token symbol to treat as a literal space for regex "
            "matching, e.g. ▁ for SentencePiece-style tokenizers."
        ),
    )
    parser.add_argument(
        "--max-spaces",
        type=int,
        default=None,
        help=(
            "Optional maximum number of spaces allowed in any retained token "
            "surface after space-symbol normalization."
        ),
    )
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="Also write a summary.json file with filtering statistics.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer_source(args: argparse.Namespace) -> SourceTokenizer:
    source_json = load_json(args.tokenizer_json)
    model = source_json.get("model")
    if not isinstance(model, dict):
        raise ValueError("tokenizer.json is missing a model object")
    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict):
        raise ValueError("tokenizer.json is missing model.vocab")
    if not isinstance(merges, list):
        raise ValueError("tokenizer.json is missing model.merges")
    return SourceTokenizer(
        source_json=source_json,
        vocab=coerce_vocab(vocab),
        merges=coerce_merges(merges),
    )


def coerce_vocab(raw_vocab: dict) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for token, token_id in raw_vocab.items():
        if not isinstance(token, str):
            raise ValueError("vocab contains a non-string token")
        if not isinstance(token_id, int):
            raise ValueError(f"vocab id for {token!r} is not an integer")
        vocab[token] = token_id
    if len(set(vocab.values())) != len(vocab):
        raise ValueError("vocab ids must be unique")
    return vocab


def coerce_merges(raw_merges: list) -> list[str]:
    merges: list[str] = []
    for merge in raw_merges:
        if not isinstance(merge, str):
            raise ValueError("merges array contains a non-string entry")
        merges.append(merge)
    return merges


def parse_merge_line(line: str) -> tuple[str, str]:
    separator = line.find(" ")
    if separator <= 0 or separator == len(line) - 1:
        raise ValueError(
            f"invalid merge line {line!r}: expected two pieces separated by one space"
        )
    return line[:separator], line[separator + 1 :]


def sorted_tokens_by_original_id(vocab: dict[str, int]) -> list[str]:
    return [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]


def filter_tokenizer(
    source: SourceTokenizer,
    token_pattern: str,
    space_symbol: str | None,
    max_spaces: int | None,
) -> FilteredTokenizer:
    matcher = re.compile(token_pattern)
    legal_tokens = {
        token
        for token in source.vocab
        if matcher.fullmatch(normalize_for_match(token, space_symbol)) is not None
        and (max_spaces is None or count_spaces_for_filter(token, space_symbol) <= max_spaces)
    }

    reachable_tokens = {
        token for token in legal_tokens if len(token) == 1
    }
    kept_merges: list[str] = []

    for merge_line in source.merges:
        left, right = parse_merge_line(merge_line)
        merged = left + right
        if left not in reachable_tokens:
            continue
        if right not in reachable_tokens:
            continue
        if merged not in legal_tokens:
            continue
        kept_merges.append(merge_line)
        reachable_tokens.add(merged)

    final_tokens_in_id_order = [
        token
        for token in sorted_tokens_by_original_id(source.vocab)
        if token in reachable_tokens
    ]
    final_vocab = {
        token: new_id for new_id, token in enumerate(final_tokens_in_id_order)
    }

    return FilteredTokenizer(
        vocab=final_vocab,
        merges=kept_merges,
        kept_token_count=len(legal_tokens),
        kept_merge_count=len(kept_merges),
        reachable_token_count=len(reachable_tokens),
        dropped_token_count=len(source.vocab) - len(final_vocab),
        dropped_merge_count=len(source.merges) - len(kept_merges),
    )


def build_tokenizer_json(
    source_json: dict,
    vocab: dict[str, int],
    merges: list[str],
) -> dict:
    version = source_json.get("version", "1.0") if isinstance(source_json, dict) else "1.0"
    truncation = copy.deepcopy(source_json.get("truncation")) if isinstance(source_json, dict) else None
    padding = copy.deepcopy(source_json.get("padding")) if isinstance(source_json, dict) else None
    normalizer = copy.deepcopy(source_json.get("normalizer")) if isinstance(source_json, dict) else None
    pre_tokenizer = copy.deepcopy(source_json.get("pre_tokenizer")) if isinstance(source_json, dict) else None
    decoder = copy.deepcopy(source_json.get("decoder")) if isinstance(source_json, dict) else None
    model = source_json.get("model") if isinstance(source_json, dict) else None
    model_type = model.get("type", "BPE") if isinstance(model, dict) else "BPE"
    dropout = copy.deepcopy(model.get("dropout")) if isinstance(model, dict) else None
    continuing_subword_prefix = (
        copy.deepcopy(model.get("continuing_subword_prefix")) if isinstance(model, dict) else None
    )
    end_of_word_suffix = (
        copy.deepcopy(model.get("end_of_word_suffix")) if isinstance(model, dict) else None
    )
    fuse_unk = bool(model.get("fuse_unk")) if isinstance(model, dict) else False
    byte_fallback = bool(model.get("byte_fallback")) if isinstance(model, dict) else False
    return {
        "version": version,
        "truncation": truncation,
        "padding": padding,
        "added_tokens": [],
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": None,
        "decoder": decoder,
        "model": {
            "type": model_type,
            "dropout": dropout,
            "unk_token": None,
            "continuing_subword_prefix": continuing_subword_prefix,
            "end_of_word_suffix": end_of_word_suffix,
            "fuse_unk": fuse_unk,
            "byte_fallback": byte_fallback,
            "vocab": vocab,
            "merges": merges,
        },
    }


def write_tokenizer_json(path: Path, tokenizer_json: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_summary_json(
    path: Path,
    *,
    source: SourceTokenizer,
    filtered: FilteredTokenizer,
    token_pattern: str,
    space_symbol: str | None,
    max_spaces: int | None,
) -> None:
    summary = {
        "token_pattern": token_pattern,
        "space_symbol": space_symbol,
        "max_spaces": max_spaces,
        "source_vocab_size": len(source.vocab),
        "source_merge_count": len(source.merges),
        "pattern_matched_vocab_size": filtered.kept_token_count,
        "reachable_vocab_size": filtered.reachable_token_count,
        "final_vocab_size": len(filtered.vocab),
        "final_merge_count": len(filtered.merges),
        "dropped_vocab_size": filtered.dropped_token_count,
        "dropped_merge_count": filtered.dropped_merge_count,
    }
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    if args.max_spaces is not None and args.max_spaces < 0:
        raise SystemExit("--max-spaces must be nonnegative")

    source = load_tokenizer_source(args)
    filtered = filter_tokenizer(
        source,
        args.token_pattern,
        args.space_symbol,
        args.max_spaces,
    )
    if not filtered.vocab:
        raise SystemExit("filter removed every vocab token")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = output_dir / "tokenizer.json"

    tokenizer_json = build_tokenizer_json(
        source.source_json,
        filtered.vocab,
        filtered.merges,
    )

    write_tokenizer_json(tokenizer_path, tokenizer_json)

    if args.summary_json:
        write_summary_json(
            output_dir / "summary.json",
            source=source,
            filtered=filtered,
            token_pattern=args.token_pattern,
            space_symbol=args.space_symbol,
            max_spaces=args.max_spaces,
        )

    print(f"source_vocab_size={len(source.vocab)}")
    print(f"source_merge_count={len(source.merges)}")
    print(f"pattern_matched_vocab_size={filtered.kept_token_count}")
    print(f"reachable_vocab_size={filtered.reachable_token_count}")
    print(f"final_vocab_size={len(filtered.vocab)}")
    print(f"final_merge_count={len(filtered.merges)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
