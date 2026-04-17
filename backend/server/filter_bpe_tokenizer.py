#!/usr/bin/env python3
"""
Filter the TinyLlama Vocabulary down to Dotter's Vocabulary
and Encode capital letters and other special characters in Dotter's prefix system.

Reads the raw HF tokenizer.json (downloaded or cached), keeps only tokens
which can be mapped to trie tokens
and performs the encoding into trie tokens.

reindexes the filtered vocabulary to a dense 0..n-1 range, and writes the filtered tokenizer.json
that the Rust `bpe` crate embeds at compile time.

adds synthetic trie start and trie stop tokens to the filtered vocabulary.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from . import token_mapping as tm

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

HF_SOURCE_TOKENIZER = Path("/tmp/tinyllama_tokenizer/tokenizer.json")

OUTPUT_DIR = REPO_ROOT / "bayesian" / "tokenizers" / "tinyllamaalpha"

HF_TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def parse_source(source_json: dict) -> tuple[dict, dict[str, int], list[tuple[str, str]]]:
    model = source_json["model"]
    orig_vocab: dict[str, int] = {str(k): int(v) for k, v in model["vocab"].items()}
    raw_merges = model["merges"]
    orig_merges: list[tuple[str, str]] = []
    for entry in raw_merges:
        # if isinstance(entry, list):
        #     merges.append((str(entry[0]), str(entry[1])))
        if isinstance(entry, str):
            i = entry.index(" ")
            orig_merges.append((entry[:i], entry[i + 1 :]))
        else:
            raise ValueError(f"unexpected merge entry type: {type(entry)}")
    return orig_vocab, orig_merges


def filter_and_reindex(
    hf_vocab: dict[str, int],
    hf_merges: list[tuple[str, str]],
) -> tuple[dict[str, int], list[str]]:
    mappable_hf_vocab = {tok:id for tok,id in hf_vocab.items() if tm.has_trie_mapping(tok)}
    mappable_hf_merges = [(tok1,tok2) for tok1,tok2 in hf_merges if tm.has_trie_mapping(tok1) and tm.has_trie_mapping(tok2)]
    trie_tok_to_hf_id = {tm.hf_token_to_trie_token(hf_tok): hf_id for hf_tok,hf_id in mappable_hf_vocab.items()}
    trie_merges = [(tm.hf_token_to_trie_token(l), tm.hf_token_to_trie_token(r)) for l, r in mappable_hf_merges]

    #
    reindexed_trie_vocab = {tok: i for i, tok in enumerate(sorted(trie_tok_to_hf_id.keys()))}

    if tm.TRIE_STOP in reindexed_trie_vocab or tm.TRIE_START in reindexed_trie_vocab:
        raise SystemExit(
            f"Collision between HF-derived tokens and trie control characters: {tm.TRIE_STOP!r} or {tm.TRIE_START!r}"
        )
    # no start/stop in the vocabulary
    assert tm.TRIE_START not in reindexed_trie_vocab
    assert tm.TRIE_STOP not in reindexed_trie_vocab
    # add start/stop
    reindexed_trie_vocab[tm.TRIE_START] = len(reindexed_trie_vocab)
    reindexed_trie_vocab[tm.TRIE_STOP] = len(reindexed_trie_vocab)
    # no symbols in the vocabulary
    assert tm.TRIE_NUMPAD not in reindexed_trie_vocab
    assert tm.TRIE_SHIFT not in reindexed_trie_vocab
    assert tm.TRIE_SPECIAL_SHIFT not in reindexed_trie_vocab
    # add special shift symbols to the vocabulary
    reindexed_trie_vocab[tm.TRIE_NUMPAD] = len(reindexed_trie_vocab)
    reindexed_trie_vocab[tm.TRIE_SHIFT] = len(reindexed_trie_vocab)
    reindexed_trie_vocab[tm.TRIE_SPECIAL_SHIFT] = len(reindexed_trie_vocab)
    # add special symbols to the merges with highest priority
    new_merges = []
    new_merges.extend([(tm.TRIE_NUMPAD, number) for number in tm.numbers])
    new_merges.extend([(tm.TRIE_SHIFT, letter) for letter in tm.letters])
    new_merges.extend([(tm.TRIE_SPECIAL_SHIFT, special) for special in tm.other_special_chars])
    trie_merges = new_merges + trie_merges
    #
    trie_raw_merges = [f"{l} {r}" for l, r in trie_merges]
    #
    return reindexed_trie_vocab, trie_raw_merges


def build_output_json(source_json: dict, vocab: dict[str, int], merges: list[str]) -> dict:
    model = source_json.get("model", {})
    out = copy.deepcopy(source_json)
    out["model"]["vocab"] = vocab
    out["model"]["merges"] = merges
    return out


def main() -> None:
    tok = AutoTokenizer.from_pretrained(HF_TOKENIZER_NAME)
    source_json_str = tok.backend_tokenizer.to_str()
    source_json = json.loads(source_json_str)
    vocab, merges = parse_source(source_json)
    reindexed_trie_vocab, trie_raw_merges = filter_and_reindex(vocab, merges)

    assert len(reindexed_trie_vocab) > 0, "filter removed every token"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_json = build_output_json(source_json, reindexed_trie_vocab, trie_raw_merges)
    tokenizer_path = OUTPUT_DIR / "tokenizer.json"
    with tokenizer_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
        f.write("\n")
    summary = {
        "trie_stop_token": tm.TRIE_STOP,
        "trie_start_token": tm.TRIE_START,
        "source_vocab_size": len(vocab),
        "source_merge_count": len(merges),
        "final_vocab_size": len(reindexed_trie_vocab),
        "final_merge_count": len(trie_raw_merges),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"source_vocab_size= {len(vocab)}")
    print(f"source_merge_count= {len(merges)}")
    print(f"final_vocab_size= {len(reindexed_trie_vocab)}")
    print(f"final_merge_count= {len(trie_raw_merges)}")
    print(f"output_dir= {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
