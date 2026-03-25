#!/usr/bin/env python3
"""Read and query the on-disk xi precompute files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from build_xi_precomp import DEFAULT_OUTPUT_DIR


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


@dataclass
class XiPrecomp:
    tokens: list[str]
    prefixes: list[str]
    bits: bytes

    def __post_init__(self) -> None:
        self.token_to_row = {token: row for row, token in enumerate(self.tokens)}
        self.prefix_to_col = {prefix: col for col, prefix in enumerate(self.prefixes)}
        self.row_nbytes = (len(self.prefixes) + 7) // 8
        expected_nbytes = len(self.tokens) * self.row_nbytes
        if len(self.bits) != expected_nbytes:
            raise ValueError(
                f"xi.bits has {len(self.bits)} bytes, expected {expected_nbytes}"
            )

    @classmethod
    def from_dir(cls, directory: str | Path) -> "XiPrecomp":
        directory = Path(directory)
        return cls(
            tokens=_read_lines(directory / "tokens.txt"),
            prefixes=_read_lines(directory / "prefixes.txt"),
            bits=(directory / "xi.bits").read_bytes(),
        )

    def has_token(self, token: str) -> bool:
        return token in self.token_to_row

    def has_prefix(self, prefix: str) -> bool:
        return prefix in self.prefix_to_col

    def xi_by_index(self, row: int, col: int) -> bool:
        byte_index = row * self.row_nbytes + (col // 8)
        bit_offset = col % 8
        return bool((self.bits[byte_index] >> bit_offset) & 1)

    def xi(self, token: str, prefix: str) -> bool:
        row = self.token_to_row[token]
        col = self.prefix_to_col[prefix]
        return self.xi_by_index(row, col)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read xi precompute files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing tokens.txt / prefixes.txt / xi.bits (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("token", help="Token row to query")
    parser.add_argument("prefix", help="Prefix column to query")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    xi = XiPrecomp.from_dir(args.input_dir)
    print(f"token_present: {xi.has_token(args.token)}")
    print(f"prefix_present: {xi.has_prefix(args.prefix)}")
    if xi.has_token(args.token) and xi.has_prefix(args.prefix):
        print(f"xi: {xi.xi(args.token, args.prefix)}")


if __name__ == "__main__":
    main()
