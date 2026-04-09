"""Thin LM-related helpers; uses the Rust `bayesian` extension for numeric kernels."""

from __future__ import annotations

import bayesian


def pair_sum(a: int, b: int) -> int:
    """Sum two integers via the Rust `add` binding."""
    return bayesian.add(a, b)


if __name__ == "__main__":
    print(pair_sum(2, 3))
