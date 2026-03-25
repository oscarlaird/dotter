from __future__ import annotations

import subprocess
from importlib.resources import files


def _binary_path() -> str:
    return str(files("lean_hello").joinpath("bin/lean_hello_bin"))


def _run(*args: str) -> str:
    result = subprocess.run(
        [_binary_path(), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def hello() -> str:
    return _run("hello")


def add(a: int, b: int) -> int:
    return int(_run(str(a), str(b)))
