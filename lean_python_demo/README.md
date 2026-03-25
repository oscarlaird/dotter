# lean-hello

This is a tiny end-to-end demo showing how to package a Lean program as a
Python library.

The Python package exposes:

- `hello() -> str`
- `add(a: int, b: int) -> int`

## How it works

1. `Main.lean` defines a tiny Lean executable.
2. `lake build lean_hello_bin` compiles that Lean code into a native binary.
3. `setup.py` hooks into the Python build and runs the Lean build first.
4. The built binary is copied into `src/lean_hello/bin/`.
5. The Python wrapper in `src/lean_hello/_core.py` calls the bundled binary
   with `subprocess` and converts the result back into Python values.

## Why this approach

Lean can be called from Python through lower-level C interop, but that is much
more manual. For a minimal demo that really works end to end, bundling a Lean
executable inside a Python package is the simplest reliable path.

## Local test commands

```bash
cd lean_python_demo
lake build lean_hello_bin
./.lake/build/bin/lean_hello_bin hello
./.lake/build/bin/lean_hello_bin 2 3
python3 -m pip install --target .pkgtest .
PYTHONPATH=.pkgtest python3 -c "import lean_hello; print(lean_hello.hello()); print(lean_hello.add(7, 8))"
```
