# Language server

Python WebSocket language model server (`lm.py`), Poetry project metadata, and related scripts (`new_lm.py`, `log_analyzer.py`, `blinks_analysis.py`, etc.).

This directory is the **Poetry project** for the `dotter` Python environment (see `pyproject.toml`). Run scripts here with that env (`poetry shell` or `poetry run python …`).

From the repository root:

```sh
cd language_server
poetry install
```

Install the Rust **`bayesian`** extension into this same venv (needed for `new_lm.py` and any `import bayesian`). Run again after you change Rust:

```sh
poetry run poe develop-bayesian
```

(`poe` comes from the **poethepoet** dev dependency; Poetry itself has no first-class “npm scripts” hook. Equivalent manual command: `poetry run bash -c 'cd ../bayesian && python -m maturin develop'`.)

If you see **`ModuleNotFoundError: No module named 'bayesian'`**, the step above was skipped or the venv changed.

Run the server (example):

```sh
poetry run python lm.py
```

Example — `new_lm.py` imports the Rust **`bayesian`** extension:

```sh
poetry run python new_lm.py
```

Build the `xi` precompute files (`tokens.txt`, `prefixes.txt`, `xi.bits`) with:

```sh
poetry run python build_xi_precomp.py
```

For a smaller smoke test:

```sh
poetry run python build_xi_precomp.py --limit-tokens 256 --output-dir /tmp/xi-precomp-smoke
```

Query a generated precompute:

```sh
poetry run python xi_precomp.py --input-dir /tmp/xi-precomp-smoke " the" " "
```

`requirements.txt` is optional legacy pip pinning; prefer Poetry for a consistent env.

## `bayesian` (Rust / PyO3)

**maturin** and **poethepoet** are dev dependencies in `pyproject.toml`. The **`bayesian`** crate is *not* a Poetry path dependency (that made `poetry lock` depend on PEP 517 isolated builds and broke on minimal Python installs). Install or refresh the extension with **`poetry run poe develop-bayesian`** (or `python -m maturin develop` from `../bayesian`).

With `poetry shell`, you can instead:

```sh
cd ../bayesian
maturin develop
```

See the [root README](../README.md) for the full local setup.
