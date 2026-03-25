# Backend

Python scripts including `lm.py`, `new_lm.py`, and related utilities.

Use the **same Poetry environment as the language server** (`language_server/`) for anything that imports heavy deps (e.g. `torch`) or the **`bayesian`** Rust extension.

From the repository root:

```sh
cd language_server
poetry install
```

Then install or refresh the compiled `bayesian` module into that env (after cloning or after changing Rust):

```sh
cd ../bayesian
maturin develop
```

Run backend scripts with that env active (`poetry shell` from `language_server/`, or `poetry run python …`):

```sh
cd language_server
poetry run python ../backend/new_lm.py
```

**Why:** `import bayesian` loads the extension last installed into **that** virtualenv (via `maturin develop` / `poetry install`), not from `backend/` or `bayesian/target/` by filesystem path.

**Poetry note:** `bayesian` is declared as a **path dependency** in `language_server/pyproject.toml` (`develop = true`), so `poetry install` will try to build it with **maturin** (Rust required). Resolving or installing that dependency uses a PEP 517 isolated build; your Python must support creating venvs (e.g. on Debian/Ubuntu install `python3.12-venv` to match the pinned interpreter). If you prefer not to build via Poetry, skip reinstalling that dep and rely on `maturin develop` alone after `poetry install`.

See the [language server README](../language_server/README.md) and the [root README](../README.md) for broader setup.
