# Backend

Active Python backend runtime and environment.

The canonical runtime entrypoint is `lm.py`.

The Poetry project for the active backend now lives in `backend/`, so rebuild the Rust
extension and run the backend from here.

From the repository root:

```sh
cd backend
poetry install
poetry run poe develop-bayesian
poetry run python lm.py
```

`develop-bayesian` rebuilds and installs the Rust facade crate from
`../bayesian/crates/bayesian` into this Poetry environment using `maturin develop`.

`requirements.txt` is optional legacy pip pinning; prefer Poetry for the active backend.
