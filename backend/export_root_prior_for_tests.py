"""Write `bayesian/testdata/root_lm_prior.json` using the same LM stack as `new_lm.py`.

Requires the Poetry env from `language_server/` (with `bayesian` built) and CUDA.

From the repo root:

    cd language_server && poetry run python ../backend/export_root_prior_for_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "bayesian" / "testdata" / "root_lm_prior.json"


def main() -> None:
    sys.path.insert(0, str(REPO / "backend"))
    import new_lm  # noqa: E402 — loads CUDA model

    new_lm.runtime.reset()
    req = new_lm.runtime.session.next_requested_prior()
    payload = new_lm.runtime.prior_model.prior_update_json_for_request(req)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(payload + "\n", encoding="utf-8")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
