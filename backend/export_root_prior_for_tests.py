"""Write `bayesian/testdata/root_lm_prior.json` using the same LM stack as `lm.py`.

Requires the Poetry env from `backend/` (with `bayesian` built) and CUDA.

From the repo root:

    cd backend && poetry run python export_root_prior_for_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "bayesian" / "testdata" / "root_lm_prior.json"


def main() -> None:
    sys.path.insert(0, str(REPO / "backend"))
    import lm  # noqa: E402 — loads CUDA model

    lm.runtime.reset()
    req = lm.runtime.session.next_requested_prior()
    payload = lm.runtime.prior_model.prior_update_json_for_request(req)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(payload + "\n", encoding="utf-8")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
