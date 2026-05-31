"""Generate /predict test payloads from real engineered data rows.

Run: uv run --no-sync python test/generate_payloads.py

Writes test/*.json — each valid file is `{"features": {...}}` ready to POST.
Picks real rows the winner predicts as 0 and as 1, plus negative-test cases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import preprocess  # noqa: E402
from src.data.loader import load_data
from src.modeling.estimators import build_feature_matrix
from src.modeling.persistence import load_winner

TEST_DIR = Path("test")


def _write(name: str, payload: dict) -> None:
    (TEST_DIR / name).write_text(json.dumps(payload, indent=2))
    print(f"wrote test/{name}")


def main() -> None:
    model, feature_columns = load_winner()
    df_pd = preprocess(load_data())
    x, _y = build_feature_matrix(df_pd)
    x = x[feature_columns]

    preds = model.predict(x)

    idx0 = next(i for i, p in enumerate(preds) if p == 0)
    idx1 = next(i for i, p in enumerate(preds) if p == 1)

    row0 = {k: float(v) for k, v in x.iloc[idx0].to_dict().items()}
    row1 = {k: float(v) for k, v in x.iloc[idx1].to_dict().items()}

    _write("predict_not_readmitted.json", {"features": row0})
    _write("predict_readmitted.json", {"features": row1})

    partial = {k: row0[k] for k in feature_columns[:5]}
    _write("invalid_missing_features.json", {"features": partial})
    _write("invalid_empty.json", {"features": {}})

    print(f"\nExpected: not_readmitted->0, readmitted->1, invalid_*->HTTP 422")


if __name__ == "__main__":
    main()
