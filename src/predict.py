"""
Batch scoring utility used by scripts/score.py.

This loads the trained pipeline and outputs churn probabilities for each row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from joblib import load


def load_threshold(threshold_path: Path) -> float:
    if not threshold_path.exists():
        return 0.5
    with threshold_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return float(obj.get("threshold", 0.5))


def score_file(
    model_path: Path,
    input_csv: Path,
    output_csv: Path,
    threshold_path: Optional[Path] = None,
) -> None:
    """
    Score a CSV with the same schema as training data (excluding target).

    Output columns:
    - churn_proba
    - churn_pred (based on chosen threshold)
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    model = load(model_path)
    df = pd.read_csv(input_csv)

    proba = model.predict_proba(df)[:, 1]
    threshold = load_threshold(threshold_path) if threshold_path else 0.5
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = pred

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
