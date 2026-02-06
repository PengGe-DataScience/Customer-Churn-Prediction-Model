"""
Data loading and basic validation utilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .config import TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERIC_COLS


@dataclass(frozen=True)
class FeatureSchema:
    """
    A lightweight schema to record what features the pipeline expects.
    Useful when scoring new data later.
    """
    target: str
    drop_cols: List[str]
    categorical_cols: List[str]
    numeric_cols: List[str]


def load_churn_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the churn CSV into a DataFrame.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Basic sanity checks
    required = set([TARGET_COL] + DROP_COLS + CATEGORICAL_COLS + NUMERIC_COLS)
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(
            "CSV is missing required columns.\n"
            f"Missing: {missing}\n"
            f"Found: {sorted(df.columns.tolist())}"
        )

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into X (features) and y (target).
    """
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def get_feature_schema() -> FeatureSchema:
    """
    Return the expected schema for this dataset.
    """
    return FeatureSchema(
        target=TARGET_COL,
        drop_cols=DROP_COLS,
        categorical_cols=CATEGORICAL_COLS,
        numeric_cols=NUMERIC_COLS,
    )


def save_schema(schema: FeatureSchema, out_path: Path) -> None:
    """
    Save schema as JSON.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(schema.__dict__, f, indent=2)
