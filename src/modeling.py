"""
Modeling utilities:
- Build preprocessing + model pipelines
- Define candidate models
- Hyperparameter search spaces
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from .config import DROP_COLS, CATEGORICAL_COLS, NUMERIC_COLS


def build_preprocessor() -> ColumnTransformer:
    """
    Create a ColumnTransformer that:
    - drops ID/text columns (RowNumber, CustomerId, Surname)
    - imputes missing values
    - one-hot encodes categoricals
    - scales numerics (helpful for logistic regression)

    Keeping preprocessing inside the pipeline prevents leakage.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ],
        remainder="drop",  # drop any column not explicitly listed
    )

    return preprocessor


def make_candidate_models(random_state: int = 42) -> Dict[str, object]:
    """
    Define candidate classifiers.
    I include:
    - Logistic Regression: interpretable baseline
    - Random Forest: non-linear + feature importance
    - HistGradientBoosting: strong tabular baseline
    """
    models = {
        "logreg": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",  # helpful for class imbalance
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
        "hgb": HistGradientBoostingClassifier(
            random_state=random_state,
            max_depth=None,
        ),
    }
    return models


def build_pipeline(model: object) -> Pipeline:
    """
    Build (preprocessor -> model) pipeline.
    """
    preprocessor = build_preprocessor()

    # Note: I drop ID/text columns at data-level (in evaluation/training code)
    # because ColumnTransformer expects the feature columns to be present.
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipe


def small_param_distributions(random_state: int = 42) -> Dict[str, Dict[str, object]]:
    """
    Lightweight hyperparameter search spaces for RandomizedSearchCV.
    """
    return {
        "logreg": {
            "model__C": np.logspace(-2, 1.5, 12),
        },
        "random_forest": {
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [None, 4, 6, 8, 12],
            "model__min_samples_leaf": [1, 2, 4, 8],
        },
        "hgb": {
            "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [20, 30, 50],
        },
    }
