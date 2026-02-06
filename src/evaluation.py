"""
Evaluation utilities:
- Compute classification metrics
- Choose a decision threshold
- Plot ROC/PR curves and confusion matrix
- Feature importance for interpretability
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Compute standard metrics at a given threshold, plus threshold-free metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
    return metrics


def choose_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Choose threshold that maximizes F1 on a validation set.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx])


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    _savefig(fig, out_path)


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.title("Precision–Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    _savefig(fig, out_path)


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, out_path: Path) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm, aspect="auto")
    plt.title(f"Confusion Matrix (threshold={threshold:.2f})")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.colorbar()

    # Add counts as text
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")

    _savefig(fig, out_path)


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_positive_class_proba(model: BaseEstimator, X) -> np.ndarray:
    """
    Return predicted probabilities for positive class, for models that support it.
    HistGradientBoosting supports predict_proba in sklearn (for binary classification).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # As a fallback, use decision_function + sigmoid-like mapping
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise TypeError("Model does not support probability prediction.")


def top_k_recall(y_true: np.ndarray, y_prob: np.ndarray, top_frac: float = 0.1) -> float:
    """
    Metric: among the top X% highest-risk customers,
    what fraction of all churners do we capture?
    """
    n = len(y_true)
    k = max(1, int(np.ceil(n * top_frac)))
    idx = np.argsort(-y_prob)[:k]
    captured_churners = y_true[idx].sum()
    total_churners = y_true.sum()
    return float(captured_churners / total_churners) if total_churners > 0 else 0.0


def plot_feature_importance_permutation(
    fitted_pipeline: Pipeline,
    X_valid,
    y_valid,
    out_path: Path,
    n_repeats: int = 8,
    random_state: int = 42,
    top_n: int = 15,
) -> None:
    """
    I compute permutation importance on the *original feature columns* (pre-pipeline).
    """
    result = permutation_importance(
        fitted_pipeline,
        X_valid,
        y_valid,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
    )

    importances = result.importances_mean
    std = result.importances_std

    # Pipeline expects original feature column names; X_valid should be a DataFrame
    feature_names = list(X_valid.columns)

    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    vals = importances[order]
    errs = std[order]

    fig = plt.figure(figsize=(7, 5))
    plt.barh(names[::-1], vals[::-1], xerr=errs[::-1])
    plt.title("Permutation Feature Importance (Top)")
    plt.xlabel("Importance (ROC-AUC decrease)")
    _savefig(fig, out_path)


def plot_logistic_odds_ratios(
    fitted_pipeline: Pipeline,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """
    For logistic regression: show largest-magnitude odds ratios.

    This requires:
    - pipeline named steps: preprocess, model
    - model has coef_
    - preprocessor includes onehot; I extract transformed feature names

    """
    model = fitted_pipeline.named_steps["model"]
    preprocess = fitted_pipeline.named_steps["preprocess"]

    if not hasattr(model, "coef_"):
        return  # not a linear model

    # Try to get feature names after preprocessing
    try:
        ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
        cat_features = list(ohe.get_feature_names_out())
        num_features = preprocess.named_transformers_["num"].named_steps  # only to confirm it exists
        transformed_names = []
        numeric_cols = preprocess.transformers_[0][2]
        transformed_names.extend(list(numeric_cols))
        # categorical
        transformed_names.extend(cat_features)
    except Exception:
        return

    coef = model.coef_.ravel()
    odds = np.exp(coef)

    # Pick top effects by distance from 1 (neutral)
    impact = np.abs(np.log(odds))
    order = np.argsort(impact)[::-1][:top_n]

    names = [transformed_names[i] for i in order]
    vals = odds[order]

    fig = plt.figure(figsize=(7, 5))
    plt.barh(names[::-1], vals[::-1])
    plt.axvline(1.0, linestyle="--")
    plt.title("Logistic Regression Odds Ratios (Top)")
    plt.xlabel("Odds Ratio (>1 increases churn odds)")
    _savefig(fig, out_path)
