"""
Train + evaluate churn models end-to-end.

Run:
  python scripts/train.py --csv data/raw/churn.csv

This script will:
- Load data
- Run EDA and save plots
- Split train/valid/test (stratified)
- Train candidate models (+ light randomized search)
- Select best model by ROC-AUC
- Pick a decision threshold on validation set (default: best F1)
- Evaluate on test set and save plots + metrics
- Save final fitted pipeline to models/best_model.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from src.config import (
    DEFAULT_RAW_CSV,
    DROP_COLS,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.data_io import load_churn_csv, split_features_target, get_feature_schema, save_schema
from src.eda import run_eda
from src.evaluation import (
    choose_threshold_by_f1,
    compute_metrics,
    get_positive_class_proba,
    plot_confusion,
    plot_feature_importance_permutation,
    plot_logistic_odds_ratios,
    plot_pr,
    plot_roc,
    save_json,
    top_k_recall,
)
from src.modeling import build_pipeline, make_candidate_models, small_param_distributions


def drop_non_predictive_cols(X: pd.DataFrame) -> pd.DataFrame:
    """
    Drop known non-predictive columns like RowNumber, CustomerId, Surname.
    """
    existing = [c for c in DROP_COLS if c in X.columns]
    return X.drop(columns=existing)


def fit_with_random_search(
    name: str,
    pipeline,
    param_dist: Dict[str, object],
    X_train,
    y_train,
    cv: StratifiedKFold,
    random_state: int,
    n_iter: int = 18,
):
    """
    Fit a pipeline with RandomizedSearchCV.
    """
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, float(search.best_score_), search.best_params_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_RAW_CSV),
        help="Path to churn.csv (raw). Default: data/raw/churn.csv",
    )
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    random_state = args.random_state

    # -------------------------
    # 1) Load data
    # -------------------------
    df = load_churn_csv(csv_path)

    # Save schema for later scoring / documentation
    schema = get_feature_schema()
    save_schema(schema, MODELS_DIR / "feature_schema.json")

    # -------------------------
    # 2) EDA (save figures)
    # -------------------------
    run_eda(df, FIGURES_DIR)

    # -------------------------
    # 3) Split data
    # -------------------------
    X, y = split_features_target(df)
    X = drop_non_predictive_cols(X)

    # Train+temp split, then temp -> valid+test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    # -------------------------
    # 4) Train candidate models
    # -------------------------
    candidates = make_candidate_models(random_state=random_state)
    param_spaces = small_param_distributions(random_state=random_state)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results = {}
    fitted_models = {}

    for name, model in candidates.items():
        pipe = build_pipeline(model)
        if name in param_spaces:
            best_est, best_cv_auc, best_params = fit_with_random_search(
                name=name,
                pipeline=pipe,
                param_dist=param_spaces[name],
                X_train=X_train,
                y_train=y_train,
                cv=cv,
                random_state=random_state,
                n_iter=18,
            )
        else:
            best_est = pipe.fit(X_train, y_train)
            # For non-tuned models, approximate score on CV would require cross_val_score;
            # I keep it simple and compute valid ROC-AUC below.
            best_cv_auc, best_params = float("nan"), {}

        # Validate
        y_valid_prob = get_positive_class_proba(best_est, X_valid)
        valid_auc = float(np.round(np.nan_to_num(np.nan), 6))  # placeholder, overwritten below
        valid_auc = float(
            __import__("sklearn.metrics").metrics.roc_auc_score(y_valid, y_valid_prob)
        )

        results[name] = {
            "cv_best_roc_auc": best_cv_auc,
            "valid_roc_auc": valid_auc,
            "best_params": best_params,
        }
        fitted_models[name] = best_est

    # Select best model by validation ROC-AUC
    best_name = max(results.keys(), key=lambda k: results[k]["valid_roc_auc"])
    best_pipeline = fitted_models[best_name]

    # -------------------------
    # 5) Choose decision threshold on validation (default: max F1)
    # -------------------------
    y_valid_prob = get_positive_class_proba(best_pipeline, X_valid)
    threshold = choose_threshold_by_f1(y_valid.values, y_valid_prob)

    save_json({"model": best_name, "threshold": threshold}, MODELS_DIR / "best_threshold.json")

    # -------------------------
    # 6) Final evaluation on test set
    # -------------------------
    y_test_prob = get_positive_class_proba(best_pipeline, X_test)

    metrics = compute_metrics(y_test.values, y_test_prob, threshold=threshold)

    metrics["top_10pct_recall"] = top_k_recall(y_test.values, y_test_prob, top_frac=0.10)
    metrics["top_05pct_recall"] = top_k_recall(y_test.values, y_test_prob, top_frac=0.05)

    report = {
        "best_model_name": best_name,
        "model_comparison": results,
        "test_metrics": metrics,
        "notes": {
            "threshold_selection": "Threshold chosen on validation set to maximize F1 for this project. "
                                   "In production, choose threshold based on cost/ROI constraints.",
        },
    }

    save_json(report, REPORTS_DIR / "metrics.json")

    # Curves and confusion matrix
    plot_roc(y_test.values, y_test_prob, FIGURES_DIR / "roc_curve.png")
    plot_pr(y_test.values, y_test_prob, FIGURES_DIR / "pr_curve.png")
    plot_confusion(y_test.values, y_test_prob, threshold, FIGURES_DIR / "confusion_matrix.png")

    # Interpretability plots
    plot_feature_importance_permutation(
        fitted_pipeline=best_pipeline,
        X_valid=X_valid,
        y_valid=y_valid,
        out_path=FIGURES_DIR / "feature_importance.png",
        top_n=15,
    )
    # If best model is logistic regression, also output odds ratios plot
    plot_logistic_odds_ratios(best_pipeline, FIGURES_DIR / "logistic_odds_ratios.png", top_n=15)

    # -------------------------
    # 7) Save final model
    # -------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(best_pipeline, MODELS_DIR / "best_model.joblib")

    print("\nTraining complete.")
    print(f"Best model: {best_name}")
    print(f"Saved model -> {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved metrics -> {REPORTS_DIR / 'metrics.json'}")
    print(f"Saved figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
