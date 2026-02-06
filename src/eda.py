"""
Exploratory data analysis (EDA) utilities.

This module generates a small set of business-friendly plots:
- Class balance (churn rate)
- Churn rate by categorical fields
- Numeric distributions split by churn
- Correlation heatmap (numeric only)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TARGET_COL, CATEGORICAL_COLS, NUMERIC_COLS


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_class_balance(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot churn vs non-churn counts.
    """
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["No churn (0)", "Churn (1)"]

    fig = plt.figure(figsize=(6, 4))
    plt.bar(labels, counts.values)
    plt.title("Class Balance (Churn vs Non-churn)")
    plt.ylabel("Number of customers")
    _savefig(fig, out_path)


def plot_churn_by_categorical(df: pd.DataFrame, cat_cols: List[str], out_path: Path) -> None:
    """
    Plot churn rate by each categorical column.
    """
    n = len(cat_cols)
    fig = plt.figure(figsize=(7, 3.5 * n))

    for i, col in enumerate(cat_cols, start=1):
        ax = plt.subplot(n, 1, i)
        rates = df.groupby(col)[TARGET_COL].mean().sort_values(ascending=False)
        ax.bar(rates.index.astype(str), rates.values)
        ax.set_title(f"Churn rate by {col}")
        ax.set_ylabel("Churn rate")
        ax.set_ylim(0, max(0.05, rates.max() * 1.15))
        ax.tick_params(axis="x", rotation=0)

    _savefig(fig, out_path)


def plot_numeric_distributions(df: pd.DataFrame, num_cols: List[str], out_path: Path) -> None:
    """
    Plot numeric distributions for churn vs non-churn.
    Uses simple histogram overlays to keep dependencies minimal.
    """
    churn = df[df[TARGET_COL] == 1]
    non = df[df[TARGET_COL] == 0]

    n = len(num_cols)
    fig = plt.figure(figsize=(7, 2.8 * n))

    for i, col in enumerate(num_cols, start=1):
        ax = plt.subplot(n, 1, i)

        # Use shared bins for fair comparison
        values = df[col].dropna().values
        bins = min(30, max(10, int(np.sqrt(len(values)))))

        ax.hist(non[col].dropna(), bins=bins, alpha=0.6, label="No churn (0)")
        ax.hist(churn[col].dropna(), bins=bins, alpha=0.6, label="Churn (1)")
        ax.set_title(f"Distribution of {col}")
        ax.set_ylabel("Count")
        ax.legend()

    _savefig(fig, out_path)


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: List[str], out_path: Path) -> None:
    """
    Correlation heatmap for numeric fields only (including target for quick signal check).
    """
    corr_df = df[num_cols + [TARGET_COL]].corr(numeric_only=True)

    fig = plt.figure(figsize=(8, 7))
    plt.imshow(corr_df.values, aspect="auto")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90)
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.title("Correlation heatmap (numeric features + target)")
    plt.colorbar()
    _savefig(fig, out_path)


def run_eda(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Generate all EDA plots to the figures directory.
    """
    plot_class_balance(df, figures_dir / "class_balance.png")
    plot_churn_by_categorical(df, CATEGORICAL_COLS, figures_dir / "churn_by_category.png")
    plot_numeric_distributions(df, NUMERIC_COLS, figures_dir / "numeric_distributions.png")
    plot_correlation_heatmap(df, NUMERIC_COLS, figures_dir / "correlation_heatmap.png")
