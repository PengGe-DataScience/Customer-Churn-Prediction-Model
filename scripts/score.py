"""
Score a CSV of customers (same feature columns as training, without Exited).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import MODELS_DIR, REPORTS_DIR
from src.predict import score_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_csv", required=True, help="Input CSV to score")
    parser.add_argument("--out", dest="output_csv", required=True, help="Output CSV path")
    args = parser.parse_args()

    model_path = MODELS_DIR / "best_model.joblib"
    threshold_path = MODELS_DIR / "best_threshold.json"

    score_file(
        model_path=model_path,
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        threshold_path=threshold_path,
    )

    print("Scoring complete.")
    print(f"Output -> {args.output_csv}")


if __name__ == "__main__":
    main()
