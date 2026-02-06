from pathlib import Path

# Project root = folder containing this file's parent (src/) parent.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Outputs
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset file name
DEFAULT_RAW_CSV = RAW_DATA_DIR / "churn.csv"

# Target label in this churn dataset
TARGET_COL = "Exited"

# Columns that are identifiers / non-predictive text fields
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

# Categorical columns in this dataset
CATEGORICAL_COLS = ["Geography", "Gender"]

# Numeric columns (binary fields are OK to treat as numeric in tree + linear models)
NUMERIC_COLS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
