from __future__ import annotations

from pathlib import Path

import pandas as pd

from heart_disease_pca.config import DATA_PATH, TARGET_COLUMN

REQUIRED_COLUMNS = {
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", TARGET_COLUMN,
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the Cleveland heart disease dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def make_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Cleveland severity labels into binary risk labels.

    Original labels: 0 = no disease, 1-4 = disease severity levels.
    Portfolio model objective: binary diagnosis/risk classification.
    """
    transformed = df.copy()
    transformed[TARGET_COLUMN] = (transformed[TARGET_COLUMN] > 0).astype(int)
    return transformed
