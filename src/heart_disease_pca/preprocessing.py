from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from heart_disease_pca.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the binary target column."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def train_test_split_stratified(
    X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
