from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Cleveland_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "heart_disease_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "target"
