"""Train the heart disease risk model from the project root.

This wrapper is intentionally VS Code / Windows friendly so you can run:
    python scripts/train_model.py
without installing GNU Make or manually setting PYTHONPATH.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from heart_disease_pca.train import train_and_select_model  # noqa: E402


if __name__ == "__main__":
    metrics = train_and_select_model()
    print(json.dumps(metrics, indent=2))
