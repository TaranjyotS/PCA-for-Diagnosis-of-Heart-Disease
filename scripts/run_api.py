"""Start the FastAPI app from the project root.

Run with:
    python scripts/run_api.py
Then open:
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    uvicorn.run(
        "heart_disease_pca.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=str(SRC_DIR),
    )
