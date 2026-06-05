from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from heart_disease_pca.config import METRICS_PATH, MODEL_PATH
from heart_disease_pca.train import train_and_select_model


@lru_cache(maxsize=1)
def load_model():
    """Load the persisted model, training it once if artifacts are unavailable."""
    if not MODEL_PATH.exists():
        train_and_select_model()
    return joblib.load(MODEL_PATH)


def predict_risk(features: dict) -> dict:
    """Run one prediction and return class, probability, and interpretation."""
    model = load_model()
    frame = pd.DataFrame([features])
    probability = float(model.predict_proba(frame)[0][1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "risk_probability": round(probability, 4),
        "label": "heart_disease_risk_detected" if prediction else "no_heart_disease_risk_detected",
        "note": "Educational ML demo only; not a clinical diagnosis.",
    }


def load_model_metadata() -> dict:
    """Return persisted model metadata for API introspection."""
    if not METRICS_PATH.exists():
        train_and_select_model()
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
