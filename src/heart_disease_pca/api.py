from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from heart_disease_pca.predict import load_model_metadata, predict_risk

app = FastAPI(
    title="Heart Disease Risk ML Platform",
    description="PCA-enabled ML inference API for educational heart disease risk prediction.",
    version="1.0.0",
)


class HeartDiseaseFeatures(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: int = Field(..., ge=60, le=250)
    chol: int = Field(..., ge=80, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=40, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=3, le=7)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    return load_model_metadata()


@app.post("/predict")
def predict(payload: HeartDiseaseFeatures) -> dict:
    return predict_risk(payload.model_dump())
