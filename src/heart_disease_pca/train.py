from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heart_disease_pca.config import FIGURES_DIR, METRICS_PATH, MODEL_DIR, MODEL_PATH, RANDOM_STATE
from heart_disease_pca.data_loader import load_data, make_binary_target
from heart_disease_pca.preprocessing import split_features_target, train_test_split_stratified
from heart_disease_pca.visualizations import generate_research_figures


def build_candidate_pipelines() -> dict[str, Pipeline]:
    """Build baseline and PCA-enabled candidate models."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "pca_logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE)),
        ]),
        "pca_gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def evaluate_pipeline(pipeline: Pipeline, X_test, y_test) -> dict[str, float]:
    """Evaluate a fitted classifier using portfolio-friendly metrics."""
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_score)), 4),
    }


def train_and_select_model(
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    generate_figures: bool = True,
) -> dict[str, Any]:
    """Train candidate models, save best model, metrics, and research figures.

    This preserves the original academic PCA/EDA story while fixing the production ML issues:
    preprocessing is fitted on training data only, PCA is part of the sklearn Pipeline, and
    the trained artifact is served through FastAPI.
    """
    raw_df = load_data()
    df = make_binary_target(raw_df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    results: dict[str, dict[str, float]] = {}
    trained: dict[str, Pipeline] = {}
    for name, pipeline in build_candidate_pipelines().items():
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_pipeline(pipeline, X_test, y_test)
        trained[name] = pipeline

    best_model_name = max(results, key=lambda model_name: results[model_name]["roc_auc"])
    best_pipeline = trained[best_model_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    y_pred = best_pipeline.predict(X_test)
    y_score = best_pipeline.predict_proba(X_test)[:, 1]
    figure_paths: list[str] = []
    if generate_figures:
        figure_paths = generate_research_figures(
            raw_df,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_score=y_score,
            metrics=results,
            output_dir=FIGURES_DIR,
        )

    metrics = {
        "objective": "Binary heart disease risk classification: 0=no disease, 1=disease present",
        "best_model": best_model_name,
        "candidate_metrics": results,
        "feature_names": list(X.columns),
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "generated_figures": figure_paths,
        "notes": [
            "Original Cleveland labels 1-4 are converted to binary disease-present labels.",
            "Scaler and PCA are fitted only on training data through sklearn Pipelines.",
            "Original exploratory plots are preserved under reports/figures.",
        ],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_select_model(), indent=2))
