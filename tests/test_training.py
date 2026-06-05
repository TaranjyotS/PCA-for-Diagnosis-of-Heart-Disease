from pathlib import Path

from heart_disease_pca.train import train_and_select_model


def test_train_and_select_model_creates_artifacts(tmp_path: Path):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    metrics = train_and_select_model(model_path=model_path, metrics_path=metrics_path)
    assert model_path.exists()
    assert metrics_path.exists()
    assert "best_model" in metrics
    assert "candidate_metrics" in metrics
