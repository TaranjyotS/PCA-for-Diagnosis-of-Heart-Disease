from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

from heart_disease_pca.config import FIGURES_DIR, TARGET_COLUMN


def _save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def add_age_range(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["age_range"] = pd.cut(
        data["age"],
        bins=[0, 39, 55, 120],
        labels=["Young", "Middle", "Elderly"],
        include_lowest=True,
    )
    return data


def generate_research_figures(
    df: pd.DataFrame,
    X_train: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    y_pred: np.ndarray | None = None,
    y_score: np.ndarray | None = None,
    metrics: Mapping[str, Mapping[str, float]] | None = None,
    output_dir: Path = FIGURES_DIR,
) -> list[str]:
    """Generate research and evaluation figures preserved from the original PCA project.

    The original Master's project contained many exploratory plots. This function keeps
    those ideas but saves them as organized, portfolio-friendly artifacts instead of
    opening interactive windows during execution.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data = add_age_range(df)
    saved: list[str] = []

    def save(name: str) -> None:
        path = output_dir / name
        _save_current(path)
        saved.append(str(path))

    # Correlation heatmap
    plt.figure(figsize=(11, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Clinical Feature Correlation Heatmap")
    save("correlation-heatmap.png")

    # Feature distributions
    numeric_cols = [c for c in df.columns if c != TARGET_COLUMN]
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=20, alpha=0.85)
        axes[i].set_title(col)
    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Clinical Feature Distributions", fontsize=16)
    save("feature-distributions.png")

    # Boxplots by target
    binary_target = (df[TARGET_COLUMN] > 0).astype(int)
    plot_df = df.drop(columns=[TARGET_COLUMN]).copy()
    plot_df["target_binary"] = binary_target
    melted = plot_df.melt(id_vars="target_binary", var_name="feature", value_name="value")
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=melted, x="feature", y="value", hue="target_binary")
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Spread by Binary Heart Disease Label")
    save("boxplots-by-target.png")

    # Age group distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x="age_range", hue=(data[TARGET_COLUMN] > 0).astype(int))
    plt.title("Heart Disease Distribution by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Patient Count")
    save("age-group-target-distribution.png")

    # Chest pain vs target
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="cp", hue=(df[TARGET_COLUMN] > 0).astype(int))
    plt.title("Chest Pain Type vs Heart Disease Label")
    plt.xlabel("Chest Pain Type")
    plt.ylabel("Patient Count")
    save("chest-pain-target-distribution.png")

    # Thalach by age
    plt.figure(figsize=(10, 5))
    sns.regplot(data=df, x="age", y="thalach", scatter_kws={"alpha": 0.6})
    plt.title("Age vs Maximum Heart Rate Achieved")
    plt.xlabel("Age")
    plt.ylabel("Maximum Heart Rate")
    save("age-vs-thalach.png")

    # PCA figures fitted once on full feature set for research visualization.
    X = df.drop(columns=[TARGET_COLUMN])
    y = (df[TARGET_COLUMN] > 0).astype(int)
    X_scaled = StandardScaler().fit_transform(X)
    pca_full = PCA().fit(X_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
    plt.axhline(0.90, linestyle="--", linewidth=1, label="90% variance")
    plt.axhline(0.95, linestyle="--", linewidth=1, label="95% variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save("pca-explained-variance.png")

    plt.figure(figsize=(8, 5))
    component_numbers = range(1, len(pca_full.explained_variance_ratio_) + 1)
    plt.bar(component_numbers, pca_full.explained_variance_ratio_)
    plt.title("PCA Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    save("scree-plot.png")

    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.75)
    plt.title("PCA Scatter Plot: First Two Components")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(*scatter.legend_elements(), title="Disease")
    plt.grid(True, alpha=0.25)
    save("pca-scatter-plot.png")

    # Model comparison chart
    if metrics:
        metric_df = pd.DataFrame(metrics).T.reset_index(names="model")
        plt.figure(figsize=(10, 6))
        melted_metrics = metric_df.melt(
            id_vars="model",
            var_name="metric",
            value_name="score",
        )
        sns.barplot(data=melted_metrics, x="score", y="model", hue="metric")
        plt.title("Candidate Model Comparison")
        plt.xlabel("Score")
        plt.ylabel("Model")
        save("model-comparison.png")

    if y_test is not None and y_pred is not None:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
        disp.plot(values_format="d")
        plt.title("Confusion Matrix")
        save("confusion-matrix.png")

    if y_test is not None and y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        save("roc-curve.png")

    return saved
