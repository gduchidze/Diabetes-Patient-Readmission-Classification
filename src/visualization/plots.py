"""EDA and model-comparison plots, written to the output directory."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.core.config import OUTPUT_DIR  # noqa: E402


def _ensure_output_dir() -> Path:
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_eda(df: pd.DataFrame) -> None:
    """Exploratory plots of features vs. readmission."""
    out = _ensure_output_dir()

    fig = plt.figure(figsize=(13, 7))
    ax = sns.kdeplot(
        df.loc[df["readmitted"] == 0, "time_in_hospital"],
        color="b",
        fill=True,
        label="Not Readmitted",
    )
    sns.kdeplot(
        df.loc[df["readmitted"] == 1, "time_in_hospital"],
        color="r",
        fill=True,
        label="Readmitted",
        ax=ax,
    )
    ax.set(xlabel="Time in Hospital", ylabel="Frequency")
    plt.title("Time in Hospital VS. Readmission")
    plt.legend()
    fig.savefig(out / "eda_time_in_hospital.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(15, 10))
    sns.countplot(y=df["age"], hue=df["readmitted"]).set_title(
        "Age of Patient VS. Readmission"
    )
    fig.savefig(out / "eda_age.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8))
    sns.countplot(y=df["race"], hue=df["readmitted"])
    fig.savefig(out / "eda_race.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8))
    sns.barplot(x=df["readmitted"], y=df["num_medications"]).set_title(
        "Number of medication used VS. Readmission"
    )
    fig.savefig(out / "eda_num_medications.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8))
    sns.barplot(y=df["service_utilization"], x=df["readmitted"]).set_title(
        "Service Utilization VS. Readmission"
    )
    fig.savefig(out / "eda_service_utilization.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8))
    sns.countplot(y=df["max_glu_serum"], hue=df["readmitted"]).set_title(
        "Glucose test serum test result VS. Readmission"
    )
    fig.savefig(out / "eda_max_glu_serum.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(15, 6))
    ax = sns.kdeplot(
        df.loc[df["readmitted"] == 0, "num_lab_procedures"],
        color="b",
        fill=True,
        label="Not readmitted",
    )
    sns.kdeplot(
        df.loc[df["readmitted"] == 1, "num_lab_procedures"],
        color="r",
        fill=True,
        label="readmitted",
        ax=ax,
    )
    ax.set(xlabel="Number of lab procedure", ylabel="Frequency")
    plt.title("Number of lab procedure VS. Readmission")
    plt.legend()
    fig.savefig(out / "eda_num_lab_procedures.png", bbox_inches="tight")
    plt.close(fig)


def plot_correlation(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap of numeric features."""
    out = _ensure_output_dir()
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    numeric = df[numeric_columns].drop(
        columns=["patient_nbr", "encounter_id"], errors="ignore"
    )
    corr = numeric.corr(method="pearson")
    fig = plt.figure(figsize=(20, 16))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation")
    fig.savefig(out / "correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results: dict[str, dict[str, float]]) -> None:
    """Grouped bar chart of model metrics."""
    out = _ensure_output_dir()
    models = list(results.keys())
    x_pos = range(len(models))
    accuracy = [results[m]["accuracy"] for m in models]
    precision = [results[m]["precision"] for m in models]
    recall = [results[m]["recall"] for m in models]

    fig = plt.figure(figsize=(14, 7))
    ax = plt.subplot(111)

    for offset, values, color, label in (
        (0.0, accuracy, "red", "Accuracy"),
        (0.15, precision, "blue", "Precision"),
        (0.3, recall, "green", "Recall"),
    ):
        positions = [i + offset for i in x_pos]
        plt.bar(positions, values, width=0.15, alpha=0.7, color=color, label=label)
        for pos, value in zip(positions, values):
            plt.text(pos, value + 0.01, f"{value:.4f}", ha="center", fontsize=10)

    plt.xticks([i + 0.15 for i in x_pos], models)
    plt.ylabel("Performance Metrics")
    plt.title("Performance Metrics for Different Models")
    plt.legend()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.savefig(out / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)
