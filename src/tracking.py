"""MLflow experiment tracking (traditional-ML path).

One run per model under a single experiment, with manual parameter/metric
logging plus the fitted sklearn model as an artifact.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI


def init_tracking() -> None:
    """Point MLflow at the local store and select/create the experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # Tag the experiment so MLflow recognizes it as a custom-model project.
    mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")


def log_model_run(
    name: str,
    model,
    params: dict[str, object],
    metrics: dict[str, float],
    input_example: pd.DataFrame,
) -> None:
    """Log one model's params, metrics, and artifact as a dedicated run."""
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name=name, input_example=input_example)
