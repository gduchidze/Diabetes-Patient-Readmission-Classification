"""Model training, resampling, and evaluation."""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.core.config import (
    DTREE_MAX_DEPTH,
    DTREE_MIN_SAMPLES_SPLIT,
    FEATURE_SET,
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_SPLIT,
    RF_N_ESTIMATORS,
    SMOTE_RANDOM_STATE,
    TEST_SIZE,
)


def _shared_params(n_features: int) -> dict[str, object]:
    """Params common to every run (split/SMOTE config + feature width)."""
    return {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "smote_random_state": SMOTE_RANDOM_STATE,
        "n_features": n_features,
    }


def model_params(kind: str, n_features: int) -> dict[str, object]:
    """Hyperparameters logged to MLflow for a given model kind."""
    base = _shared_params(n_features)
    if kind == "Logistic Regression":
        return {**base, "penalty": "l1", "solver": "liblinear"}
    if kind == "Decision Tree":
        return {
            **base,
            "criterion": "entropy",
            "max_depth": DTREE_MAX_DEPTH,
            "min_samples_split": DTREE_MIN_SAMPLES_SPLIT,
        }
    if kind == "Random Forests":
        return {
            **base,
            "criterion": "gini",
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": RF_MAX_DEPTH,
            "min_samples_split": RF_MIN_SAMPLES_SPLIT,
        }
    raise ValueError(f"Unknown model kind: {kind}")


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select available model features and the target (notebook cells 54/66)."""
    features = [f for f in FEATURE_SET if f in df.columns]
    x = df[features]
    y = df["readmitted"]
    return x, y


def resample_smote(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Balance classes with SMOTE oversampling (notebook cells 60/72)."""
    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    x_res, y_res = smote.fit_resample(x, y)
    x_res = pd.DataFrame(x_res, columns=list(x.columns))
    return x_res, y_res


def split(x: pd.DataFrame, y: pd.Series) -> tuple:
    """Train/test split with the notebook's fixed parameters."""
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_logistic(x_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """L1-penalized logistic regression (notebook cells 56/61)."""
    model = LogisticRegression(
        fit_intercept=True, penalty="l1", solver="liblinear", random_state=RANDOM_STATE
    )
    model.fit(x_train, y_train)
    return model


def train_decision_tree(
    x_train: pd.DataFrame, y_train: pd.Series
) -> DecisionTreeClassifier:
    """Entropy decision tree (notebook cell 68)."""
    model = DecisionTreeClassifier(
        max_depth=DTREE_MAX_DEPTH,
        criterion="entropy",
        min_samples_split=DTREE_MIN_SAMPLES_SPLIT,
    )
    model.fit(x_train, y_train)
    return model


def train_random_forest(
    x_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """Gini random forest (notebook cell 73)."""
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        criterion="gini",
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
    )
    model.fit(x_train, y_train)
    return model


def evaluate(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute accuracy/precision/recall/f1 for a fitted model."""
    pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }
