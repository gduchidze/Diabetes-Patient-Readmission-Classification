"""End-to-end diabetes readmission classification pipeline.

Mirrors the original notebook flow: load -> clean -> feature engineer ->
transform -> EDA -> train (LogReg / DecisionTree / RandomForest with SMOTE) ->
compare. Run with ``uv run python main.py``.
"""

from __future__ import annotations

from src.config import DIAG_DROP_COLS
from src.cleaning import drop_invalid_rows, drop_sparse_columns
from src.data_loader import load_data
from src.feature_engineering import (
    add_interaction_terms,
    add_numchange,
    add_nummed,
    add_service_utilization,
    build_diagnosis_levels,
    cast_nominal_to_object,
    categorize_diagnoses,
    encode_binary_columns,
    encode_lab_results,
    encode_target,
    map_age_midpoint,
    map_age_ordinal,
    recast_numeric,
    recode_admission_discharge,
)
from src.models import (
    build_feature_matrix,
    evaluate,
    resample_smote,
    split,
    train_decision_tree,
    train_logistic,
    train_random_forest,
)
from src.transform import (
    drop_redundant_columns,
    log_transform_report,
    numeric_columns,
    one_hot_encode,
    standardize_and_remove_outliers,
)
from src.visualization import plot_correlation, plot_eda, plot_model_comparison


def preprocess(df):
    """Clean + engineer features through to the modeling-ready one-hot frame."""
    df = drop_sparse_columns(df)
    df = drop_invalid_rows(df)
    df = add_service_utilization(df)
    df = add_numchange(df)
    df = recode_admission_discharge(df)
    df = encode_binary_columns(df)
    df = encode_lab_results(df)
    df = map_age_ordinal(df)
    df = encode_target(df)
    df = build_diagnosis_levels(df)
    df = categorize_diagnoses(df)

    # EDA uses the ordinal age, raw race, and service_utilization before they
    # are transformed / dropped downstream.
    plot_eda(df)

    df = map_age_midpoint(df)
    df = cast_nominal_to_object(df)
    df = add_nummed(df)

    # Skew/kurtosis report only (no data mutation), matching the notebook.
    log_transform_report(df, numeric_columns(df))

    df = drop_redundant_columns(df)
    # Capture the numeric set BEFORE drugs are recast to int, so only the
    # continuous features get standardized later.
    numerics = numeric_columns(df)
    df = recast_numeric(df)

    # No-op after binary target encoding; preserved for parity with the notebook.
    df["readmitted"] = df["readmitted"].apply(lambda x: 0 if x == 2 else x)

    df = df.drop(columns=DIAG_DROP_COLS)
    df = add_interaction_terms(df)

    df2 = df.drop_duplicates(subset=["patient_nbr"], keep="first")
    df2 = standardize_and_remove_outliers(df2, numerics)

    plot_correlation(df2)

    return one_hot_encode(df2)


def _print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {name} ===")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")


def run() -> dict[str, dict[str, float]]:
    """Run the full pipeline and return per-model metrics."""
    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    df_pd = preprocess(df)

    x, y = build_feature_matrix(df_pd)
    print(f"\nFeature matrix: {x.shape[0]} rows x {x.shape[1]} features")
    print("Target distribution:")
    print(y.value_counts())

    results: dict[str, dict[str, float]] = {}

    # Logistic regression on SMOTE-balanced data (notebook cells 56-63).
    x_train, _x_test, y_train, _y_test = split(x, y)
    x_res, y_res = resample_smote(x_train, y_train)
    x_tr, x_te, y_tr, y_te = split(x_res, y_res)
    logit = train_logistic(x_tr, y_tr)
    results["Logistic Regression"] = evaluate(logit, x_te, y_te)
    _print_metrics("Logistic Regression", results["Logistic Regression"])

    # Decision tree reuses the logistic SMOTE split (notebook cells 66-71).
    dtree = train_decision_tree(x_tr, y_tr)
    results["Decision Tree"] = evaluate(dtree, x_te, y_te)
    _print_metrics("Decision Tree", results["Decision Tree"])

    # Random forest on a fresh full-data SMOTE split (notebook cells 72-75).
    x_res2, y_res2 = resample_smote(x, y)
    x_tr2, x_te2, y_tr2, y_te2 = split(x_res2, y_res2)
    forest = train_random_forest(x_tr2, y_tr2)
    results["Random Forests"] = evaluate(forest, x_te2, y_te2)
    _print_metrics("Random Forests", results["Random Forests"])

    plot_model_comparison(results)
    print("\nPlots written to ./outputs/")
    return results


if __name__ == "__main__":
    run()
