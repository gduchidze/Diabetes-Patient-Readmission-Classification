"""End-to-end diabetes readmission pipeline: load, train, track, save winner."""

from __future__ import annotations

from src.core.config import DIAG_DROP_COLS
from src.data.cleaning import drop_invalid_rows, drop_sparse_columns
from src.data.loader import load_data
from src.core.logger import logger
from src.modeling.persistence import save_winner
from src.data.feature_engineering import (
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
from src.modeling.estimators import (
    build_feature_matrix,
    evaluate,
    model_params,
    resample_smote,
    split,
    train_decision_tree,
    train_logistic,
    train_random_forest,
)
from src.modeling.tracking import init_tracking, log_model_run
from src.data.transform import (
    drop_redundant_columns,
    log_transform_report,
    numeric_columns,
    one_hot_encode,
    standardize_and_remove_outliers,
)
from src.visualization.plots import plot_correlation, plot_eda, plot_model_comparison


def preprocess(df):
    """Clean and feature-engineer the raw frame into the model-ready matrix."""
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

    plot_eda(df)

    df = map_age_midpoint(df)
    df = cast_nominal_to_object(df)
    df = add_nummed(df)
    log_transform_report(df, numeric_columns(df))

    df = drop_redundant_columns(df)
    numerics = numeric_columns(df)
    df = recast_numeric(df)

    df["readmitted"] = df["readmitted"].apply(lambda x: 0 if x == 2 else x)

    df = df.drop(columns=DIAG_DROP_COLS)
    df = add_interaction_terms(df)

    df2 = df.drop_duplicates(subset=["patient_nbr"], keep="first")
    df2 = standardize_and_remove_outliers(df2, numerics)

    plot_correlation(df2)

    return one_hot_encode(df2)


def _log_metrics(name: str, metrics: dict[str, float]) -> None:
    logger.info(
        "{} | acc={:.4f} prec={:.4f} recall={:.4f} f1={:.4f}",
        name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )


def run() -> dict[str, dict[str, float]]:
    """Run the full pipeline and return per-model metrics."""
    df_pd = preprocess(load_data())
    init_tracking()

    x, y = build_feature_matrix(df_pd)
    n_features = x.shape[1]

    results: dict[str, dict[str, float]] = {}

    x_train, _x_test, y_train, _y_test = split(x, y)
    x_res, y_res = resample_smote(x_train, y_train)
    x_tr, x_te, y_tr, y_te = split(x_res, y_res)
    logit = train_logistic(x_tr, y_tr)
    results["Logistic Regression"] = evaluate(logit, x_te, y_te)
    _log_metrics("Logistic Regression", results["Logistic Regression"])
    log_model_run(
        "Logistic Regression",
        logit,
        model_params("Logistic Regression", n_features),
        results["Logistic Regression"],
        x_te.head(),
    )

    dtree = train_decision_tree(x_tr, y_tr)
    results["Decision Tree"] = evaluate(dtree, x_te, y_te)
    _log_metrics("Decision Tree", results["Decision Tree"])
    log_model_run(
        "Decision Tree",
        dtree,
        model_params("Decision Tree", n_features),
        results["Decision Tree"],
        x_te.head(),
    )

    x_res2, y_res2 = resample_smote(x, y)
    x_tr2, x_te2, y_tr2, y_te2 = split(x_res2, y_res2)
    forest = train_random_forest(x_tr2, y_tr2)
    results["Random Forests"] = evaluate(forest, x_te2, y_te2)
    _log_metrics("Random Forests", results["Random Forests"])
    log_model_run(
        "Random Forests",
        forest,
        model_params("Random Forests", n_features),
        results["Random Forests"],
        x_te2.head(),
    )

    save_winner(forest, list(x_tr2.columns), x_te2.iloc[0].to_dict())

    plot_model_comparison(results)
    return results


if __name__ == "__main__":
    run()
