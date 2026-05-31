"""Numeric transforms: skew reporting, standardization, outlier removal, one-hot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as sp_stats

from src.core.config import (
    LOG_ZERO_FRACTION,
    ONE_HOT_COLS,
    REDUNDANT_COLS,
    SKEW_KURT_THRESHOLD,
    ZSCORE_THRESHOLD,
)


def numeric_columns(df: pd.DataFrame, exclude: tuple[str, ...] = ("readmitted",)) -> list[str]:
    """Numeric column names, minus the excluded ones."""
    numeric = df.select_dtypes(include=["int64", "float64"]).columns
    return list(set(numeric) - set(exclude))


def log_transform_report(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """Report skew/kurtosis per column; does not modify the data."""
    rows: list[dict[str, object]] = []
    for col in num_cols:
        skew_before = df[col].skew()
        kurt_before = df[col].kurtosis()
        sd_before = df[col].std()

        needs_log = (abs(skew_before) > SKEW_KURT_THRESHOLD) and (
            abs(kurt_before) > SKEW_KURT_THRESHOLD
        )

        skew_after = skew_before
        kurt_after = kurt_before
        sd_after = sd_before
        log_type = ""

        if needs_log:
            zero_fraction = len(df[df[col] == 0]) / len(df)
            if zero_fraction <= LOG_ZERO_FRACTION:
                log_type = "log"
                transformed = np.log(df[df[col] > 0][col])
            else:
                log_type = "log1p"
                transformed = np.log1p(df[df[col] >= 0][col])
            skew_after = transformed.skew()
            kurt_after = transformed.kurtosis()
            sd_after = transformed.std()

        rows.append(
            {
                "numeric_column": col,
                "skew_before": skew_before,
                "skew_after": skew_after,
                "kurt_before": kurt_before,
                "kurt_after": kurt_after,
                "std_before": sd_before,
                "std_after": sd_after,
                "log_transform_needed": "Yes" if needs_log else "No",
                "log_type": log_type,
            }
        )
    return pd.DataFrame(rows)


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop raw visit counts now folded into service_utilization."""
    return df.drop(columns=REDUNDANT_COLS)


def standardize_and_remove_outliers(df: pd.DataFrame, numerics: list[str]) -> pd.DataFrame:
    """Z-standardize numeric features and drop rows beyond the z-score threshold."""
    out = df.copy()
    out[numerics] = (out[numerics] - np.mean(out[numerics], axis=0)) / np.std(
        out[numerics], axis=0
    )
    mask = (np.abs(sp_stats.zscore(out[numerics])) < ZSCORE_THRESHOLD).all(axis=1)
    return out[mask].copy()


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categorical columns and race."""
    out = df.copy()
    out["level1_diag1"] = out["level1_diag1"].astype("object")
    encoded = pd.get_dummies(out, columns=ONE_HOT_COLS, drop_first=True)
    race_dummies = pd.get_dummies(encoded["race"])
    encoded = pd.concat([encoded, race_dummies], axis=1)
    encoded = encoded.drop(columns=["race"])
    return encoded
