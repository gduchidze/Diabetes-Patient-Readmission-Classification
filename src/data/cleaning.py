"""Row/column cleaning: drop sparse columns and invalid records."""

from __future__ import annotations

import pandas as pd

from src.core.config import SPARSE_COLS

# discharge_disposition_id 11 = patient expired, so cannot be readmitted.
_EXPIRED_DISCHARGE_ID = 11


def drop_sparse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with too many missing values."""
    return df.drop(columns=SPARSE_COLS)


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing diagnoses/race, expired patients, or unknown gender."""
    invalid = (
        (df["diag_1"] == "?")
        | (df["diag_2"] == "?")
        | (df["diag_3"] == "?")
        | (df["race"] == "?")
        | (df["discharge_disposition_id"] == _EXPIRED_DISCHARGE_ID)
        | (df["gender"] == "Unknown/Invalid")
    )
    return df.loc[~invalid].copy()
