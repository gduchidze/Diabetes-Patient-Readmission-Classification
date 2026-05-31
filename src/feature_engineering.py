"""Feature engineering: derived features, recoding, encoding, interactions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    ADMISSION_TYPE_MAP,
    AGE_DICT,
    DISCHARGE_DISP_MAP,
    DRUG_KEYS,
    INTERACTION_TERMS,
)

_STABLE_DRUG_STATES = ("No", "Steady")
_ACTIVE_DRUG_STATES = ("Steady", "Up", "Down")


def add_service_utilization(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["service_utilization"] = (
        out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]
    )
    return out


def add_numchange(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numchange = pd.Series(0, index=out.index)
    for col in DRUG_KEYS:
        numchange = numchange + out[col].apply(
            lambda x: 0 if x in _STABLE_DRUG_STATES else 1
        )
    out["numchange"] = numchange
    return out


def recode_admission_discharge(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["admission_type_id"] = out["admission_type_id"].replace(ADMISSION_TYPE_MAP)
    out["discharge_disposition_id"] = out["discharge_disposition_id"].replace(
        DISCHARGE_DISP_MAP
    )
    return out


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["change"] = out["change"].replace({"Ch": 1, "No": 0})
    out["gender"] = out["gender"].replace({"Male": 1, "Female": 0})
    out["diabetesMed"] = out["diabetesMed"].replace({"Yes": 1, "No": 0})
    drug_map = {"No": 0}
    drug_map.update({state: 1 for state in _ACTIVE_DRUG_STATES})
    for col in DRUG_KEYS:
        out[col] = out[col].replace(drug_map)
    return out


def encode_lab_results(df: pd.DataFrame) -> pd.DataFrame:
    """Encode A1Cresult and max_glu_serum (notebook cell 16)."""
    out = df.copy()
    out["A1Cresult"] = out["A1Cresult"].replace(
        {">7": 1, ">8": 1, "Norm": 0, "None": -99}
    )
    out["max_glu_serum"] = out["max_glu_serum"].replace(
        {">200": 1, ">300": 1, "Norm": 0, "None": -99}
    )
    return out


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["readmitted"] = out["readmitted"].replace({">30": 0, "<30": 1, "NO": 0})
    return out


def map_age_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for i in range(10):
        bucket = f"[{10 * i}-{10 * (i + 1)})"
        out["age"] = out["age"].replace(bucket, i + 1)
    return out


def map_age_midpoint(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age"] = out["age"].astype("int64").map(AGE_DICT)
    return out


def _categorize_diagnosis(value: float) -> int:
    if (390 <= value < 460) or (np.floor(value) == 785):
        return 1
    if (460 <= value < 520) or (np.floor(value) == 786):
        return 2
    if (520 <= value < 580) or (np.floor(value) == 787):
        return 3
    if np.floor(value) == 250:
        return 4
    if 800 <= value < 1000:
        return 5
    if 710 <= value < 740:
        return 6
    if (580 <= value < 630) or (np.floor(value) == 788):
        return 7
    if 140 <= value < 240:
        return 8
    return 0


def _categorize_level2_diagnosis(value: float) -> int:
    if 390 <= value < 399:
        return 1
    if 401 <= value < 415:
        return 2
    if 415 <= value < 460:
        return 3
    if np.floor(value) == 785:
        return 4
    if 460 <= value < 489:
        return 5
    if 490 <= value < 497:
        return 6
    if 500 <= value < 520:
        return 7
    if np.floor(value) == 786:
        return 8
    if 520 <= value < 530:
        return 9
    if 530 <= value < 544:
        return 10
    if 550 <= value < 554:
        return 11
    if 555 <= value < 580:
        return 12
    if np.floor(value) == 787:
        return 13
    if np.floor(value) == 250:
        return 14
    if 800 <= value < 1000:
        return 15
    if 710 <= value < 740:
        return 16
    if 580 <= value < 630:
        return 17
    if np.floor(value) == 788:
        return 18
    if 140 <= value < 240:
        return 19
    if 240 <= value < 280 and np.floor(value) != 250:
        return 20
    if (680 <= value < 710) or (np.floor(value) == 782):
        return 21
    if 290 <= value < 320:
        return 22
    return 0


def build_diagnosis_levels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for src in ("diag_1", "diag_2", "diag_3"):
        idx = src[-1]
        out[f"level1_diag{idx}"] = out[src].astype(object)
        out[f"level2_diag{idx}"] = out[src].astype(object)
        is_supplementary = out[src].str.contains("V") | out[src].str.contains("E")
        out.loc[is_supplementary, [f"level1_diag{idx}", f"level2_diag{idx}"]] = 0
        for level in ("level1", "level2"):
            col = f"{level}_diag{idx}"
            out[col] = out[col].replace("?", -1).astype(float)
    return out


def categorize_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for idx in ("1", "2", "3"):
        out[f"level1_diag{idx}"] = out[f"level1_diag{idx}"].apply(_categorize_diagnosis)
        out[f"level2_diag{idx}"] = out[f"level2_diag{idx}"].apply(
            _categorize_level2_diagnosis
        )
    return out


def cast_nominal_to_object(df: pd.DataFrame) -> pd.DataFrame:
    """Cast nominal/categorical columns to object dtype
    """
    out = df.copy()
    nominal = [
        "encounter_id",
        "patient_nbr",
        "gender",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        *DRUG_KEYS,
        "change",
        "diabetesMed",
        "age",
        "A1Cresult",
        "max_glu_serum",
        "level1_diag1",
        "level1_diag2",
        "level1_diag3",
        "level2_diag1",
        "level2_diag2",
        "level2_diag3",
    ]
    present = [c for c in nominal if c in out.columns]
    out[present] = out[present].astype("object")
    return out


def add_nummed(df: pd.DataFrame) -> pd.DataFrame:
    """Sum the binary drug indicators into a single count"""
    out = df.copy()
    nummed = pd.Series(0, index=out.index)
    for col in DRUG_KEYS:
        nummed = nummed + out[col]
    out["nummed"] = nummed
    return out


def recast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Restore int64 dtype on selected columns

    Runs after the numeric column set for standardization has been captured, so
    drug indicators rejoin the frame as ints without being standardized.
    """
    out = df.copy()
    for col in ("encounter_id", "patient_nbr", "diabetesMed", "change"):
        out[col] = out[col].astype("int64")
    drug_like = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
        "A1Cresult",
    ]
    out[drug_like] = out[drug_like].fillna(-1).astype("int64")
    return out


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Create pairwise product interaction features"""
    out = df.copy()
    for left, right in INTERACTION_TERMS:
        out[f"{left}|{right}"] = out[left] * out[right]
    return out
