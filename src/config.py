"""Central configuration: paths, column lists, mappings, and thresholds.

Every literal that was scattered across the notebook lives here so the pipeline
modules stay declarative.
"""

from __future__ import annotations

DATA_PATH: str = "diabetic_data.csv"
OUTPUT_DIR: str = "outputs"

SPARSE_COLS: list[str] = [
    "weight",
    "payer_code",
    "medical_specialty",
    "citoglipton",
    "examide",
]

DRUG_KEYS: list[str] = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "insulin",
    "glyburide-metformin",
    "tolazamide",
    "metformin-pioglitazone",
    "metformin-rosiglitazone",
    "glimepiride-pioglitazone",
    "glipizide-metformin",
    "troglitazone",
    "tolbutamide",
    "acetohexamide",
]

ADMISSION_TYPE_MAP: dict[int, int] = {2: 1, 7: 1, 6: 5, 8: 5}
DISCHARGE_DISP_MAP: dict[int, int] = {
    6: 1,
    8: 1,
    9: 1,
    13: 1,
    3: 2,
    4: 2,
    5: 2,
    14: 2,
    22: 2,
    23: 2,
    24: 2,
    12: 10,
    15: 10,
    16: 10,
    17: 10,
    25: 18,
}

AGE_DICT: dict[int, int] = {1: 5, 2: 15, 3: 25, 4: 35, 5: 45, 6: 55, 7: 65, 8: 75, 9: 85, 10: 95}

INTERACTION_TERMS: list[tuple[str, str]] = [
    ("num_medications", "time_in_hospital"),
    ("num_medications", "num_procedures"),
    ("time_in_hospital", "num_lab_procedures"),
    ("num_medications", "num_lab_procedures"),
    ("num_medications", "number_diagnoses"),
    ("age", "number_diagnoses"),
    ("change", "num_medications"),
    ("number_diagnoses", "time_in_hospital"),
    ("num_medications", "numchange"),
]

ONE_HOT_COLS: list[str] = [
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "level1_diag1",
]
NON_NUM_COLS: list[str] = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "level1_diag1",
]

REDUNDANT_COLS: list[str] = [
    "number_outpatient",
    "number_inpatient",
    "number_emergency",
    "service_utilization",
]

DIAG_DROP_COLS: list[str] = [
    "diag_1",
    "diag_2",
    "diag_3",
    "level2_diag1",
    "level1_diag2",
    "level2_diag2",
    "level1_diag3",
    "level2_diag3",
]

FEATURE_SET: list[str] = [
    "age",
    "time_in_hospital",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "AfricanAmerican",
    "Asian",
    "Caucasian",
    "Hispanic",
    "Other",
    "gender_1",
    "admission_type_id_3",
    "admission_type_id_5",
    "discharge_disposition_id_2",
    "discharge_disposition_id_7",
    "discharge_disposition_id_10",
    "discharge_disposition_id_18",
    "admission_source_id_4",
    "admission_source_id_7",
    "admission_source_id_9",
    "num_medications|time_in_hospital",
    "num_medications|num_procedures",
    "time_in_hospital|num_lab_procedures",
    "num_medications|num_lab_procedures",
    "num_medications|number_diagnoses",
    "age|number_diagnoses",
    "change|num_medications",
    "number_diagnoses|time_in_hospital",
    "num_medications|numchange",
]

ZSCORE_THRESHOLD: float = 3.0
SKEW_KURT_THRESHOLD: float = 2.0
LOG_ZERO_FRACTION: float = 0.02
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 0
SMOTE_RANDOM_STATE: int = 20

DTREE_MAX_DEPTH: int = 28
DTREE_MIN_SAMPLES_SPLIT: int = 10
RF_N_ESTIMATORS: int = 10
RF_MAX_DEPTH: int = 25
RF_MIN_SAMPLES_SPLIT: int = 10

# --- MLflow experiment tracking ------------------------------------------
MLFLOW_TRACKING_URI: str = "mlruns"
MLFLOW_EXPERIMENT: str = "diabetes-readmission"
