"""Central configuration: paths, column lists, mappings, and thresholds.

Every literal that was scattered across the notebook lives here so the pipeline
modules stay declarative.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

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

# --- Validated runtime settings ------------------------------------------
class Settings(BaseModel):
    """Validated scalar config: tunables, paths, tracking, and serving.

    Column lists/mappings above are structural and stay as module constants;
    these scalars carry invariants worth enforcing with pydantic validators.
    """

    # Paths
    data_path: str = "diabetic_data.csv"
    output_dir: str = "outputs"

    # Transform thresholds
    zscore_threshold: float = 3.0
    skew_kurt_threshold: float = 2.0
    log_zero_fraction: float = 0.02

    # Split / resampling
    test_size: float = 0.20
    random_state: int = 0
    smote_random_state: int = 20

    # Model hyperparameters
    dtree_max_depth: int = 28
    dtree_min_samples_split: int = 10
    rf_n_estimators: int = 10
    rf_max_depth: int = 25
    rf_min_samples_split: int = 10

    # MLflow tracking
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment: str = "diabetes-readmission"

    # Model serving (FastAPI)
    model_dir: str = "models"
    winner_model_path: str = "models/winner_random_forest.joblib"
    sample_input_path: str = "models/sample_input.json"

    @field_validator("data_path")
    @classmethod
    def _csv_path(cls, v: str) -> str:
        if not v.endswith(".csv"):
            raise ValueError("data_path must point to a .csv file")
        return v

    @field_validator("test_size", "log_zero_fraction")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("must be strictly between 0 and 1")
        return v

    @field_validator("zscore_threshold", "skew_kurt_threshold")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be greater than 0")
        return v

    @field_validator("random_state", "smote_random_state")
    @classmethod
    def _non_negative_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @field_validator(
        "dtree_max_depth",
        "dtree_min_samples_split",
        "rf_n_estimators",
        "rf_max_depth",
        "rf_min_samples_split",
    )
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be greater than 0")
        return v

    @field_validator("mlflow_experiment")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


settings = Settings()

DATA_PATH: str = settings.data_path
OUTPUT_DIR: str = settings.output_dir
ZSCORE_THRESHOLD: float = settings.zscore_threshold
SKEW_KURT_THRESHOLD: float = settings.skew_kurt_threshold
LOG_ZERO_FRACTION: float = settings.log_zero_fraction
TEST_SIZE: float = settings.test_size
RANDOM_STATE: int = settings.random_state
SMOTE_RANDOM_STATE: int = settings.smote_random_state
DTREE_MAX_DEPTH: int = settings.dtree_max_depth
DTREE_MIN_SAMPLES_SPLIT: int = settings.dtree_min_samples_split
RF_N_ESTIMATORS: int = settings.rf_n_estimators
RF_MAX_DEPTH: int = settings.rf_max_depth
RF_MIN_SAMPLES_SPLIT: int = settings.rf_min_samples_split
MLFLOW_TRACKING_URI: str = settings.mlflow_tracking_uri
MLFLOW_EXPERIMENT: str = settings.mlflow_experiment
MODEL_DIR: str = settings.model_dir
WINNER_MODEL_PATH: str = settings.winner_model_path
SAMPLE_INPUT_PATH: str = settings.sample_input_path
