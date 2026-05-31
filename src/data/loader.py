"""Dataset loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.config import DATA_PATH


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the diabetes dataset as object-dtype strings (notebook semantics).

    pandas 3.0 infers a dedicated string dtype by default, which rejects the
    notebook's int-into-string assignments. Opting out restores object columns.
    """
    # Escape hatch: keep classic object dtype for string columns.
    pd.set_option("future.infer_string", False)
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path.resolve()}'. "
            "Place diabetic_data.csv in the project root or pass an explicit path."
        )
    return pd.read_csv(csv_path)
