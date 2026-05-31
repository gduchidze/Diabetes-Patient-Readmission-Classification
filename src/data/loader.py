"""Dataset loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.config import DATA_PATH


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the diabetes CSV."""
    pd.set_option("future.infer_string", False)
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found at '{csv_path.resolve()}'.")
    return pd.read_csv(csv_path)
