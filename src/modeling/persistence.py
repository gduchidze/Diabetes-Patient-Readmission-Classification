"""Persist and load the winning model for serving."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.core.config import MODEL_DIR, SAMPLE_INPUT_PATH, WINNER_MODEL_PATH
from src.core.logger import logger


def save_winner(model, feature_columns: list[str], sample_row: dict) -> None:
    """Persist the winning model + its feature order, plus a sample payload."""
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": list(feature_columns)},
        WINNER_MODEL_PATH,
    )
    sample = {key: float(value) for key, value in sample_row.items()}
    Path(SAMPLE_INPUT_PATH).write_text(json.dumps(sample, indent=2))
    logger.info("Saved winner model to {}", WINNER_MODEL_PATH)
    logger.info("Saved sample input to {}", SAMPLE_INPUT_PATH)


def load_winner() -> tuple[object, list[str]]:
    """Load the winning model and its feature column order.

    Raises:
        FileNotFoundError: if the model was never trained/saved.
    """
    path = Path(WINNER_MODEL_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"No saved model at '{path.resolve()}'. "
            "Run `uv run python main.py` first to train and save the winner."
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_columns"]
