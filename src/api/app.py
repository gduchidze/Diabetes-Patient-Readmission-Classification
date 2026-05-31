"""FastAPI service serving the Random Forest winner (run: uvicorn src.api.app:app)."""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.core.logger import logger
from src.modeling.persistence import load_winner
from src.api.schemas import PredictRequest, PredictResponse

_model, _feature_columns = load_winner()

_LABELS = {0: "not readmitted <30 days", 1: "readmitted <30 days"}

app = FastAPI(title="Diabetes Readmission Classifier", version="1.0.0")


@app.get("/health")
def health() -> dict[str, object]:
    """Service status and feature count."""
    return {"status": "ok", "n_features": len(_feature_columns)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict 30-day readmission from an engineered feature vector."""
    missing = [c for c in _feature_columns if c not in request.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing required features: {missing}")
    row = pd.DataFrame([request.features])[_feature_columns]
    prediction = int(_model.predict(row)[0])
    proba = float(_model.predict_proba(row)[0][1])
    logger.info("Prediction={} p(<30)={:.4f}", prediction, proba)
    return PredictResponse(
        prediction=prediction,
        label=_LABELS[prediction],
        probability_readmit_lt30=proba,
    )
