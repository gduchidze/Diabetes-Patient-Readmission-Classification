"""Request/response schemas for the prediction API."""

from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    features: dict[str, float]

    @field_validator("features")
    @classmethod
    def _non_empty(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("features must not be empty")
        return value


class PredictResponse(BaseModel):
    prediction: int
    label: str
    probability_readmit_lt30: float