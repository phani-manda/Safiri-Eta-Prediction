"""FastAPI service for Safiri shipment predictions."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.predictor import predict_shipment  # noqa: E402


class ShipmentPayload(BaseModel):
    origin: str
    destination: str
    scheduled_port_arrival: datetime
    scheduled_delivery: datetime
    customs_processing_hours: float = Field(ge=0)
    inland_transit_hours: float = Field(ge=0)
    departure_delay_hours: float = Field(ge=0)
    port_delay_hours: float = Field(ge=0)
    port_congestion: int = Field(ge=0, le=3)
    weather_severity: int = Field(ge=0, le=3)
    customs_complexity: int = Field(ge=0, le=3)
    document_readiness: float = Field(ge=0, le=1)
    month: int = Field(ge=1, le=12)
    day_of_week: int = Field(ge=0, le=6)


app = FastAPI(title="Safiri ETA Prediction")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: ShipmentPayload) -> dict:
    return predict_shipment(payload.model_dump())
