"""Inference wrapper for shipment ETA and delay-risk predictions."""

from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.explainability.explainer import (  # noqa: E402
    explain_prediction,
    generate_propagation_sentence,
    get_feature_importances,
)
from src.models.train import CLASSIFIER_MODEL_PATH, MODEL_PATH  # noqa: E402


@lru_cache(maxsize=1)
def _load_models():
    eta_model = joblib.load(MODEL_PATH)
    delay_classifier = joblib.load(CLASSIFIER_MODEL_PATH)
    eta_importances = get_feature_importances(eta_model, eta_model.feature_names_in_)
    return eta_model, delay_classifier, eta_importances


def _risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    if probability <= 0.6:
        return "MEDIUM"
    return "HIGH"


def _build_feature_row(payload: dict, feature_names: list[str]) -> pd.DataFrame:
    origin = payload["origin"]
    destination = payload["destination"]
    departure_delay = float(payload["departure_delay_hours"])
    port_delay = float(payload["port_delay_hours"])
    scheduled_port_arrival = pd.to_datetime(payload["scheduled_port_arrival"])
    scheduled_delivery = pd.to_datetime(payload["scheduled_delivery"])
    planned_window_hours = (
        scheduled_delivery - scheduled_port_arrival
    ).total_seconds() / 3600.0
    schedule_slack = planned_window_hours - (
        float(payload["customs_processing_hours"])
        + float(payload["inland_transit_hours"])
    )

    values = {
        "route": f"{origin} -> {destination}",
        "month": int(payload["month"]),
        "day_of_week": int(payload["day_of_week"]),
        "departure_delay_hours": departure_delay,
        "port_delay_hours": port_delay,
        "port_congestion": int(payload["port_congestion"]),
        "weather_severity": int(payload["weather_severity"]),
        "customs_complexity": int(payload["customs_complexity"]),
        "document_readiness": float(payload["document_readiness"]),
        "port_arrival_missing": int(payload.get("port_arrival_missing", 0)),
        "cumulative_delay_so_far": departure_delay + port_delay,
        "schedule_slack": schedule_slack,
        "previous_stage_delay": port_delay,
    }
    return pd.DataFrame([values], columns=feature_names)


def predict_shipment(payload: dict) -> dict:
    """Predict delay hours, ETA, risk probability, and explanations."""
    eta_model, delay_classifier, eta_importances = _load_models()
    feature_names = list(eta_model.feature_names_in_)
    feature_frame = _build_feature_row(payload, feature_names)
    feature_row = feature_frame.iloc[0]

    predicted_delay_hours = float(eta_model.predict(feature_frame)[0])
    delay_probability = float(delay_classifier.predict_proba(feature_frame)[0, 1])
    scheduled_delivery = pd.to_datetime(payload["scheduled_delivery"]).to_pydatetime()
    predicted_eta = scheduled_delivery + timedelta(hours=predicted_delay_hours)

    return {
        "predicted_delay_hours": predicted_delay_hours,
        "predicted_eta": predicted_eta.isoformat(),
        "delay_probability": delay_probability,
        "risk_level": _risk_level(delay_probability),
        "contributors": explain_prediction(
            eta_model, feature_row, eta_importances, top_n=3
        ),
        "propagation": generate_propagation_sentence(feature_row),
    }
