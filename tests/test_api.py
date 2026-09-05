"""API endpoint tests for the Safiri FastAPI service.

Tests run against the ASGI app directly through starlette's TestClient, so no
server process is needed. The ``/predict`` endpoint loads the persisted model
artifacts from ``models/``; those must exist (trained by ``src/models/train.py``)
before this module runs.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

# A complete, valid payload for the current /predict contract. Every planned
# schedule field required to compute schedule_slack is included.
VALID_PAYLOAD = {
    "origin": "Santos",
    "destination": "Hamburg",
    "scheduled_port_arrival": "2026-04-29T01:15:15.521823",
    "scheduled_delivery": "2026-05-04T06:47:43.483745",
    "customs_processing_hours": 13.58,
    "inland_transit_hours": 17.55,
    "departure_delay_hours": 1.11,
    "port_delay_hours": 2.76,
    "port_congestion": 3,
    "weather_severity": 2,
    "customs_complexity": 3,
    "document_readiness": 0.195,
    "month": 4,
    "day_of_week": 4,
}

EXPECTED_KEYS = {
    "predicted_delay_hours",
    "predicted_eta",
    "delay_probability",
    "risk_level",
    "contributors",
    "propagation",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload_returns_expected_keys(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert EXPECTED_KEYS.issubset(response.json().keys())


def test_predict_missing_required_field_returns_422(client: TestClient) -> None:
    payload = dict(VALID_PAYLOAD)
    del payload["scheduled_port_arrival"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_out_of_range_severity_returns_422(client: TestClient) -> None:
    payload = dict(VALID_PAYLOAD)
    payload["port_congestion"] = 10  # valid range is 0-3
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_probability_bounded_and_eta_iso_parseable(
    client: TestClient,
) -> None:
    body = client.post("/predict", json=VALID_PAYLOAD).json()
    assert 0.0 <= body["delay_probability"] <= 1.0
    # Parses successfully (raises ValueError) only if ETA is a valid ISO datetime.
    parsed_eta = datetime.fromisoformat(body["predicted_eta"])
    assert parsed_eta.tzinfo is None or parsed_eta.utcoffset() is not None