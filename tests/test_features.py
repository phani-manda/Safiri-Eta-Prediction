"""Unit tests for build_features() in src/features/engineering.py.

Each test builds a single-row raw frame with known values and asserts the
engineered output matches the documented formula for that row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import REQUIRED_RAW_COLUMNS, build_features


def make_raw_frame(**overrides) -> pd.DataFrame:
    """One canonical-schema raw row, with any field overridable."""
    defaults = {
        "route": "Santos -> Hamburg",
        "scheduled_departure": "2026-03-01 08:00:00",
        "scheduled_port_arrival": "2026-03-05 00:00:00",
        "scheduled_delivery": "2026-03-07 00:00:00",
        "customs_processing_hours": 24.0,
        "inland_transit_hours": 36.0,
        "departure_delay_hours": 1.5,
        "port_delay_hours": 2.25,
        "port_congestion": 2,
        "weather_severity": 1,
        "customs_complexity": 2,
        "document_readiness": 0.7,
        "actual_port_arrival_missing": 0,
        "total_delay_hours": 8.0,
        "is_delayed": True,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


def test_raw_frame_builder_covers_required_columns() -> None:
    missing = REQUIRED_RAW_COLUMNS - set(make_raw_frame().columns)
    assert not missing, f"fixture missing required raw columns: {sorted(missing)}"


def test_cumulative_delay_equals_sum_of_observed_stage_delays() -> None:
    raw = make_raw_frame(departure_delay_hours=1.5, port_delay_hours=2.25)
    features = build_features(raw)
    assert features.loc[0, "cumulative_delay_so_far"] == pytest.approx(3.75)


def test_schedule_slack_matches_canonical_formula() -> None:
    # Window: 2026-01-01 00:00 -> 2026-01-04 00:00 = 72 h. Planned work:
    # customs 10 h + inland 20 h. Canonical slack = 72 - 30 = 42 h.
    raw = make_raw_frame(
        scheduled_port_arrival="2026-01-01 00:00:00",
        scheduled_delivery="2026-01-04 00:00:00",
        customs_processing_hours=10.0,
        inland_transit_hours=20.0,
    )
    features = build_features(raw)
    assert features.loc[0, "schedule_slack"] == pytest.approx(42.0)


def test_missing_actual_port_arrival_sets_port_arrival_missing() -> None:
    raw = make_raw_frame(actual_port_arrival_missing=1, port_delay_hours=np.nan)
    features = build_features(raw)
    assert features.loc[0, "port_arrival_missing"] == 1