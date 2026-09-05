"""Leakage-boundary tests for src/features/engineering.py ``build_features()``.

The prediction cutoff is the moment ``actual_port_arrival`` is observed. Anything
that describes the world after that instant is permanently forbidden as a feature,
so these tests assert -- as set memberships, not visual checks -- that no
post-cutoff column ever appears in the engineered output.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.engineering import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    POST_CUTOFF_COLUMNS,
    TARGET_COLUMNS,
    build_features,
)

# The specific columns the container cannot tolerate in its output, per the
# canonical schema. "is_delayed" itself is the label and is expected to stay.
FORBIDDEN_COLUMNS = frozenset(
    {
        "customs_delay_hours",
        "inland_delay_hours",
        "actual_customs_clearance",
        "actual_delivery",
        "actual_customs_clearance_missing",
        "actual_delivery_missing",
    }
)


def make_raw_frame() -> pd.DataFrame:
    """A single valid canonical-schema raw row."""
    row = {
        "route": "Shanghai -> Rotterdam",
        "scheduled_departure": "2026-01-15 12:00:00",
        "scheduled_port_arrival": "2026-01-20 12:00:00",
        "scheduled_delivery": "2026-01-25 12:00:00",
        "customs_processing_hours": 30.0,
        "inland_transit_hours": 40.0,
        "departure_delay_hours": 0.5,
        "port_delay_hours": 1.0,
        "port_congestion": 1,
        "weather_severity": 1,
        "customs_complexity": 1,
        "document_readiness": 0.5,
        "actual_port_arrival_missing": 0,
        "total_delay_hours": 5.0,
        "is_delayed": False,
    }
    return pd.DataFrame([row])


def test_no_post_cutoff_column_appears_in_output() -> None:
    features = build_features(make_raw_frame())
    leaked = FORBIDDEN_COLUMNS & set(features.columns)
    assert leaked == frozenset(), f"leaking columns present in output: {sorted(leaked)}"


def test_output_matches_documented_allow_list_exactly() -> None:
    features = build_features(make_raw_frame())
    allowed = set(FEATURE_COLUMNS) | set(TARGET_COLUMNS) | set(METADATA_COLUMNS)
    assert set(features.columns) == allowed, (
        f"unexpected columns in feature frame: {sorted(set(features.columns) - allowed)}"
    )


def test_post_cutoff_set_is_broad_than_forbidden_list() -> None:
    # Keep the two guards in step: everything FORBIDDEN_COLUMNS names must also be
    # listed in the module's canonical POST_CUTOFF_COLUMNS.
    assert FORBIDDEN_COLUMNS.issubset(POST_CUTOFF_COLUMNS)