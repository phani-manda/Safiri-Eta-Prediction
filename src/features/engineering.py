"""Feature construction for the Safiri ETA model.

Everything in this module exists to serve one constraint: the **leakage
boundary**. Prediction happens at the instant `actual_port_arrival` is observed
-- the shipment has departed and docked, but has not yet cleared customs or
started inland transport. Any field describing what happened *after* that
instant is unknowable in production, so using it as an input would produce a
model that scores beautifully offline and is worthless in service.

Leakage is dangerous precisely because it is silent: nothing crashes, no warning
is emitted, the metrics simply improve. So the boundary here is expressed as
data (`POST_CUTOFF_COLUMNS`) and enforced at runtime (`_assert_no_leakage`)
rather than left as a comment for a future reader to honour.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns the model is allowed to see, in output order. Every one of these is
# observable at or before the moment of port arrival.
FEATURE_COLUMNS: tuple[str, ...] = (
    "route",
    "month",
    "day_of_week",
    "departure_delay_hours",
    "port_delay_hours",
    "port_congestion",
    "weather_severity",
    "customs_complexity",
    "document_readiness",
    "port_arrival_missing",
    "cumulative_delay_so_far",
    "schedule_slack",
    "previous_stage_delay",
)

# Carried through the feature frame for bookkeeping only -- never a model input.
# The temporal split in src/models/train.py needs a chronological key, and
# recovering it by re-reading the raw file invites row-order drift between the two.
#
# Deliberately not a feature: month and day_of_week already expose the useful
# calendar signal, whereas the raw timestamp is monotonic, so a model could fit a
# time trend against it and score well by learning "later rows have index N" --
# exactly the illusion the temporal split exists to expose.
METADATA_COLUMNS: tuple[str, ...] = ("scheduled_departure",)

# Carried in the output frame so downstream code has labels to train against,
# but never legal as an input to the model itself.
TARGET_COLUMNS: tuple[str, ...] = (
    "total_delay_hours",
    "is_delayed",
)

# Describes the world after the cutoff. Must never appear in the feature frame
# under any name -- not as a value, a timestamp, or a missingness flag.
POST_CUTOFF_COLUMNS: frozenset[str] = frozenset(
    {
        "customs_delay_hours",
        "inland_delay_hours",
        "actual_customs_clearance",
        "actual_delivery",
        "actual_customs_clearance_missing",
        "actual_delivery_missing",
    }
)

# Raw columns build_features reads. Checked up front so a schema drift produces
# one clear error naming everything absent, instead of a KeyError on whichever
# line happened to touch the missing column first.
REQUIRED_RAW_COLUMNS: frozenset[str] = frozenset(
    {
        "route",
        "scheduled_departure",
        "scheduled_port_arrival",
        "scheduled_delivery",
        "customs_processing_hours",
        "inland_transit_hours",
        "departure_delay_hours",
        "port_delay_hours",
        "port_congestion",
        "weather_severity",
        "customs_complexity",
        "document_readiness",
        "actual_port_arrival_missing",
        "total_delay_hours",
        "is_delayed",
    }
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Project the raw shipments frame down to cutoff-legal features + targets.

    Builds a brand-new frame rather than dropping columns from a copy of `df`.
    Allow-listing is the safer direction for a leakage boundary: if the
    generator later gains a column, an allow-list silently ignores it, whereas a
    deny-list would wave it straight through into the model.

    Args:
        df: Raw shipments, as loaded from data/raw/shipments.csv.

    Returns:
        A new frame with the columns of `FEATURE_COLUMNS` followed by
        `TARGET_COLUMNS`, sharing `df`'s index.

    Raises:
        KeyError: if `df` lacks a column this function needs.
        AssertionError: if a post-cutoff column somehow reached the output.
    """
    missing = sorted(REQUIRED_RAW_COLUMNS - set(df.columns))
    if missing:
        raise KeyError(f"raw dataframe is missing required columns: {missing}")

    out = pd.DataFrame(index=df.index)

    out["route"] = df["route"]

    # Calendar features are derived from the *scheduled* departure, not the
    # actual one. The schedule is fixed at booking time, so these are known long
    # before the cutoff and stay available even for shipments whose actual
    # timestamps were never recorded.
    scheduled_departure = pd.to_datetime(df["scheduled_departure"], errors="coerce")
    out["month"] = scheduled_departure.dt.month
    out["day_of_week"] = scheduled_departure.dt.dayofweek  # Monday=0 .. Sunday=6

    # Both stage delays that have actually been observed by the cutoff.
    out["departure_delay_hours"] = df["departure_delay_hours"]
    out["port_delay_hours"] = df["port_delay_hours"]

    # Operational conditions: known or forecast at the cutoff, not outcomes.
    out["port_congestion"] = df["port_congestion"]
    out["weather_severity"] = df["weather_severity"]
    out["customs_complexity"] = df["customs_complexity"]
    out["document_readiness"] = df["document_readiness"]

    # Only the port-arrival flag survives the boundary. The customs and delivery
    # missingness flags describe post-cutoff records, and whether such a record
    # eventually gets written correlates with how the shipment ended -- so they
    # would leak the outcome through the back door.
    out["port_arrival_missing"] = df["actual_port_arrival_missing"]

    # How far behind the shipment already is at the cutoff. Both terms are
    # observed, so the sum is legal, and it is the quantity that propagates:
    # hours lost upstream compress the time left for customs and inland transit.
    out["cumulative_delay_so_far"] = df["departure_delay_hours"] + df["port_delay_hours"]

    # How much room the plan leaves between docking and the delivery promise, once
    # the work still owed inside that window is subtracted. Positive is genuine
    # buffer that can absorb an upstream delay; negative means the plan was already
    # infeasible before customs began, so the shipment arrives late even if every
    # remaining stage runs exactly to its planned duration.
    #
    # Cutoff-legal despite describing stages that have not happened yet: the two
    # timestamps are fixed at booking, and customs_processing_hours /
    # inland_transit_hours are *planned baseline* durations, not observed actuals.
    # The observed counterparts (customs_delay_hours, inland_delay_hours) are the
    # forbidden ones, and neither is touched here.
    scheduled_port_arrival = pd.to_datetime(df["scheduled_port_arrival"], errors="coerce")
    scheduled_delivery = pd.to_datetime(df["scheduled_delivery"], errors="coerce")
    planned_window_hours = (
        scheduled_delivery - scheduled_port_arrival
    ).dt.total_seconds() / 3600.0
    out["schedule_slack"] = planned_window_hours - (
        df["customs_processing_hours"] + df["inland_transit_hours"]
    )

    # Identical to port_delay_hours at this cutoff, and carried as its own column
    # deliberately. "The most recent observed stage delay" is the concept the
    # propagation story actually depends on; once a later cutoff is added (after
    # customs clearance, say) it becomes customs_delay_hours instead, and every
    # consumer reading previous_stage_delay keeps working without modification.
    out["previous_stage_delay"] = df["port_delay_hours"]

    out["total_delay_hours"] = df["total_delay_hours"]
    out["is_delayed"] = df["is_delayed"]

    # Sort key for the temporal split. Kept as a real timestamp rather than a
    # string so consumers can sort without reparsing.
    out["scheduled_departure"] = scheduled_departure

    out = out[list(METADATA_COLUMNS) + list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)]
    _assert_no_leakage(out)
    return out


def _assert_no_leakage(features: pd.DataFrame) -> None:
    """Fail loudly if the feature frame drifted away from the cutoff contract.

    Checks both directions. A post-cutoff column present is outright leakage; an
    unrecognised column is a weaker smell, but it means someone added a feature
    without deciding whether it is knowable at the cutoff -- which is exactly how
    leakage arrives in practice. Crashing here is far cheaper than tracing an
    implausibly good validation score back to its source weeks later.
    """
    leaked = sorted(POST_CUTOFF_COLUMNS.intersection(features.columns))
    if leaked:
        raise AssertionError(f"post-cutoff columns leaked into features: {leaked}")

    allowed = set(FEATURE_COLUMNS) | set(TARGET_COLUMNS) | set(METADATA_COLUMNS)
    unexpected = sorted(set(features.columns) - allowed)
    if unexpected:
        raise AssertionError(f"unvetted columns in feature frame: {unexpected}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    raw_path = repo_root / "data" / "raw" / "shipments.csv"
    out_path = repo_root / "data" / "processed" / "features_v1.csv"

    raw = pd.read_csv(raw_path)
    features = build_features(raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)

    print(f"raw      {raw_path.relative_to(repo_root).as_posix():34s} "
          f"{raw.shape[0]} rows x {raw.shape[1]} cols")
    print(f"features {out_path.relative_to(repo_root).as_posix():34s} "
          f"{features.shape[0]} rows x {features.shape[1]} cols")

    print(f"\n{len(FEATURE_COLUMNS)} FEATURE columns (model inputs):")
    for name in FEATURE_COLUMNS:
        print(f"  {name:26s} {features[name].dtype}")

    print(f"\n{len(TARGET_COLUMNS)} TARGET columns (labels only, never inputs):")
    for name in TARGET_COLUMNS:
        print(f"  {name:26s} {features[name].dtype}")

    print(f"\n{len(METADATA_COLUMNS)} METADATA column (split key only, never an input):")
    for name in METADATA_COLUMNS:
        print(f"  {name:26s} {features[name].dtype}")

    # Raw columns that carry through under a new name are absent from
    # features.columns, so report them separately rather than letting them look
    # like they were dropped.
    renamed = {"actual_port_arrival_missing": "port_arrival_missing"}

    excluded = sorted(set(raw.columns) - set(features.columns) - set(renamed))
    print(f"\n{len(excluded)} raw columns excluded:")
    for name in excluded:
        reason = "POST-CUTOFF (leakage)" if name in POST_CUTOFF_COLUMNS else "not requested yet"
        print(f"  {name:34s} {reason}")

    print(f"\n{len(renamed)} raw column kept under a new name:")
    for old, new in renamed.items():
        print(f"  {old:34s} -> {new}")

    breaches = sorted(POST_CUTOFF_COLUMNS.intersection(features.columns))
    print(f"\nleakage check: {'FAILED -> ' + str(breaches) if breaches else 'PASS (no post-cutoff column present)'}")

    slack = features["schedule_slack"]
    print("\nschedule_slack .describe():")
    print(slack.describe().round(2).to_string())

    n_negative = int((slack < 0).sum())
    print(
        f"\n  negative slack (no room left): {n_negative} / {len(slack)} "
        f"({n_negative / len(slack):.1%})"
    )
    print(f"  spread (max - min):            {slack.max() - slack.min():.2f} h")
