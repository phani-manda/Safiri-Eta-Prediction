"""
Synthetic shipment dataset generator for the Safiri take-home assignment.

Assignment constraints followed:
- 100-300 shipment records (we generate 300)
- Fields: origin/destination, scheduled + actual timestamps per stage,
  port processing times, per-stage delays, external factors
  (congestion/weather/customs complexity/document readiness)
- Delays are NOT independent random noise: each stage's delay is generated
  as a function of the upstream stage's delay plus a local severity effect
  plus noise, so that stage-to-stage propagation genuinely exists in the
  ground truth and is discoverable by a model (per PRD section 1.7 / 2.A).
- ~10-15% of intermediate timestamps are simulated as missing, with
  explicit missingness indicator columns (per PRD section 1.7 / suggested
  challenge "incomplete or missing intermediate data" in the brief).
- port_delay_hours is nulled alongside a missing actual_port_arrival, so the
  simulated gap actually bites on the input side (see the missingness block).
- Delay magnitudes are scaled by DELAY_SCALE so that the canonical 6-hour
  is_delayed threshold produces a usable class balance rather than a 98%
  positive class (see DELAY_SCALE for why this preserves the causal structure).

Schema matches the "Canonical schema & conventions" section of the build
prompts file exactly, including is_delayed == total_delay_hours > 6.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SHIPMENTS = 300
SEED = 42
PROPAGATION_COEF = 0.55  # fraction of upstream delay that carries into the next stage
MISSING_PROB = 0.12

# Fixed by the canonical schema: is_delayed == total_delay_hours > 6.
DELAY_THRESHOLD_HOURS = 6.0

# The local-severity ranges below were tuned against a 24-hour operating point,
# which put ~41% of shipments in the "delayed" class -- a healthy balance for a
# classifier. Applying the canonical 6h threshold to that same raw scale instead
# labels 98% of shipments delayed, leaving nothing to learn: "always delayed"
# would score 98% accuracy.
#
# Canon pins the threshold but says nothing about delay magnitudes, so the
# magnitudes are what move. Each delay below is a sum of local terms plus
# PROPAGATION_COEF * upstream, clipped at zero -- a positively homogeneous
# function -- so scaling the finished chain by a constant is algebraically
# identical to scaling every local term, and correlations are scale-invariant, so
# the causal structure survives untouched. Choosing the factor as 6/24 lands the
# canonical threshold on precisely the quantile the 24h point occupied, which
# preserves the class balance exactly as well.
RAW_OPERATING_POINT_HOURS = 24.0
DELAY_SCALE = DELAY_THRESHOLD_HOURS / RAW_OPERATING_POINT_HOURS  # 0.25

PORTS = [
    "Shanghai", "Rotterdam", "Singapore", "Mumbai", "Los Angeles",
    "Sydney", "Dubai", "Hamburg", "Santos", "Busan",
]

# severity (0-3) -> (low, high) hours range for the LOCAL effect at that stage.
# Kept modest on purpose: if the local severity effect dwarfs the propagated
# upstream-delay term, stage-to-stage correlation washes out to near zero,
# which would defeat the point of this dataset (see PRD 1.7 / 2.A).
CONGESTION_RANGES = {0: (0, 1), 1: (1, 3), 2: (3, 6), 3: (6, 10)}
WEATHER_RANGES = {0: (0, 0.5), 1: (0.5, 2), 2: (2, 5), 3: (5, 9)}
CUSTOMS_RANGES = {0: (0, 1), 1: (1, 3), 2: (3, 6), 3: (6, 10)}

START_DATE = datetime(2026, 1, 1)


def _severity_hours(severity: int, ranges: dict, rng: np.random.Generator) -> float:
    lo, hi = ranges[severity]
    return float(rng.uniform(lo, hi))


def generate_shipments(n: int = N_SHIPMENTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        shipment_id = f"SHP-{i + 1:04d}"

        origin, destination = rng.choice(PORTS, size=2, replace=False)
        route = f"{origin} -> {destination}"

        # --- scheduled timestamps -------------------------------------------------
        dep_offset_days = int(rng.integers(0, 180))
        dep_offset_hours = int(rng.integers(0, 24))
        scheduled_departure = START_DATE + timedelta(days=dep_offset_days, hours=dep_offset_hours)

        scheduled_port_arrival = scheduled_departure + timedelta(
            days=5, hours=float(rng.uniform(-6, 6))
        )
        scheduled_customs_clearance = scheduled_port_arrival + timedelta(
            days=1, hours=float(rng.uniform(-4, 4))
        )
        scheduled_delivery = scheduled_customs_clearance + timedelta(
            days=4, hours=float(rng.uniform(-6, 6))
        )

        # --- external factors -------------------------------------------------
        port_congestion = int(rng.integers(0, 4))
        weather_severity = int(rng.integers(0, 4))
        customs_complexity = int(rng.integers(0, 4))
        document_readiness = float(rng.uniform(0, 1))

        # --- processing time context (not delays themselves) -------------------
        port_processing_hours = float(rng.uniform(12, 48))
        customs_processing_hours = float(rng.uniform(12, 48))
        inland_transit_hours = float(rng.uniform(12, 48))

        # --- causal delay propagation ------------------------------------------
        # departure: driven by document readiness (poor docs -> later departure)
        departure_delay_hours = max(
            0.0, (1 - document_readiness) * 5 + rng.normal(0, 1.0)
        )

        # port: local congestion effect + a fraction of departure delay carries over
        port_delay_hours = max(
            0.0,
            _severity_hours(port_congestion, CONGESTION_RANGES, rng)
            + PROPAGATION_COEF * departure_delay_hours
            + rng.normal(0, 1.0),
        )

        # customs: local complexity/document effect + a fraction of port delay carries over
        customs_delay_hours = max(
            0.0,
            _severity_hours(customs_complexity, CUSTOMS_RANGES, rng)
            + (1 - document_readiness) * 1.5
            + PROPAGATION_COEF * port_delay_hours
            + rng.normal(0, 1.0),
        )

        # inland: local weather effect + a fraction of customs delay carries over
        inland_delay_hours = max(
            0.0,
            _severity_hours(weather_severity, WEATHER_RANGES, rng)
            + PROPAGATION_COEF * customs_delay_hours
            + rng.normal(0, 1.0),
        )

        # Rescale the finished chain into the units the canonical 6h threshold
        # assumes. Done here, once, rather than inside every local term above:
        # the two are algebraically identical (see DELAY_SCALE) and this keeps the
        # propagation math readable as propagation math.
        departure_delay_hours *= DELAY_SCALE
        port_delay_hours *= DELAY_SCALE
        customs_delay_hours *= DELAY_SCALE
        inland_delay_hours *= DELAY_SCALE

        # Round BEFORE deriving anything from these values, not on the way out.
        # is_delayed is *defined* as total_delay_hours > 6 and the CSV stores the
        # total rounded to 2dp, so deriving the label from the unrounded value
        # lets a 6.004h shipment be written as 6.0 while labelled delayed --
        # anything that recomputes the label from the CSV then disagrees with the
        # stored one. Rounding first also makes total == sum(stage delays) and
        # actual == scheduled + delay hold exactly in the written file, so both
        # canonical identities are verifiable from the data itself.
        departure_delay_hours = round(departure_delay_hours, 2)
        port_delay_hours = round(port_delay_hours, 2)
        customs_delay_hours = round(customs_delay_hours, 2)
        inland_delay_hours = round(inland_delay_hours, 2)

        total_delay_hours = round(
            departure_delay_hours + port_delay_hours + customs_delay_hours + inland_delay_hours, 2
        )
        is_delayed = total_delay_hours > DELAY_THRESHOLD_HOURS

        # --- actual timestamps, consistent with cumulative delay at each stage ---
        actual_departure = scheduled_departure + timedelta(hours=departure_delay_hours)
        actual_port_arrival = scheduled_port_arrival + timedelta(
            hours=departure_delay_hours + port_delay_hours
        )
        actual_customs_clearance = scheduled_customs_clearance + timedelta(
            hours=departure_delay_hours + port_delay_hours + customs_delay_hours
        )
        actual_delivery = scheduled_delivery + timedelta(hours=total_delay_hours)

        rows.append(
            {
                "shipment_id": shipment_id,
                "origin": origin,
                "destination": destination,
                "route": route,
                "scheduled_departure": scheduled_departure,
                "scheduled_port_arrival": scheduled_port_arrival,
                "scheduled_customs_clearance": scheduled_customs_clearance,
                "scheduled_delivery": scheduled_delivery,
                "actual_departure": actual_departure,
                "actual_port_arrival": actual_port_arrival,
                "actual_customs_clearance": actual_customs_clearance,
                "actual_delivery": actual_delivery,
                "port_congestion": port_congestion,
                "weather_severity": weather_severity,
                "customs_complexity": customs_complexity,
                "document_readiness": round(document_readiness, 3),
                "port_processing_hours": round(port_processing_hours, 2),
                "customs_processing_hours": round(customs_processing_hours, 2),
                "inland_transit_hours": round(inland_transit_hours, 2),
                # Already rounded above, deliberately: the label and the actual
                # timestamps are derived from these exact values.
                "departure_delay_hours": departure_delay_hours,
                "port_delay_hours": port_delay_hours,
                "customs_delay_hours": customs_delay_hours,
                "inland_delay_hours": inland_delay_hours,
                "total_delay_hours": total_delay_hours,
                "is_delayed": is_delayed,
            }
        )

    df = pd.DataFrame(rows)

    # --- simulate missing intermediate data (assignment: "incomplete or missing
    # intermediate data" is an explicitly named challenge) ---------------------
    for ts_col, missing_col in [
        ("actual_port_arrival", "actual_port_arrival_missing"),
        ("actual_customs_clearance", "actual_customs_clearance_missing"),
        ("actual_delivery", "actual_delivery_missing"),
    ]:
        mask = rng.random(len(df)) < MISSING_PROB
        df[missing_col] = mask.astype(int)
        df.loc[mask, ts_col] = pd.NaT

    # A stage delay is defined as actual - scheduled, so a stage whose actual
    # timestamp never arrived has no computable delay. port_delay_hours is nulled
    # to match, because it is a *feature*: it has to reflect what the tracking
    # feed had actually delivered at the prediction cutoff, and an unrecorded
    # arrival means the operator genuinely cannot compute it at that moment.
    # Leaving it populated would make the missingness simulation cosmetic -- any
    # imputation built later would be validated against rows where nothing is
    # really missing, and would look perfect for the wrong reason.
    #
    # The customs and inland delays and both targets are deliberately left
    # populated. Those are assembled post-hoc during reconciliation, once the
    # full shipment record exists, so point-in-time correctness binds the inputs
    # and not the labels. Nulling them would instead wipe out ~27% of the
    # training labels for no modelling benefit.
    df.loc[df["actual_port_arrival_missing"] == 1, "port_delay_hours"] = np.nan

    return df


if __name__ == "__main__":
    df = generate_shipments()
    out_path = "data/raw/shipments.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}\n")

    print("--- Causal propagation sanity check ---")
    corr = df["port_delay_hours"].corr(df["departure_delay_hours"])
    print(f"corr(port_delay_hours, departure_delay_hours) = {corr:.3f}  (should be clearly positive)")
    corr2 = df["customs_delay_hours"].corr(df["port_delay_hours"])
    print(f"corr(customs_delay_hours, port_delay_hours)   = {corr2:.3f}  (should be clearly positive)")
    corr3 = df["inland_delay_hours"].corr(df["customs_delay_hours"])
    print(f"corr(inland_delay_hours, customs_delay_hours) = {corr3:.3f}  (should be clearly positive)\n")

    print("--- total_delay_hours describe() ---")
    print(df["total_delay_hours"].describe())

    print(f"\n--- is_delayed distribution ---")
    print(df["is_delayed"].value_counts(normalize=True))

    print("\n--- missingness rates ---")
    for col in ["actual_port_arrival_missing", "actual_customs_clearance_missing", "actual_delivery_missing"]:
        print(f"{col}: {df[col].mean():.1%}")

    print("\n--- null check ---")
    print("targets + post-cutoff delays, reconciled post-hoc, must be complete:")
    for col in [
        "departure_delay_hours", "customs_delay_hours", "inland_delay_hours",
        "total_delay_hours", "is_delayed",
    ]:
        n_null = int(df[col].isnull().sum())
        print(f"  {col:24s} nulls={n_null:3d}  {'OK' if n_null == 0 else 'UNEXPECTED'}")

    print("port_delay_hours is a cutoff-time feature, so it follows its timestamp:")
    n_null = int(df["port_delay_hours"].isnull().sum())
    n_expected = int(df["actual_port_arrival_missing"].sum())
    print(f"  {'port_delay_hours':24s} nulls={n_null:3d}  "
          f"{'OK' if n_null == n_expected else 'UNEXPECTED'} (expected {n_expected})")