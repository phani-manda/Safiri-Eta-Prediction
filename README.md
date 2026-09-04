# Safiri ETA Prediction

## Problem

Freight shipments move through five sequential stages — origin departure, port
arrival, customs clearance, inland transport, and final delivery — and a delay
incurred at an early stage propagates downstream, compressing the time
available for every stage that follows. This prototype predicts a shipment's
final ETA and its delay risk partway through that journey, while there is still
time to act on the forecast. Beyond the prediction itself, it explains *why*:
which operational factors drive the estimate, and how much of it is attributable
to delay already accumulated at upstream stages. The goal is a forecast an
operations team can interrogate and trust, not just a number.

## Assumptions

*All four resolved as of Phase 1. See `report/phase1_report.md` for full derivations.*

- **Prediction cutoff definition:** prediction happens the moment
  `actual_port_arrival` is observed — the shipment has departed and docked, but has
  not yet cleared customs or begun inland transport. Legal inputs are everything
  known by then: the four operational drivers, `route`, `month`, `day_of_week`, the
  planned processing durations, and the two *observed* stage delays
  (`departure_delay_hours`, `port_delay_hours`). Permanently forbidden as inputs:
  `customs_delay_hours`, `inland_delay_hours`, `actual_customs_clearance`,
  `actual_delivery`, their missingness flags, and both targets. Enforced at runtime
  by `_assert_no_leakage` in `src/features/engineering.py`, which checks in both
  directions — forbidden columns present, and unrecognised columns present.
- **Delay threshold definition:** `is_delayed` is `total_delay_hours > 6`, exactly
  as the canonical schema specifies. The generator's stage-delay magnitudes are
  scaled by `DELAY_SCALE = 0.25` so that this threshold splits the data ~40/60
  instead of labelling 98% of shipments delayed, which would leave the classifier
  nothing to learn. Canon pins the threshold but not the delay magnitudes, so the
  magnitudes are what moved. The factor is applied to the finished delay chain,
  which is positively homogeneous, so it rescales units without altering a single
  correlation — the causal structure is preserved exactly.
- **Congestion/weather/customs complexity scales:** `port_congestion`,
  `weather_severity` and `customs_complexity` are integer severity levels 0–3
  (0 = none, 3 = severe), drawn uniformly and independently. Each maps to a range
  of *local* delay hours at its own stage, pre-scaling: congestion and customs
  `{0:(0,1), 1:(1,3), 2:(3,6), 3:(6,10)}`, weather
  `{0:(0,0.5), 1:(0.5,2), 2:(2,5), 3:(5,9)}`, with the actual contribution drawn
  uniformly inside the band. `document_readiness` is different in kind: a
  continuous 0–1 score where **higher is better**, so its correlation with delay is
  legitimately *negative*. Ranges are kept modest relative to the propagation term
  on purpose — if a local severity effect dwarfs the inherited upstream delay,
  stage-to-stage correlation washes out and the dataset loses the propagation it
  exists to demonstrate.
- **Missing data handling approach:** ~12% of intermediate actual timestamps are
  simulated as missing, each with its own 0/1 indicator column.
  `port_delay_hours` is nulled wherever `actual_port_arrival` is missing, because
  it is a model *input* and must reflect what the tracking feed had actually
  delivered at the prediction cutoff — a delay derived from an unobserved
  timestamp is not knowable then. The post-cutoff stage delays and both targets
  stay populated: labels are assembled post-hoc during reconciliation, when the
  full shipment record exists, so point-in-time correctness binds the features and
  not the targets.

  Imputation is handled by `PropagationConsistentImputer`
  (`src/models/train.py`), fitted inside the model pipeline so the median comes
  from the training fold only. It fills the single underlying quantity
  (`port_delay_hours`) with the training median and then *recomputes* the two
  features defined in terms of it. A plain per-column median imputer would fill
  all three independently and leave `cumulative_delay_so_far ≠
  departure_delay_hours + port_delay_hours` on the affected rows — a feature
  contradicting its own definition, which is indefensible in a project whose
  deliverable is upstream-stage attribution.
