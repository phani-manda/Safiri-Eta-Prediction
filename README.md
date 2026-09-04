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

*Remaining placeholders are resolved and expanded in a later stage of the build.*

- Prediction cutoff definition: TBD
- **Delay threshold definition:** `is_delayed` is `total_delay_hours > 6`, exactly
  as the canonical schema specifies. The generator's stage-delay magnitudes are
  scaled by `DELAY_SCALE = 0.25` so that this threshold splits the data ~40/60
  instead of labelling 98% of shipments delayed, which would leave the classifier
  nothing to learn. Canon pins the threshold but not the delay magnitudes, so the
  magnitudes are what moved. The factor is applied to the finished delay chain,
  which is positively homogeneous, so it rescales units without altering a single
  correlation — the causal structure is preserved exactly.
- Congestion/weather/customs complexity scales: TBD
- **Missing data handling approach:** ~12% of intermediate actual timestamps are
  simulated as missing, each with its own 0/1 indicator column.
  `port_delay_hours` is nulled wherever `actual_port_arrival` is missing, because
  it is a model *input* and must reflect what the tracking feed had actually
  delivered at the prediction cutoff — a delay derived from an unobserved
  timestamp is not knowable then. The post-cutoff stage delays and both targets
  stay populated: labels are assembled post-hoc during reconciliation, when the
  full shipment record exists, so point-in-time correctness binds the features and
  not the targets. Imputation strategy for the resulting null feature rows: TBD.
