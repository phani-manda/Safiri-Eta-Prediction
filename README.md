# Safiri ETA Prediction

A freight-shipment prototype that, at the vessel-docking moment, predicts
**how late** delivery will be, **whether** it will be delayed, and **why** —
explained so an operations team can interrogate it.

## Problem

A shipment passes through five sequential stages — origin departure → port
arrival → customs clearance → inland transport → delivery. Time lost early
propagates downstream into the final delivery delay. The system must forecast
that delay at the port-arrival cutoff, estimate the risk of a "delayed"
shipment, and explain which factors and upstream delays produced the forecast.

## Approach summary

Synthetic 300-shipment data with a **built-in causal propagation chain** (0.55
of each stage's delay carries into the next) → a **leakage-safe feature frame**
(13 inputs, all knowable at the cutoff) → a **chronological 70/15/15 split** →
regressors and classifiers compared on identical pipelines → winners persisted
as `.joblib` → a **FastAPI service** returning prediction + risk + contributors
+ narrative → **12 automated tests** guarding the contract.

## Architecture

```text
data/raw/shipments.csv  (300 x 28, generated)
        |  src/features/engineering.py      (leakage-safe projection)
        v
data/processed/features_v1.csv  (300 x 16)
        |  src/models/train.py              (temporal split, comparison, persistence)
        v
models/eta_regressor.joblib  models/delay_classifier.joblib
        |  src/explainability/explainer.py  (importances, contributors, narrative)
        |  src/models/predictor.py          (payload -> feature row -> prediction)
        v
src/api/main.py                     (FastAPI: GET /health, POST /predict)
        |
        v
tests/  test_api.py  test_features.py  test_leakage.py
```

- **`generator.py`** — deterministic synthetic data with causal propagation, `SEED = 42`, 12% missing timestamps.
- **`engineering.py`** — projects raw rows to the 13 cutoff-legal features; runtime leakage guard.
- **`train.py`** — temporal split, model registries, comparison, persists both winners.
- **`components.py`** — import-safe `RouteMeanRegressor` + `PropagationConsistentImputer`.
- **`explainer.py`** — importances, per-prediction contributors, propagation sentence.
- **`predictor.py`** — inference wrapper; owns the payload → feature mapping.
- **`main.py`** — FastAPI app, pydantic-validated payloads.
- **`tests/`** — 5 API, 4 feature-identity, 3 leakage-boundary tests.

## Dataset

- 300 rows × 28 columns, 10 ports / 86 routes, `SEED = 42`; scheduled
  departures `2026-01-01`..`2026-06-29`; `total_delay_hours` mean 5.54, max
  11.58; 120/300 (40%) shipments exceed the 6 h threshold.
- Stage delays are causal: local severity effect + `0.55 × upstream delay` +
  noise, scaled by `DELAY_SCALE = 0.25` for a usable class balance.
- Propagation correlations: 0.314 (dep→port), 0.515 (port→customs),
  0.582 (customs→inland). Missingness 13.3% / 13.7% / 14.3% (port / customs /
  delivery), each with an indicator; `port_delay_hours` is nulled when the port
  timestamp is missing. All data files are gitignored build outputs.
## Feature engineering

`build_features()` (allow-list in `src/features/engineering.py`) produces the
13 model inputs below plus two targets and a split key.

| Feature | What it captures | Why it matters for propagation |
|---|---|---|
| `route` | origin → destination lane | per-lane baselines; model falls back to a global mean on unseen lanes |
| `month`, `day_of_week` | scheduled-departure calendar | booking-time seasonality and weekday effects |
| `departure_delay_hours` | how late the vessel left (observed) | the root of the cascade, inherited by every stage |
| `port_delay_hours` | how late the vessel docked (observed) | the second observed deviation, still propagating into customs/inland |
| `port_congestion`, `weather_severity`, `customs_complexity` | severity 0–3 per stage | local conditions add on top of the inherited delay |
| `document_readiness` | 0–1 paperwork score (higher = better) | drives departure lateness and customs clearance time |
| `port_arrival_missing` | 1 if the port timestamp is missing | flags the imputed row so the model knows port delay was not observed |
| `cumulative_delay_so_far` | `departure + port` delay at the cutoff | the quantity that propagates: hours lost upstream compress downstream |
| `schedule_slack` | planned window − planned work (canonical formula) | tells whether inherited delay still fits the delivery promise |
| `previous_stage_delay` | most recent observed stage delay (= port delay) | the concept the propagation story needs; becomes customs delay at a later cutoff |

## Prediction cutoff & leakage

**Prediction happens the moment `actual_port_arrival` is observed** — departed
and docked; customs and inland transport have not happened yet. Permanently
forbidden as inputs: `customs_delay_hours`, `inland_delay_hours`,
`actual_customs_clearance`, `actual_delivery`, their missingness flags, and
both targets. Enforced at build time (`_assert_no_leakage`), by the API
(derived features recomputed from planned fields only), and by
`tests/test_leakage.py` — set-membership assertions that no post-cutoff column
can ever appear in `build_features()` output.

## Models & results

Same temporal split for every model (210 / 45 / 45, chronological). Fresh run
of `src/models/train.py`:

```text
model                        val MAE    val RMSE      val R2    test MAE   test RMSE     test R2
------------------------------------------------------------------------------------------------
Route-mean baseline            1.795       2.316      -0.538       2.642       3.091      -0.514
Linear regression              0.732       0.960       0.736       0.733       0.920       0.866
Random forest                  0.704       0.875       0.780       0.806       0.954       0.856
Gradient boosting              0.644       0.784       0.824       0.648       0.746       0.912
```

**Selected regressor: gradient boosting** (lowest test MAE, 0.648 h ≈ 39 min) →
`models/eta_regressor.joblib`.

```text
model                       val Prec  val Recall  val F1   val ROC-AUC  test Prec  test Recall  test F1   test ROC-AUC
------------------------------------------------------------------------------------------------------------------------
Majority-class baseline       0.000      0.000    0.000      0.500        0.000       0.000     0.000       0.500
Logistic regression           0.870      0.833    0.851      0.938        0.870       0.952     0.909       0.984
Random forest                 0.800      0.833    0.816      0.915        0.875       1.000     0.933       0.980
```

**Selected classifier: random forest** (highest test positive-class recall,
1.000 — every delayed shipment flagged) → `models/delay_classifier.joblib`.
Rule: highest test Recall, tie-break F1 (missed delays are the priority).

## Explainability

`src/explainability/explainer.py` reports built-in and permutation importances
(route one-hot indicators aggregated back to one `route` feature). Permutation
ranks `port_delay_hours` (0.746 MAE loss) and `customs_complexity` (0.588) top
for the regressor — the causal structure the generator built is the structure
the model uses. Real output (first test-set row, a Busan → Hamburg shipment):

```text
Example test-set prediction
predicted_delay_hours: 8.048
top contributors:
  schedule_slack: 3.497 h
  cumulative_delay_so_far: 3.279 h
  customs_complexity: -1.273 h
propagation:
  A 1.33h departure delay contributed to a 2.22h port delay, bringing the
  shipment to 3.55h cumulative delay and leaving 78.31h of schedule slack
  for the remaining stages.
```

Stated in the code: `impact_hours` is a transparent rescaled heuristic (global
importance × deviation from training mean), not a rigorous Shapley
decomposition — exact attribution is ill-defined across the near-duplicates.

## Running locally

From a clean clone (Python 3.11+, Windows PowerShell shown):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\reproduce.py --tests
```

`reproduce.py` runs the canonical sequence exactly: `generator.py` →
`engineering.py` → `train.py` → `explainer.py` → full `pytest`. (The same four
scripts reproduce manually; all data and model files are gitignored rebuilds.)

Serve the API (health check: `Invoke-RestMethod http://127.0.0.1:8000/health`
→ `{"status":"ok"}`):

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

## Example request/response

POST `/predict` with `examples/predict_payload.json` (Santos → Hamburg):

```json
{
  "predicted_delay_hours": 10.146,
  "predicted_eta": "2026-05-04T16:56:29.406393",
  "delay_probability": 1.0,
  "risk_level": "HIGH",
  "contributors": [
    {"factor": "schedule_slack", "impact_hours": 4.912},
    {"factor": "cumulative_delay_so_far", "impact_hours": 3.931},
    {"factor": "customs_complexity", "impact_hours": 1.303}
  ],
  "propagation": "A 1.11h departure delay contributed to a 2.76h port delay, bringing the shipment to 3.87h cumulative delay and leaving 94.41h of schedule slack for the remaining stages."
}
```

`predicted_eta = scheduled_delivery + predicted_delay_hours` (ISO 8601);
`risk_level` maps probability to LOW (<0.3) / MEDIUM (0.3–0.6) / HIGH (>0.6).

## Limitations

- Synthetic, single-seed, 300 rows — held-out metrics (45 rows) carry wide
  uncertainty.
- Fixed prediction cutoff at port arrival; no multi-stage re-prediction.
- `impact_hours` is an approximation, not exact Shapley attribution.
- API assumes the port-arrival timestamp was observed (missingness handled at
  train time only). No live data-source integration.

## Future improvements

Model cards + drift checks; multi-stage/streaming prediction; batch endpoint;
exact-attribution explainability; a deployment path (Docker); live data.