"""Train and compare delay-regression models under a temporal split.

Two design choices dominate this module.

**The split is temporal, not random.** A random split lets a model train on
shipments that departed *after* the ones it is scored on. Nothing in the metrics
reveals that, but it cannot happen in production, and it flatters every model --
especially on a dataset with per-route structure, where a random split scatters
each route across train and test and quietly hands the model route-level
information it would not have on a genuinely new lane. Sorting by
`scheduled_departure` and cutting forward in time reproduces the deployment
situation: fit on the past, predict the future.

**Models are registered, not hardcoded into the reporting path.** Every model is
an estimator that consumes the same raw feature frame and owns its own
preprocessing, so `MODEL_REGISTRY` is the only thing a later prompt needs to touch
to add a model to the comparison. The split, the metrics and the table are shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.components import PropagationConsistentImputer, RouteMeanRegressor

FEATURES_CSV = REPO_ROOT / "data" / "processed" / "features_v1.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "eta_regressor.joblib"

# The selected model is the one with the lowest test MAE.
SELECTION_SPLIT = "test"
SELECTION_METRIC = "MAE"

TARGET = "total_delay_hours"
SORT_KEY = "scheduled_departure"
ROUTE_COLUMN = "route"

# Present in features_v1.csv but never model inputs: the chronological key exists
# only to order rows for the split, and the two targets are what we predict.
NON_FEATURE_COLUMNS = (SORT_KEY, "total_delay_hours", "is_delayed")

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15  # the remainder becomes the test split


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalSplit:
    """Chronologically ordered train/val/test partition, sort key already dropped."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_period: tuple[pd.Timestamp, pd.Timestamp]
    val_period: tuple[pd.Timestamp, pd.Timestamp]
    test_period: tuple[pd.Timestamp, pd.Timestamp]

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)

    @property
    def categorical_features(self) -> list[str]:
        return [ROUTE_COLUMN]

    @property
    def numeric_features(self) -> list[str]:
        """Everything that is not the route. Derived rather than hardcoded so a
        feature added in engineering.py flows through without editing this file."""
        return [c for c in self.X_train.columns if c != ROUTE_COLUMN]


def load_features(path: Path = FEATURES_CSV) -> pd.DataFrame:
    """Load the engineered feature frame, parsing the chronological key."""
    df = pd.read_csv(path, parse_dates=[SORT_KEY])
    if TARGET not in df.columns:
        raise KeyError(f"{path} has no '{TARGET}' column to regress on")
    return df


def make_temporal_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> TemporalSplit:
    """Cut the frame into earliest-70% / next-15% / latest-15% by departure date.

    A stable (mergesort) sort is used so that shipments sharing a departure
    timestamp keep their file order, which keeps the split byte-reproducible
    across runs and platforms.

    The sort key is dropped from the feature matrices here rather than being left
    for each model to remember: a raw monotonic timestamp is precisely the kind of
    column a model can exploit to fake a time trend, and dropping it once at the
    boundary means no downstream model can accidentally receive it.
    """
    ordered = df.sort_values(SORT_KEY, kind="mergesort").reset_index(drop=True)

    n = len(ordered)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    if not 0 < train_end < val_end < n:
        raise ValueError(
            f"split fractions produce empty partitions for n={n}: "
            f"train_end={train_end}, val_end={val_end}"
        )

    parts = {
        "train": ordered.iloc[:train_end],
        "val": ordered.iloc[train_end:val_end],
        "test": ordered.iloc[val_end:],
    }

    def features_of(part: pd.DataFrame) -> pd.DataFrame:
        return part.drop(columns=list(NON_FEATURE_COLUMNS))

    def period_of(part: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        return part[SORT_KEY].min(), part[SORT_KEY].max()

    return TemporalSplit(
        X_train=features_of(parts["train"]),
        y_train=parts["train"][TARGET],
        X_val=features_of(parts["val"]),
        y_val=parts["val"][TARGET],
        X_test=features_of(parts["test"]),
        y_test=parts["test"][TARGET],
        train_period=period_of(parts["train"]),
        val_period=period_of(parts["val"]),
        test_period=period_of(parts["test"]),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_pipeline(estimator: BaseEstimator, split: TemporalSplit) -> Pipeline:
    """Wrap `estimator` in the shared preprocessing: one-hot route, median-impute numerics.

    Every model in the comparison goes through this same preprocessor on purpose.
    If each estimator prepared its own inputs, a gap in the results table could be
    a difference in preprocessing rather than a difference in the model, and the
    comparison would not answer the question it appears to answer.

    Imputation lives inside the pipeline deliberately. `port_delay_hours` (and the
    two features derived from it) are null wherever the port-arrival timestamp was
    never recorded, and LinearRegression rejects NaN outright. As a pipeline step,
    the median is computed from the *training* fold only -- imputing before the
    split would leak val/test distribution information back into training.

    Two imputation stages, doing different jobs: `PropagationConsistentImputer`
    fills the port delay and rebuilds its derived features so the canonical
    identities survive, and the `SimpleImputer` behind it is a safety net that
    catches any other numeric NaN a future feature might introduce. On today's
    feature set the second stage is a no-op.

    `add_indicator` is left off because `port_arrival_missing` is already a feature,
    so a model can distinguish imputed rows without a duplicate column.

    `handle_unknown="ignore"` is required, not defensive: the temporal split puts
    routes in val and test that never appear in training, and the default would
    raise. Those rows get an all-zero route block and lean on the numeric features.

    Caveat worth knowing when reading the table: the tree models receive the same
    81-column one-hot route block. Trees generally do better with native
    categorical handling than with a wide sparse indicator matrix, so this is not
    the strongest possible tree configuration -- but holding the representation
    fixed is what keeps the comparison honest.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "route",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                split.categorical_features,
            ),
            ("numeric", SimpleImputer(strategy="median", add_indicator=False), split.numeric_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("consistent_impute", PropagationConsistentImputer()),
            ("prep", preprocessor),
            ("model", estimator),
        ]
    )


# Add later models here. Each factory receives the split so it can size itself to
# the current feature set; nothing else in this file needs to change.
ModelFactory = Callable[[TemporalSplit], BaseEstimator]

MODEL_REGISTRY: list[tuple[str, ModelFactory]] = [
    ("Route-mean baseline", lambda split: RouteMeanRegressor()),
    ("Linear regression", lambda split: build_pipeline(LinearRegression(), split)),
    (
        "Random forest",
        lambda split: build_pipeline(
            RandomForestRegressor(n_estimators=200, random_state=42), split
        ),
    ),
    (
        "Gradient boosting",
        lambda split: build_pipeline(GradientBoostingRegressor(random_state=42), split),
    ),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

METRIC_NAMES = ("MAE", "RMSE", "R2")


def evaluate(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE and R2 for one set of predictions.

    MAE and RMSE are both reported because they disagree in a useful way: RMSE
    punishes the occasional badly-missed shipment far harder than MAE, so a model
    that wins on MAE but loses on RMSE is one that is usually close and
    occasionally very wrong.
    """
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_comparison(split: TemporalSplit) -> tuple[pd.DataFrame, dict[str, BaseEstimator]]:
    """Fit every registered model on train and score it on val and test.

    Returns the fitted estimators alongside the metrics so the selected model can
    be persisted without refitting. Refitting would waste work and, more
    importantly, risks saving an object that is not the one the reported numbers
    were measured on.
    """
    records: list[dict[str, object]] = []
    fitted: dict[str, BaseEstimator] = {}

    for name, factory in MODEL_REGISTRY:
        model = factory(split)
        model.fit(split.X_train, split.y_train)
        fitted[name] = model
        for split_name, X, y in (
            ("val", split.X_val, split.y_val),
            ("test", split.X_test, split.y_test),
        ):
            records.append({"model": name, "split": split_name, **evaluate(y, model.predict(X))})

    return pd.DataFrame.from_records(records), fitted


def select_best_model(results: pd.DataFrame) -> tuple[str, pd.Series]:
    """Return the lowest-test-MAE model's name, plus every model's test MAE.

    MAE is the selector rather than RMSE or R2 because it is denominated in hours:
    "best" then means the model whose ETA is on average closest in the unit an
    operations team actually acts on. R2 would be a poor choice here since it is
    variance-relative and the val and test splits have visibly different target
    spreads, so it can rank models by which split they landed on as much as by skill.
    """
    scores = (
        results[results["split"] == SELECTION_SPLIT]
        .set_index("model")[SELECTION_METRIC]
        .reindex([name for name, _ in MODEL_REGISTRY])
    )
    if scores.isna().any():
        missing = sorted(scores[scores.isna()].index)
        raise ValueError(f"no {SELECTION_SPLIT} {SELECTION_METRIC} recorded for: {missing}")
    return str(scores.idxmin()), scores


def format_comparison(results: pd.DataFrame) -> str:
    """Render the comparison as a fixed-width table, in registry order."""
    header = f"{'model':<24}" + "".join(
        f"{f'{s} {m}':>12}" for s in ("val", "test") for m in METRIC_NAMES
    )
    lines = [header, "-" * len(header)]

    for name, _ in MODEL_REGISTRY:
        cells = []
        for split_name in ("val", "test"):
            row = results[(results["model"] == name) & (results["split"] == split_name)]
            for metric in METRIC_NAMES:
                cells.append(f"{row[metric].iloc[0]:>12.3f}")
        lines.append(f"{name:<24}" + "".join(cells))

    return "\n".join(lines)


if __name__ == "__main__":
    features = load_features()
    split = make_temporal_split(features)

    print(f"loaded {FEATURES_CSV.relative_to(REPO_ROOT).as_posix()}  "
          f"{len(features)} rows x {len(features.columns)} cols")
    print(f"\ntemporal split on '{SORT_KEY}' "
          f"({TRAIN_FRACTION:.0%}/{VAL_FRACTION:.0%}/"
          f"{1 - TRAIN_FRACTION - VAL_FRACTION:.0%}):")
    for label, X, period in (
        ("train", split.X_train, split.train_period),
        ("val", split.X_val, split.val_period),
        ("test", split.X_test, split.test_period),
    ):
        print(f"  {label:<5} {len(X):>4} rows   "
              f"{period[0].date()} .. {period[1].date()}")

    print(f"\n{len(split.feature_names)} model input features "
          f"({len(split.categorical_features)} categorical, "
          f"{len(split.numeric_features)} numeric); "
          f"'{SORT_KEY}' dropped after sorting")

    # Unseen-route counts explain the baseline's fallback rate, so report them
    # rather than leaving the reader to wonder why it underperforms.
    train_routes = set(split.X_train[ROUTE_COLUMN])
    for label, X in (("val", split.X_val), ("test", split.X_test)):
        unseen = (~X[ROUTE_COLUMN].isin(train_routes)).sum()
        print(f"  routes in {label} unseen during training: {unseen}/{len(X)}")

    results, fitted = run_comparison(split)
    print("\n" + format_comparison(results))

    best_name, test_mae = select_best_model(results)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], MODEL_PATH)

    ranked = test_mae.sort_values()
    best_score = float(ranked.iloc[0])

    print(f"\nselected: {best_name}")
    print(f"  criterion: lowest {SELECTION_SPLIT} {SELECTION_METRIC}, in hours\n")
    for name, value in ranked.items():
        if name == best_name:
            note = "<-- selected"
        else:
            note = f"+{value - best_score:.3f} h worse ({value / best_score:.2f}x)"
        print(f"    {name:<24}{value:>8.3f}   {note}")

    print(f"\nsaved -> {MODEL_PATH.relative_to(REPO_ROOT).as_posix()}")
