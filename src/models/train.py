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
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = REPO_ROOT / "data" / "processed" / "features_v1.csv"

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


class RouteMeanRegressor(BaseEstimator, RegressorMixin):
    """Baseline: predict each route's mean target, learned from training data only.

    The fallback is the load-bearing part. With 86 routes across 300 shipments
    there are only ~2.6 training rows per route, so a held-out shipment can easily
    travel a lane never seen in training. Those rows fall back to the global
    training mean; without that, the baseline would emit NaN and no metric would
    be computable.

    Kept as an estimator rather than a function so it fits the same
    fit/predict contract as every other entry in MODEL_REGISTRY.
    """

    def __init__(self, route_column: str = ROUTE_COLUMN) -> None:
        self.route_column = route_column

    def fit(self, X: pd.DataFrame, y) -> "RouteMeanRegressor":
        target = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.global_mean_ = float(target.mean())
        self.route_means_ = target.groupby(X[self.route_column]).mean().to_dict()
        self.n_routes_seen_ = len(self.route_means_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mapped = X[self.route_column].map(self.route_means_)
        return mapped.fillna(self.global_mean_).to_numpy(dtype=float)


def build_linear_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """One-hot the route, median-impute the numerics, then plain LinearRegression.

    Imputation is inside the pipeline on purpose. `port_delay_hours` (and the two
    features derived from it) are null wherever the port-arrival timestamp was
    never recorded, and LinearRegression rejects NaN outright. Fitting the
    imputer as a pipeline step means the median is computed from the *training*
    fold only -- imputing before the split would leak val/test distribution
    information back into training.

    `add_indicator` is left off because `port_arrival_missing` is already a
    feature, so the model can distinguish imputed rows without a duplicate flag.

    `handle_unknown="ignore"` is required, not defensive: the temporal split puts
    routes in val/test that never appear in training, and the default would raise.
    Such rows get an all-zero route block and fall back to the intercept plus
    their numeric features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("route", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("numeric", SimpleImputer(strategy="median", add_indicator=False), numeric_features),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", preprocessor), ("model", LinearRegression())])


# Add later models here. Each factory receives the split so it can size itself to
# the current feature set; nothing else in this file needs to change.
ModelFactory = Callable[[TemporalSplit], BaseEstimator]

MODEL_REGISTRY: list[tuple[str, ModelFactory]] = [
    ("Route-mean baseline", lambda split: RouteMeanRegressor()),
    (
        "Linear regression",
        lambda split: build_linear_pipeline(split.numeric_features, split.categorical_features),
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


def run_comparison(split: TemporalSplit) -> pd.DataFrame:
    """Fit every registered model on train and score it on val and test."""
    records: list[dict[str, object]] = []
    for name, factory in MODEL_REGISTRY:
        model = factory(split)
        model.fit(split.X_train, split.y_train)
        for split_name, X, y in (
            ("val", split.X_val, split.y_val),
            ("test", split.X_test, split.y_test),
        ):
            records.append({"model": name, "split": split_name, **evaluate(y, model.predict(X))})
    return pd.DataFrame.from_records(records)


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

    results = run_comparison(split)
    print("\n" + format_comparison(results))
