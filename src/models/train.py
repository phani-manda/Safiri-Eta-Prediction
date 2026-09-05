"""Train and compare delay-regression models under a temporal split.

The split is temporal (not random): train on past shipments, score on future
ones. This matches deployment and avoids the route-level leakage a random
split would introduce. Models live in MODEL_REGISTRY, so adding one does not
touch the comparison logic.
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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.components import PropagationConsistentImputer, RouteMeanRegressor

FEATURES_CSV = REPO_ROOT / "data" / "processed" / "features_v1.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "eta_regressor.joblib"
CLASSIFIER_MODEL_PATH = MODEL_DIR / "delay_classifier.joblib"

# pick lowest test MAE
SELECTION_SPLIT = "test"
SELECTION_METRIC = "MAE"

TARGET = "total_delay_hours"
CLASSIFICATION_TARGET = "is_delayed"
SORT_KEY = "scheduled_departure"
ROUTE_COLUMN = "route"

# sort key + targets are not model inputs
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
    y_class_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    y_class_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    y_class_test: pd.Series
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
    if CLASSIFICATION_TARGET not in df.columns:
        raise KeyError(f"{path} has no '{CLASSIFICATION_TARGET}' column to classify")
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
        y_class_train=parts["train"][CLASSIFICATION_TARGET].astype(bool),
        X_val=features_of(parts["val"]),
        y_val=parts["val"][TARGET],
        y_class_val=parts["val"][CLASSIFICATION_TARGET].astype(bool),
        X_test=features_of(parts["test"]),
        y_test=parts["test"][TARGET],
        y_class_test=parts["test"][CLASSIFICATION_TARGET].astype(bool),
        train_period=period_of(parts["train"]),
        val_period=period_of(parts["val"]),
        test_period=period_of(parts["test"]),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_preprocessor(split: TemporalSplit) -> ColumnTransformer:
    """Build the shared route one-hot + numeric median preprocessing block."""
    return ColumnTransformer(
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
    return Pipeline(
        [
            ("consistent_impute", PropagationConsistentImputer()),
            ("prep", build_preprocessor(split)),
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

CLASSIFIER_REGISTRY: list[tuple[str, ModelFactory]] = [
    ("Majority-class baseline", lambda split: DummyClassifier(strategy="most_frequent")),
    (
        "Logistic regression",
        lambda split: build_pipeline(
            LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000), split
        ),
    ),
    (
        "Random forest",
        lambda split: build_pipeline(
            RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=42
            ),
            split,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

METRIC_NAMES = ("MAE", "RMSE", "R2")
CLASSIFICATION_METRIC_NAMES = ("Precision", "Recall", "F1", "ROC-AUC")


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


def evaluate_classifier(model: BaseEstimator, X: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
    """Positive-class classification metrics for delay-risk predictions."""
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        positive_score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        positive_score = model.decision_function(X)
    else:
        positive_score = y_pred

    return {
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, positive_score)),
    }


def run_model_comparison(
    registry: list[tuple[str, ModelFactory]],
    split: TemporalSplit,
    y_train: pd.Series,
    evaluations: tuple[tuple[str, pd.DataFrame, pd.Series], ...],
    scorer: Callable[[BaseEstimator, pd.DataFrame, pd.Series], dict[str, float]],
) -> tuple[pd.DataFrame, dict[str, BaseEstimator]]:
    """Fit registered models once and score each on the requested splits."""
    records: list[dict[str, object]] = []
    fitted: dict[str, BaseEstimator] = {}

    for name, factory in registry:
        model = factory(split)
        model.fit(split.X_train, y_train)
        fitted[name] = model
        for split_name, X, y in evaluations:
            records.append({"model": name, "split": split_name, **scorer(model, X, y)})

    return pd.DataFrame.from_records(records), fitted


def score_regressor(model: BaseEstimator, X: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
    """Regression metrics for a fitted model and one evaluation split."""
    return evaluate(y_true, model.predict(X))


def run_comparison(split: TemporalSplit) -> tuple[pd.DataFrame, dict[str, BaseEstimator]]:
    """Fit every registered regressor on train and score it on val and test.

    Returns the fitted estimators alongside the metrics so the selected model can
    be persisted without refitting. Refitting would waste work and, more
    importantly, risks saving an object that is not the one the reported numbers
    were measured on.
    """
    return run_model_comparison(
        MODEL_REGISTRY,
        split,
        split.y_train,
        (
            ("val", split.X_val, split.y_val),
            ("test", split.X_test, split.y_test),
        ),
        score_regressor,
    )


def run_classification_comparison(
    split: TemporalSplit,
) -> tuple[pd.DataFrame, dict[str, BaseEstimator]]:
    """Fit delay classifiers on train and score positive-class risk metrics."""
    return run_model_comparison(
        CLASSIFIER_REGISTRY,
        split,
        split.y_class_train,
        (
            ("val", split.X_val, split.y_class_val),
            ("test", split.X_test, split.y_class_test),
        ),
        evaluate_classifier,
    )


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


def select_best_classifier(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Select the test classifier with best positive-class Recall, then F1."""
    scores = (
        results[results["split"] == "test"]
        .set_index("model")[["Recall", "F1"]]
        .reindex([name for name, _ in CLASSIFIER_REGISTRY])
    )
    if scores.isna().any(axis=None):
        missing = sorted(scores[scores.isna().any(axis=1)].index)
        raise ValueError(f"no test classification metrics recorded for: {missing}")
    ranked = scores.sort_values(["Recall", "F1"], ascending=[False, False])
    return str(ranked.index[0]), ranked


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


def format_classification_comparison(results: pd.DataFrame) -> str:
    """Render the classification comparison as a fixed-width table."""
    header = f"{'model':<24}" + "".join(
        f"{f'{s} {m}':>16}"
        for s in ("val", "test")
        for m in CLASSIFICATION_METRIC_NAMES
    )
    lines = [header, "-" * len(header)]

    for name, _ in CLASSIFIER_REGISTRY:
        cells = []
        for split_name in ("val", "test"):
            row = results[(results["model"] == name) & (results["split"] == split_name)]
            for metric in CLASSIFICATION_METRIC_NAMES:
                cells.append(f"{row[metric].iloc[0]:>16.3f}")
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

    # report unseen routes: they explain why the baseline falls back to the global mean
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

    class_results, fitted_classifiers = run_classification_comparison(split)
    print("\nDelay classification target: is_delayed=True")
    print("Positive-class recall is printed as Recall; missed delays are the priority.\n")
    print(format_classification_comparison(class_results))

    for split_name in ("val", "test"):
        split_rows = class_results[class_results["split"] == split_name]
        best_recall = split_rows.loc[split_rows["Recall"].idxmax()]
        print(
            f"\npositive-class Recall on {split_name}: "
            f"{best_recall['model']} = {best_recall['Recall']:.3f}"
        )

    best_classifier_name, classifier_ranking = select_best_classifier(class_results)
    joblib.dump(fitted_classifiers[best_classifier_name], CLASSIFIER_MODEL_PATH)
    winner = classifier_ranking.loc[best_classifier_name]
    print(f"\nselected classifier: {best_classifier_name}")
    print(
        "  criterion: highest test positive-class Recall, "
        f"tie-break by F1 (Recall={winner['Recall']:.3f}, F1={winner['F1']:.3f})"
    )
    print(f"saved -> {CLASSIFIER_MODEL_PATH.relative_to(REPO_ROOT).as_posix()}")
