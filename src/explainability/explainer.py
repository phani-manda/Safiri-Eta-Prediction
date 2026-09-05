"""Model explainability helpers for Safiri ETA and delay-risk models."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.train import (  # noqa: E402
    CLASSIFIER_MODEL_PATH,
    MODEL_PATH,
    load_features,
    make_temporal_split,
)

_TRAINING_MEANS: pd.Series | None = None


def _final_estimator(model):
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def _normalize_transformed_name(name: str) -> str:
    if name.startswith("numeric__"):
        return name.removeprefix("numeric__")
    if name.startswith("route__"):
        return "route"
    return name


def _importance_frame(importances, names: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": names, "importance": importances})
        .groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def get_feature_importances(model, feature_names) -> pd.DataFrame:
    """Return built-in model importances, sorted descending.

    Pipelines expose importances on their final estimator after preprocessing.
    If one-hot encoding expands the feature space, route indicators are grouped
    back into a single ``route`` feature so the table matches the raw feature
    contract used at inference time.
    """
    estimator = _final_estimator(model)
    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError("model does not expose feature_importances_")

    importances = estimator.feature_importances_
    names = list(feature_names)
    if len(importances) != len(names):
        if not hasattr(model, "named_steps") or "prep" not in model.named_steps:
            raise ValueError(
                f"{len(importances)} importances do not match {len(names)} feature names"
            )
        names = [
            _normalize_transformed_name(name)
            for name in model.named_steps["prep"].get_feature_names_out()
        ]

    return _importance_frame(importances, names)


def get_permutation_importances(
    model,
    X_val,
    y_val,
    feature_names,
    n_repeats=10,
    random_state=42,
) -> pd.DataFrame:
    """Return permutation importances on the original feature matrix."""
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _training_means() -> pd.Series:
    global _TRAINING_MEANS
    if _TRAINING_MEANS is None:
        split = make_temporal_split(load_features())
        _TRAINING_MEANS = split.X_train.select_dtypes(include="number").mean()
    return _TRAINING_MEANS


def explain_prediction(
    model,
    feature_row: pd.Series,
    feature_importances: pd.DataFrame,
    top_n=3,
) -> list[dict]:
    """Return approximate top contributors for a single prediction.

    This is a lightweight heuristic, not a SHAP-equivalent decomposition. It
    ranks numeric features by ``importance * deviation_from_training_mean`` and
    scales the selected values so their absolute impact roughly tracks the
    predicted delay magnitude.
    """
    means = _training_means()
    importance_by_feature = feature_importances.set_index("feature")["importance"]
    rows: list[dict[str, float | str]] = []

    for feature, importance in importance_by_feature.items():
        if feature not in feature_row.index or feature not in means.index:
            continue
        value = pd.to_numeric(pd.Series([feature_row[feature]]), errors="coerce").iloc[0]
        if pd.isna(value):
            value = means[feature]
        raw_contribution = float(importance) * float(value - means[feature])
        rows.append({"factor": feature, "raw_contribution": raw_contribution})

    top = sorted(rows, key=lambda item: abs(float(item["raw_contribution"])), reverse=True)[
        :top_n
    ]
    if not top:
        return []

    feature_frame = feature_row.to_frame().T
    if hasattr(model, "feature_names_in_"):
        feature_frame = feature_frame[list(model.feature_names_in_)]
    predicted_delay = float(model.predict(feature_frame)[0])
    raw_total = sum(abs(float(item["raw_contribution"])) for item in top)
    scale = predicted_delay / raw_total if raw_total else 0.0

    return [
        {
            "factor": str(item["factor"]),
            "impact_hours": float(float(item["raw_contribution"]) * scale),
        }
        for item in top
    ]


def generate_propagation_sentence(feature_row: pd.Series) -> str:
    """Describe how upstream delays have propagated for one shipment."""
    departure_delay = float(feature_row["departure_delay_hours"])
    port_delay = pd.to_numeric(pd.Series([feature_row["port_delay_hours"]]), errors="coerce").iloc[0]
    cumulative_delay = pd.to_numeric(
        pd.Series([feature_row["cumulative_delay_so_far"]]), errors="coerce"
    ).iloc[0]
    schedule_slack = float(feature_row["schedule_slack"])

    port_text = "an unknown port delay" if pd.isna(port_delay) else f"a {port_delay:.2f}h port delay"
    cumulative_text = (
        "an unknown cumulative delay"
        if pd.isna(cumulative_delay)
        else f"{cumulative_delay:.2f}h cumulative delay"
    )
    if schedule_slack < 0:
        slack_text = "no remaining slack for the remaining stages"
    else:
        slack_text = f"{schedule_slack:.2f}h of schedule slack for the remaining stages"

    return (
        f"A {departure_delay:.2f}h departure delay contributed to {port_text}, "
        f"bringing the shipment to {cumulative_text} and leaving {slack_text}."
    )


def _print_side_by_side(title: str, left: pd.DataFrame, right: pd.DataFrame) -> None:
    print(f"\n{title}")
    combined = pd.concat(
        [
            left.head(8).reset_index(drop=True).add_prefix("built_in_"),
            right.head(8).reset_index(drop=True).add_prefix("permutation_"),
        ],
        axis=1,
    )
    print(combined.round(4).to_string(index=False))


def _flag_if_unexpected(name: str, table: pd.DataFrame) -> None:
    expected = {"port_congestion", "cumulative_delay_so_far", "schedule_slack"}
    top_features = set(table.head(5)["feature"])
    if not expected.intersection(top_features):
        print(
            f"WARNING: {name} top importances do not include expected operational "
            "features; check the generator or feature pipeline."
        )


if __name__ == "__main__":
    features = load_features()
    split = make_temporal_split(features)
    feature_names = split.feature_names

    eta_model = joblib.load(MODEL_PATH)
    delay_model = joblib.load(CLASSIFIER_MODEL_PATH)

    eta_builtin = get_feature_importances(eta_model, feature_names)
    eta_permutation = get_permutation_importances(
        eta_model, split.X_val, split.y_val, feature_names
    )
    delay_builtin = get_feature_importances(delay_model, feature_names)
    delay_permutation = get_permutation_importances(
        delay_model, split.X_val, split.y_class_val, feature_names
    )

    _print_side_by_side("ETA regressor importances", eta_builtin, eta_permutation)
    _print_side_by_side("Delay classifier importances", delay_builtin, delay_permutation)

    _flag_if_unexpected("ETA regressor", eta_builtin)
    _flag_if_unexpected("Delay classifier", delay_builtin)

    example = split.X_test.iloc[0]
    predicted_delay = float(eta_model.predict(example.to_frame().T)[0])
    contributors = explain_prediction(eta_model, example, eta_builtin, top_n=3)

    print("\nExample test-set prediction")
    print(f"predicted_delay_hours: {predicted_delay:.3f}")
    print("top contributors:")
    for item in contributors:
        print(f"  {item['factor']}: {item['impact_hours']:.3f} h")
    print("propagation:")
    print(f"  {generate_propagation_sentence(example)}")
