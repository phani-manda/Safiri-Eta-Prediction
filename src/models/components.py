"""Reusable sklearn components for Safiri model pipelines.

These live outside ``train.py`` so persisted joblib artifacts reference an
importable module path instead of ``__main__`` when the training script is run
directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


class RouteMeanRegressor(BaseEstimator, RegressorMixin):
    """Baseline: predict each route's mean target, learned from training data only."""

    def __init__(self, route_column: str = "route") -> None:
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


class PropagationConsistentImputer(BaseEstimator, TransformerMixin):
    """Impute ``port_delay_hours``, then rebuild features derived from it."""

    REQUIRED = (
        "port_delay_hours",
        "departure_delay_hours",
        "cumulative_delay_so_far",
        "previous_stage_delay",
    )

    def fit(self, X: pd.DataFrame, y=None) -> "PropagationConsistentImputer":
        missing = [c for c in self.REQUIRED if c not in X.columns]
        if missing:
            raise KeyError(f"PropagationConsistentImputer requires columns: {missing}")
        self.port_delay_median_ = float(X["port_delay_hours"].median())

        # Pipeline.feature_names_in_ delegates to its first step. Recording this
        # here lets inference code validate incoming payload columns.
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        port = out["port_delay_hours"].fillna(self.port_delay_median_)
        out["port_delay_hours"] = port
        out["cumulative_delay_so_far"] = out["departure_delay_hours"] + port
        out["previous_stage_delay"] = port
        return out
