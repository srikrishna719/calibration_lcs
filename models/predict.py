"""Prediction helpers for trained calibration models."""

from __future__ import annotations

import pandas as pd
from sklearn.base import RegressorMixin


def predict_with_model(
    model: RegressorMixin,
    dataframe: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """Generate timestamped predictions for a fitted model."""
    features = dataframe.drop(columns=[target_column, timestamp_column], errors="ignore")
    predictions = model.predict(features)
    result = pd.DataFrame({"prediction": predictions})
    if timestamp_column in dataframe.columns:
        result.insert(0, timestamp_column, dataframe[timestamp_column].values)
    return result
