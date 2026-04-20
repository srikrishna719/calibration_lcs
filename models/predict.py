"""Prediction helpers for trained calibration models."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.base import RegressorMixin


def predict_with_model(
    model: RegressorMixin,
    dataframe: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """Generate predictions for a dataset using a trained model.

    Parameters
    ----------
    model:
        Trained regression estimator.
    dataframe:
        Dataset containing input features.
    target_column:
        Target field removed from the feature matrix when present.
    timestamp_column:
        Timestamp field preserved in the prediction output.

    Returns
    -------
    pd.DataFrame
        Timestamped predictions.
    """
    features = dataframe.copy()
    timestamps = features[timestamp_column] if timestamp_column in features.columns else None

    drop_columns: Iterable[str] = [column for column in [target_column, timestamp_column] if column in features.columns]
    features = features.drop(columns=list(drop_columns))
    predictions = model.predict(features)

    result = pd.DataFrame({"prediction": predictions})
    if timestamps is not None:
        result.insert(0, timestamp_column, timestamps.values)
    return result
