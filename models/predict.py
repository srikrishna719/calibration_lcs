"""Prediction helpers for trained calibration models."""

from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd
from sklearn.base import RegressorMixin


def select_model_features(
    dataframe: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build the feature matrix a fitted model expects.

    When ``feature_names`` is given (normally ``TrainingResult.feature_names``),
    the columns are selected in exactly the order the model was fitted on. This
    matters whenever training used a feature subset, because passing the whole
    prepared dataset would otherwise hand the estimator columns it never saw.

    When ``feature_names`` is omitted the legacy behaviour applies: every column
    except the target and timestamp is used.
    """
    if feature_names is None:
        return dataframe.drop(columns=[target_column, timestamp_column], errors="ignore")

    requested: List[str] = [str(name) for name in feature_names]
    missing = [name for name in requested if name not in dataframe.columns]
    if missing:
        raise ValueError(
            "Dataset is missing feature column(s) the model was trained on: "
            + ", ".join(missing)
        )
    return dataframe[requested]


def predict_with_model(
    model: RegressorMixin,
    dataframe: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Generate timestamped predictions for a fitted model.

    Parameters
    ----------
    feature_names:
        Columns the model was fitted on, in fit order. Pass
        ``TrainingResult.feature_names`` so a model trained on a feature subset
        is fed exactly those columns.
    """
    features = select_model_features(
        dataframe=dataframe,
        target_column=target_column,
        timestamp_column=timestamp_column,
        feature_names=feature_names,
    )
    predictions = model.predict(features)
    result = pd.DataFrame({"prediction": predictions})
    if timestamp_column in dataframe.columns:
        result.insert(0, timestamp_column, dataframe[timestamp_column].values)
    return result
