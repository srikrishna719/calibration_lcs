"""Feature engineering utilities for calibration models."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def create_lag_features(dataframe: pd.DataFrame, feature_columns: Iterable[str], lag_steps: List[int]) -> pd.DataFrame:
    """Create lagged copies of selected feature columns.

    Parameters
    ----------
    dataframe:
        Dataset to transform.
    feature_columns:
        Feature columns to lag.
    lag_steps:
        Positive lag steps to create.

    Returns
    -------
    pd.DataFrame
        Dataset with lag features appended.
    """
    engineered = dataframe.copy()
    for column in feature_columns:
        for lag in lag_steps:
            engineered[f"{column}_lag_{lag}"] = engineered[column].shift(lag)
    return engineered


def create_rolling_features(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
    windows: List[int],
) -> pd.DataFrame:
    """Create rolling mean and standard deviation features.

    Parameters
    ----------
    dataframe:
        Dataset to transform.
    feature_columns:
        Feature columns used for rolling calculations.
    windows:
        Rolling window sizes.

    Returns
    -------
    pd.DataFrame
        Dataset with rolling features appended.
    """
    engineered = dataframe.copy()
    for column in feature_columns:
        for window in windows:
            rolling = engineered[column].rolling(window=window, min_periods=1)
            engineered[f"{column}_roll_mean_{window}"] = rolling.mean()
            engineered[f"{column}_roll_std_{window}"] = rolling.std().fillna(0.0)
    return engineered


def create_time_features(dataframe: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Derive calendar features from a timestamp column.

    Parameters
    ----------
    dataframe:
        Dataset to transform.
    timestamp_column:
        Timestamp field.

    Returns
    -------
    pd.DataFrame
        Dataset with calendar features.
    """
    engineered = dataframe.copy()
    timestamps = pd.to_datetime(engineered[timestamp_column])
    engineered["hour"] = timestamps.dt.hour
    engineered["day_of_week"] = timestamps.dt.dayofweek
    engineered["month"] = timestamps.dt.month
    engineered["is_weekend"] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
    return engineered


def engineer_features(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    config: Dict[str, object],
) -> pd.DataFrame:
    """Apply configured feature engineering steps.

    Parameters
    ----------
    dataframe:
        Aligned merged dataset.
    timestamp_column:
        Timestamp field.
    target_column:
        Training target column to exclude from feature creation.
    config:
        Feature engineering settings.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataset.
    """
    if not bool(config.get("enabled", True)):
        return dataframe.copy()

    feature_columns = [
        column
        for column in dataframe.select_dtypes(include=[np.number]).columns.tolist()
        if column != target_column
    ]

    engineered = create_lag_features(
        dataframe=dataframe,
        feature_columns=feature_columns,
        lag_steps=[int(step) for step in config.get("lag_steps", [1, 2, 3])],
    )
    engineered = create_rolling_features(
        dataframe=engineered,
        feature_columns=feature_columns,
        windows=[int(window) for window in config.get("rolling_windows", [3, 6])],
    )

    if bool(config.get("add_time_features", True)):
        engineered = create_time_features(
            dataframe=engineered,
            timestamp_column=timestamp_column,
        )

    engineered = engineered.dropna().reset_index(drop=True)
    return engineered
