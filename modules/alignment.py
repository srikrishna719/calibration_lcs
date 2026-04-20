"""Timestamp alignment and lag detection utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def resample_timeseries(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    rule: str,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Resample a time series dataset by timestamp.

    Parameters
    ----------
    dataframe:
        Dataset to resample.
    timestamp_column:
        Timestamp field.
    rule:
        Pandas resampling frequency rule.
    aggregation:
        Aggregation function name.

    Returns
    -------
    pd.DataFrame
        Resampled dataset.
    """
    resampled = (
        dataframe.set_index(timestamp_column)
        .resample(rule)
        .agg(aggregation)
        .dropna(how="all")
        .reset_index()
    )
    return resampled


def detect_optimal_lag(
    reference_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    timestamp_column: str,
    reference_target_column: str,
    sensor_feature_column: str,
    max_lag_steps: int = 6,
) -> int:
    """Estimate the lag that maximizes correlation between reference and sensor series.

    Parameters
    ----------
    reference_df:
        Reference dataset.
    sensor_df:
        Sensor dataset.
    timestamp_column:
        Timestamp field shared by both datasets.
    reference_target_column:
        Reference target series used for alignment.
    sensor_feature_column:
        Sensor series compared against the target.
    max_lag_steps:
        Maximum lag in either direction.

    Returns
    -------
    int
        Best lag in resampled time steps. Positive lag shifts sensor data forward.
    """
    merged = reference_df[[timestamp_column, reference_target_column]].merge(
        sensor_df[[timestamp_column, sensor_feature_column]],
        on=timestamp_column,
        how="inner",
    ).dropna()

    if merged.empty:
        return 0

    reference_series = merged[reference_target_column]
    sensor_series = merged[sensor_feature_column]
    best_lag = 0
    best_score = -np.inf

    for lag in range(-max_lag_steps, max_lag_steps + 1):
        shifted = sensor_series.shift(lag)
        candidate = pd.concat([reference_series, shifted], axis=1).dropna()
        if candidate.empty:
            continue
        score = candidate.corr().iloc[0, 1]
        if pd.notna(score) and score > best_score:
            best_score = float(score)
            best_lag = lag

    return best_lag


def apply_lag(
    sensor_df: pd.DataFrame,
    timestamp_column: str,
    lag_steps: int,
) -> pd.DataFrame:
    """Shift sensor timestamps by a number of rows to compensate detected lag.

    Parameters
    ----------
    sensor_df:
        Sensor dataset to shift.
    timestamp_column:
        Timestamp field.
    lag_steps:
        Number of steps to shift forward or backward.

    Returns
    -------
    pd.DataFrame
        Lag-adjusted dataset.
    """
    shifted = sensor_df.copy()
    shifted = shifted.sort_values(timestamp_column).reset_index(drop=True)
    non_timestamp_columns = [column for column in shifted.columns if column != timestamp_column]
    shifted[non_timestamp_columns] = shifted[non_timestamp_columns].shift(lag_steps)
    shifted = shifted.dropna().reset_index(drop=True)
    return shifted


def align_and_merge_datasets(
    reference_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    timestamp_column: str,
    reference_target_column: str,
    sensor_prefix: str,
    reference_prefix: str,
    config: Dict[str, object],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Resample, lag-align, and merge reference and sensor datasets.

    Parameters
    ----------
    reference_df:
        Reference-grade dataset.
    sensor_df:
        Low-cost sensor dataset.
    timestamp_column:
        Shared timestamp field.
    reference_target_column:
        Target field in the reference dataset.
    sensor_prefix:
        Prefix applied to sensor columns before merge.
    reference_prefix:
        Prefix applied to reference columns before merge.
    config:
        Alignment settings.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, object]]
        Merged dataset and alignment metadata.
    """
    rule = str(config.get("resample_rule", "1H"))
    aggregation = str(config.get("aggregation", "mean"))

    reference_resampled = resample_timeseries(
        dataframe=reference_df,
        timestamp_column=timestamp_column,
        rule=rule,
        aggregation=aggregation,
    )
    sensor_resampled = resample_timeseries(
        dataframe=sensor_df,
        timestamp_column=timestamp_column,
        rule=rule,
        aggregation=aggregation,
    )

    sensor_numeric_columns: List[str] = [
        column for column in sensor_resampled.columns if column != timestamp_column
    ]
    if not sensor_numeric_columns:
        raise ValueError("Sensor dataset must contain at least one numeric feature column.")

    lag_column = str(config.get("lag_column", "auto"))
    sensor_feature_column = sensor_numeric_columns[0] if lag_column == "auto" else lag_column
    if sensor_feature_column not in sensor_resampled.columns:
        raise ValueError(f"Lag detection column '{sensor_feature_column}' not found in sensor dataset.")

    lag_steps = detect_optimal_lag(
        reference_df=reference_resampled,
        sensor_df=sensor_resampled,
        timestamp_column=timestamp_column,
        reference_target_column=reference_target_column,
        sensor_feature_column=sensor_feature_column,
        max_lag_steps=int(config.get("max_lag_steps", 6)),
    )
    sensor_aligned = apply_lag(
        sensor_df=sensor_resampled,
        timestamp_column=timestamp_column,
        lag_steps=lag_steps,
    )

    reference_renamed = reference_resampled.rename(
        columns={
            column: f"{reference_prefix}_{column}"
            for column in reference_resampled.columns
            if column != timestamp_column
        }
    )
    sensor_renamed = sensor_aligned.rename(
        columns={
            column: f"{sensor_prefix}_{column}"
            for column in sensor_aligned.columns
            if column != timestamp_column
        }
    )

    merged = reference_renamed.merge(sensor_renamed, on=timestamp_column, how="inner")
    merged = merged.dropna().reset_index(drop=True)
    if merged.empty:
        raise ValueError("No overlapping aligned records were found between the datasets.")

    metadata = {
        "lag_steps": lag_steps,
        "resample_rule": rule,
        "aggregation": aggregation,
        "lag_detection_column": sensor_feature_column,
    }
    return merged, metadata
