"""Timestamp alignment and dataset merge utilities.

Supports resampling, cross-correlation-based lag detection,
and both inner-join and nearest-timestamp (merge_asof) merge strategies.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def non_numeric_columns(
    dataframe: pd.DataFrame,
    timestamp_column: str,
) -> List[str]:
    """Return the columns that cannot be aggregated during resampling."""
    numeric = set(dataframe.select_dtypes(include=[np.number]).columns.tolist())
    return [
        str(column)
        for column in dataframe.columns
        if column != timestamp_column and column not in numeric
    ]


def resample_timeseries(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    rule: str,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Resample a time series dataset by timestamp.

    Only numeric columns are aggregated. Text columns (site IDs, station names,
    QA flags) are dropped rather than passed to the aggregation function, which
    would otherwise raise an opaque pandas TypeError. Use
    :func:`non_numeric_columns` to report what will be dropped.

    Parameters
    ----------
    dataframe:
        Input dataset with a timestamp column.
    timestamp_column:
        Name of the timestamp column.
    rule:
        Pandas resample rule string (e.g., '1h', '15min').
    aggregation:
        Aggregation function ('mean', 'median', 'sum').

    Returns
    -------
    pd.DataFrame
        Resampled dataset containing the timestamp and numeric columns only.
    """
    df = dataframe.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    numeric_cols = [
        str(column)
        for column in df.select_dtypes(include=[np.number]).columns
        if column != timestamp_column
    ]
    if not numeric_cols:
        raise ValueError(
            "Dataset has no numeric columns to resample. "
            "Check that measurement columns were parsed as numbers, not text."
        )

    return (
        df[[timestamp_column, *numeric_cols]]
        .set_index(timestamp_column)
        .resample(rule.strip())
        .agg(aggregation)
        .dropna(how="all")
        .reset_index()
    )


def detect_optimal_lag(
    reference_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    timestamp_column: str,
    reference_target_column: str,
    sensor_feature_column: str,
    max_lag_steps: int = 0,
) -> int:
    """Estimate the lag maximizing cross-correlation between two series.

    Parameters
    ----------
    reference_df:
        Reference-grade dataset.
    sensor_df:
        Low-cost sensor dataset.
    timestamp_column:
        Timestamp column name shared by both datasets.
    reference_target_column:
        Target column in the reference dataset.
    sensor_feature_column:
        Primary sensor column to correlate with.
    max_lag_steps:
        Maximum lag (positive and negative) to test.

    Returns
    -------
    int
        Optimal lag in timesteps.  Positive = sensor leads.
    """
    merged = reference_df[[timestamp_column, reference_target_column]].merge(
        sensor_df[[timestamp_column, sensor_feature_column]],
        on=timestamp_column,
        how="inner",
    ).dropna()
    if merged.empty:
        return 0

    ref_series = merged[reference_target_column]
    sen_series = merged[sensor_feature_column]
    best_lag = 0
    best_score = -np.inf

    for lag in range(-max_lag_steps, max_lag_steps + 1):
        shifted = sen_series.shift(lag)
        candidate = pd.concat([ref_series, shifted], axis=1).dropna()
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
    """Shift sensor columns by the detected lag."""
    shifted = sensor_df.sort_values(timestamp_column).copy()
    non_ts_cols = [c for c in shifted.columns if c != timestamp_column]
    shifted[non_ts_cols] = shifted[non_ts_cols].shift(lag_steps)
    return shifted.dropna().reset_index(drop=True)


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
        Preprocessed reference dataset.
    sensor_df:
        Preprocessed sensor dataset.
    timestamp_column:
        Shared timestamp column name.
    reference_target_column:
        Target column in the reference data.
    sensor_prefix:
        Prefix for sensor column names after merge.
    reference_prefix:
        Prefix for reference column names after merge.
    config:
        Alignment configuration dict with keys:
        ``resample_rule``, ``aggregation``, ``max_lag_steps``,
        ``lag_column``, ``merge_strategy``.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, object]]
        Merged dataset and alignment metadata.
    """
    rule = str(config.get("resample_rule", "1h"))
    aggregation = str(config.get("aggregation", "mean"))
    merge_strategy = str(config.get("merge_strategy", "inner"))

    dropped_columns = {
        "reference": non_numeric_columns(reference_df, timestamp_column),
        "sensor": non_numeric_columns(sensor_df, timestamp_column),
    }

    ref_resampled = resample_timeseries(reference_df, timestamp_column, rule, aggregation)
    sen_resampled = resample_timeseries(sensor_df, timestamp_column, rule, aggregation)

    sensor_numeric_cols: List[str] = [
        c for c in sen_resampled.columns if c != timestamp_column
    ]
    if not sensor_numeric_cols:
        raise ValueError("Sensor dataset must contain at least one numeric feature column.")

    lag_column = str(config.get("lag_column", "auto"))
    sensor_feature_col = sensor_numeric_cols[0] if lag_column == "auto" else lag_column
    if sensor_feature_col not in sen_resampled.columns:
        raise ValueError(f"Lag detection column '{sensor_feature_col}' not found in sensor dataset.")

    lag_steps = detect_optimal_lag(
        reference_df=ref_resampled,
        sensor_df=sen_resampled,
        timestamp_column=timestamp_column,
        reference_target_column=reference_target_column,
        sensor_feature_column=sensor_feature_col,
        max_lag_steps=int(config.get("max_lag_steps", 0)),
    )

    sensor_aligned = apply_lag(sen_resampled, timestamp_column, lag_steps)

    # Rename columns with prefixes
    ref_renamed = ref_resampled.rename(
        columns={
            c: f"{reference_prefix}_{c}"
            for c in ref_resampled.columns if c != timestamp_column
        }
    )
    sen_renamed = sensor_aligned.rename(
        columns={
            c: f"{sensor_prefix}_{c}"
            for c in sensor_aligned.columns if c != timestamp_column
        }
    )

    # Merge using selected strategy
    if merge_strategy == "nearest":
        ref_renamed[timestamp_column] = pd.to_datetime(ref_renamed[timestamp_column])
        sen_renamed[timestamp_column] = pd.to_datetime(sen_renamed[timestamp_column])
        ref_sorted = ref_renamed.sort_values(timestamp_column)
        sen_sorted = sen_renamed.sort_values(timestamp_column)
        merged = pd.merge_asof(
            ref_sorted, sen_sorted,
            on=timestamp_column,
            direction="nearest",
        )
    else:
        merged = ref_renamed.merge(sen_renamed, on=timestamp_column, how="inner")

    merged = merged.dropna().reset_index(drop=True)
    if merged.empty:
        raise ValueError("No overlapping aligned records were found between the datasets.")

    matched_records = len(merged)
    reference_records = len(ref_resampled)
    sensor_records = len(sen_resampled)
    unmatched_records = max(0, reference_records + sensor_records - (matched_records * 2))
    alignment_percentage = round(
        (matched_records / max(reference_records, sensor_records, 1)) * 100,
        2,
    )

    return merged, {
        "lag_steps": lag_steps,
        "resample_rule": rule,
        "aggregation": aggregation,
        "merge_strategy": merge_strategy,
        "lag_detection_column": sensor_feature_col,
        "merged_rows": matched_records,
        "matched_records": matched_records,
        "unmatched_records": unmatched_records,
        "alignment_percentage": alignment_percentage,
        "reference_rows_after_resample": reference_records,
        "sensor_rows_after_resample": sensor_records,
        "dropped_non_numeric_columns": dropped_columns,
    }
