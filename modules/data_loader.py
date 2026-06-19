"""Dataset loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd


DataSource = Union[str, Path, BinaryIO, pd.DataFrame]


def load_csv(source: DataSource) -> pd.DataFrame:
    """Load a CSV dataset or clone an existing DataFrame.

    Parameters
    ----------
    source:
        File path, file-like object, or DataFrame.

    Returns
    -------
    pd.DataFrame
        Loaded dataset copy.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(source).copy()


def validate_dataset(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Validate required structure for an uploaded dataset.

    Parameters
    ----------
    dataframe:
        Dataset to validate.
    timestamp_column:
        Name of the timestamp column.
    dataset_name:
        Human-readable dataset name for error messages.

    Returns
    -------
    pd.DataFrame
        Validated dataset.
    """
    if dataframe.empty:
        raise ValueError(f"{dataset_name} dataset is empty.")
    if timestamp_column not in dataframe.columns:
        raise ValueError(
            f"{dataset_name} dataset must contain timestamp column '{timestamp_column}'."
        )

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        raise ValueError(f"{dataset_name} dataset must contain at least one numeric column.")

    return dataframe.copy()


def parse_and_normalize_timestamps(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Convert timestamps to timezone-naive UTC-normalized datetimes.

    Parameters
    ----------
    dataframe:
        Dataset with a timestamp column.
    timestamp_column:
        Timestamp field to parse.
    timezone:
        Timezone assumed for naive timestamps before conversion to UTC.

    Returns
    -------
    pd.DataFrame
        Dataset with normalized timestamps.
    """
    transformed = dataframe.copy()
    timestamps = pd.to_datetime(transformed[timestamp_column], errors="coerce")
    if timestamps.isna().all():
        raise ValueError(
            f"Dataset contains no valid timestamps in '{timestamp_column}'."
        )

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")

    transformed[timestamp_column] = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    transformed = transformed.dropna(subset=[timestamp_column])
    transformed = transformed.sort_values(timestamp_column)
    transformed = transformed.drop_duplicates(subset=[timestamp_column]).reset_index(drop=True)
    return transformed


def load_and_validate_dataset(
    source: DataSource,
    timestamp_column: str,
    dataset_name: str,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Load, validate, and normalize a dataset in one step.

    Parameters
    ----------
    source:
        File path, file-like object, or DataFrame.
    timestamp_column:
        Timestamp field expected in the dataset.
    dataset_name:
        Human-readable dataset name.
    timezone:
        Assumed timezone for naive timestamps.

    Returns
    -------
    pd.DataFrame
        Cleanly loaded dataset.
    """
    dataframe = load_csv(source)
    validated = validate_dataset(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        dataset_name=dataset_name,
    )
    return parse_and_normalize_timestamps(
        dataframe=validated,
        timestamp_column=timestamp_column,
        timezone=timezone,
    )
