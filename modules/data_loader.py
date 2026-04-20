"""Dataset loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd


DataSource = Union[str, Path, BinaryIO, pd.DataFrame]


def load_csv(source: DataSource, dataset_name: str) -> pd.DataFrame:
    """Load a CSV dataset or clone an existing DataFrame.

    Parameters
    ----------
    source:
        File path, file-like object, or DataFrame.
    dataset_name:
        Human-readable dataset name used in validation errors.

    Returns
    -------
    pd.DataFrame
        Loaded dataset copy.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()

    dataframe = pd.read_csv(source)
    return dataframe.copy()


def validate_dataset(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Validate the expected structure and types of an input dataset.

    Parameters
    ----------
    dataframe:
        Dataset to validate.
    timestamp_column:
        Name of the timestamp column expected in the dataset.
    dataset_name:
        Human-readable dataset name used in error messages.

    Returns
    -------
    pd.DataFrame
        Validated dataset with parsed timestamps.
    """
    if dataframe.empty:
        raise ValueError(f"{dataset_name} dataset is empty.")

    if timestamp_column not in dataframe.columns:
        raise ValueError(
            f"{dataset_name} dataset must contain timestamp column '{timestamp_column}'."
        )

    validated = dataframe.copy()
    validated[timestamp_column] = pd.to_datetime(
        validated[timestamp_column],
        errors="coerce",
    )

    if validated[timestamp_column].isna().all():
        raise ValueError(
            f"{dataset_name} dataset contains no valid timestamps in '{timestamp_column}'."
        )

    validated = validated.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
    validated = validated.drop_duplicates(subset=[timestamp_column]).reset_index(drop=True)
    return validated


def load_and_validate_dataset(
    source: DataSource,
    timestamp_column: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Load and validate a dataset in one step.

    Parameters
    ----------
    source:
        File path, file-like object, or DataFrame.
    timestamp_column:
        Name of the timestamp column expected in the dataset.
    dataset_name:
        Human-readable dataset name used in validation errors.

    Returns
    -------
    pd.DataFrame
        Cleanly loaded dataset ready for downstream processing.
    """
    dataframe = load_csv(source=source, dataset_name=dataset_name)
    return validate_dataset(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        dataset_name=dataset_name,
    )
