"""Dataset loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


DataSource = Union[str, Path, BinaryIO, pd.DataFrame]

# How to resolve rows that share a timestamp. "error" is the default because a
# repeated timestamp means the file holds more than one series, and silently
# keeping one arbitrary row per timestamp interleaves them.
DUPLICATE_STRATEGIES = ("error", "first", "last", "mean", "median")

# A grouping key has to separate the series without slicing the data to dust.
_MAX_GROUP_CARDINALITY = 200


@dataclass
class DuplicateTimestampSummary:
    """What repeated timestamps were found, and what was done about them."""

    total_rows: int = 0
    unique_timestamps: int = 0
    duplicate_rows: int = 0
    resolved_rows: int = 0
    strategy: str = "error"
    group_column: Optional[str] = None
    group_value: Optional[str] = None
    candidate_group_columns: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.duplicate_rows > 0

    @property
    def rows_discarded(self) -> int:
        return max(0, self.total_rows - self.resolved_rows)

    def message(self) -> str:
        """Human-readable account, suitable for a UI notice or an exception."""
        selection = (
            f" Kept only {self.group_column}='{self.group_value}' "
            f"({self.resolved_rows} rows)."
            if self.group_value is not None else ""
        )
        if not self.duplicate_rows:
            return ("Timestamps are unique." + selection).strip()

        text = (
            f"{self.duplicate_rows} of {self.total_rows} rows repeat a timestamp "
            f"({self.unique_timestamps} distinct timestamps)."
        )
        if self.candidate_group_columns:
            text += (
                " Timestamps become unique once split by: "
                + ", ".join(f"'{c}'" for c in self.candidate_group_columns)
                + ", so the file probably holds several series (one per device or site)."
            )
        if selection:
            text += selection
        elif self.strategy in ("mean", "median"):
            text += f" Rows sharing a timestamp were combined with the {self.strategy}."
        elif self.strategy in ("first", "last"):
            text += (
                f" Kept the {self.strategy} row per timestamp, discarding "
                f"{self.rows_discarded}."
            )
        return text


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


def find_group_columns(
    dataframe: pd.DataFrame,
    timestamp_column: str,
) -> List[str]:
    """Columns that make timestamps unique when combined with them.

    A device or site identifier satisfies this by construction: co-located
    instruments report on the same clock, so the timestamp alone is ambiguous
    while (device, timestamp) is not.
    """
    candidates: List[str] = []
    for column in dataframe.columns:
        if column == timestamp_column:
            continue
        distinct = dataframe[column].nunique(dropna=False)
        if distinct <= 1 or distinct > _MAX_GROUP_CARDINALITY:
            continue
        if not dataframe.duplicated(subset=[timestamp_column, column]).any():
            candidates.append(str(column))
    return candidates


def summarize_duplicate_timestamps(
    dataframe: pd.DataFrame,
    timestamp_column: str,
) -> DuplicateTimestampSummary:
    """Report repeated timestamps and any column that would separate them."""
    summary = DuplicateTimestampSummary(
        total_rows=len(dataframe),
        unique_timestamps=int(dataframe[timestamp_column].nunique(dropna=False)),
        duplicate_rows=int(dataframe.duplicated(subset=[timestamp_column]).sum()),
    )
    if summary.duplicate_rows:
        summary.candidate_group_columns = find_group_columns(dataframe, timestamp_column)
    return summary


def resolve_duplicate_timestamps(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    strategy: str = "error",
    group_column: Optional[str] = None,
    group_value: Optional[object] = None,
    dataset_name: str = "Dataset",
) -> Tuple[pd.DataFrame, DuplicateTimestampSummary]:
    """Reduce the frame to one row per timestamp under an explicit strategy.

    Downstream stages assume a single series, so this is where a multi-device
    file has to become one. Selecting ``group_value`` keeps that device's rows;
    ``mean``/``median`` combine co-located devices into a composite series;
    ``first``/``last`` keep an arbitrary row and are only appropriate for
    genuine exact duplicates.
    """
    normalized = str(strategy or "error").strip().lower()
    if normalized not in DUPLICATE_STRATEGIES:
        raise ValueError(
            f"Duplicate timestamp strategy must be one of {list(DUPLICATE_STRATEGIES)}, "
            f"got '{strategy}'."
        )

    # Describe the file as it arrived, so the account survives a group filter
    # that happens to remove the duplication entirely.
    summary = summarize_duplicate_timestamps(dataframe, timestamp_column)
    summary.strategy = normalized
    summary.group_column = group_column
    summary.group_value = None if group_value is None else str(group_value)

    frame = dataframe
    if group_column is not None and group_value is not None:
        if group_column not in frame.columns:
            raise ValueError(f"{dataset_name} has no column '{group_column}' to select on.")
        frame = frame[frame[group_column].astype(str) == str(group_value)]
        if frame.empty:
            raise ValueError(
                f"{dataset_name} has no rows with {group_column}='{group_value}'."
            )

    if not frame.duplicated(subset=[timestamp_column]).any():
        summary.resolved_rows = len(frame)
        return frame.reset_index(drop=True), summary

    if normalized == "error":
        raise ValueError(
            f"{dataset_name}: {summary.message()} Choose a series to keep, or combine "
            "them, via the duplicate-timestamp setting."
        )

    if normalized in ("first", "last"):
        resolved = frame.drop_duplicates(subset=[timestamp_column], keep=normalized)
    else:
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
        other = [c for c in frame.columns if c not in numeric and c != timestamp_column]
        aggregation = {c: normalized for c in numeric if c != timestamp_column}
        aggregation.update({c: "first" for c in other})
        resolved = (
            frame.groupby(timestamp_column, as_index=False)
            .agg(aggregation)
            .reindex(columns=frame.columns)
        )

    resolved = resolved.sort_values(timestamp_column).reset_index(drop=True)
    summary.resolved_rows = len(resolved)
    return resolved, summary


def parse_and_normalize_timestamps(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    timezone: str = "UTC",
    duplicate_strategy: str = "error",
    group_column: Optional[str] = None,
    group_value: Optional[object] = None,
    dataset_name: str = "Dataset",
) -> Tuple[pd.DataFrame, DuplicateTimestampSummary]:
    """Convert timestamps to timezone-naive UTC-normalized datetimes.

    Parameters
    ----------
    dataframe:
        Dataset with a timestamp column.
    timestamp_column:
        Timestamp field to parse.
    timezone:
        Timezone assumed for naive timestamps before conversion to UTC.
    duplicate_strategy:
        How to reduce rows sharing a timestamp. See
        :func:`resolve_duplicate_timestamps`.
    group_column, group_value:
        Optional device/site column and the value to keep.

    Returns
    -------
    Tuple[pd.DataFrame, DuplicateTimestampSummary]
        Dataset with normalized, unique timestamps, and an account of any
        duplicates that were resolved.
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

    return resolve_duplicate_timestamps(
        dataframe=transformed,
        timestamp_column=timestamp_column,
        strategy=duplicate_strategy,
        group_column=group_column,
        group_value=group_value,
        dataset_name=dataset_name,
    )


def load_and_validate_dataset(
    source: DataSource,
    timestamp_column: str,
    dataset_name: str,
    timezone: str = "UTC",
    duplicate_strategy: str = "error",
    group_column: Optional[str] = None,
    group_value: Optional[object] = None,
    return_summary: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, DuplicateTimestampSummary]:
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
    duplicate_strategy, group_column, group_value:
        How to reduce rows sharing a timestamp; see
        :func:`resolve_duplicate_timestamps`.
    return_summary:
        Also return the duplicate-timestamp account.

    Returns
    -------
    pd.DataFrame
        Cleanly loaded dataset, or ``(dataset, summary)`` when
        ``return_summary`` is set.
    """
    dataframe = load_csv(source)
    validated = validate_dataset(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        dataset_name=dataset_name,
    )
    result, summary = parse_and_normalize_timestamps(
        dataframe=validated,
        timestamp_column=timestamp_column,
        timezone=timezone,
        duplicate_strategy=duplicate_strategy,
        group_column=group_column,
        group_value=group_value,
        dataset_name=dataset_name,
    )
    return (result, summary) if return_summary else result
