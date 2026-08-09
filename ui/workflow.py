"""Workflow helpers for the Streamlit UI.

This module keeps navigation, mode filtering, and column-name suggestions out
of the main page file so the application logic can be split gradually without
changing the user-facing workflow.
"""

from __future__ import annotations

from typing import Iterable, Sequence


APP_MODES = ("Basic", "Advanced")

ADVANCED_STEP_KEYS = frozenset(
    {
        "statistical_diagnostics",
        "residual_analysis",
        "readme",
    }
)

TIMESTAMP_CANDIDATES = (
    "timestamp",
    "datetime",
    "date_time",
    "date time",
    "sample_time",
    "sampling_time",
    "created_at",
    "time",
    "date",
)

TARGET_CANDIDATES = (
    "pm25",
    "pm2_5",
    "pm2.5",
    "reference_pm25",
    "pm10",
    "no2",
    "o3",
    "co",
    "so2",
)


def is_advanced_mode(app_mode: str | None) -> bool:
    return app_mode == "Advanced"


def visible_step_pairs(
    steps: Sequence[str],
    step_keys: Sequence[str],
    app_mode: str | None,
) -> list[tuple[str, str]]:
    pairs = list(zip(steps, step_keys))
    if is_advanced_mode(app_mode):
        return pairs
    return [(label, key) for label, key in pairs if key not in ADVANCED_STEP_KEYS]


def visible_steps(
    steps: Sequence[str],
    step_keys: Sequence[str],
    app_mode: str | None,
) -> list[str]:
    return [label for label, _ in visible_step_pairs(steps, step_keys, app_mode)]


def normalize_current_step(
    current_step: str,
    steps: Sequence[str],
    step_keys: Sequence[str],
    app_mode: str | None,
) -> str:
    visible = visible_steps(steps, step_keys, app_mode)
    if current_step in visible:
        return current_step
    return visible[0] if visible else current_step


def next_step(
    current_step: str,
    steps: Sequence[str],
    step_keys: Sequence[str],
    app_mode: str | None,
) -> str:
    visible = visible_steps(steps, step_keys, app_mode)
    if current_step not in visible:
        return visible[0] if visible else current_step
    idx = visible.index(current_step)
    if idx < len(visible) - 1:
        return visible[idx + 1]
    return current_step


def _normalize_column_name(name: object) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _first_candidate(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    columns = list(columns)
    exact = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        match = exact.get(candidate.lower())
        if match:
            return match

    normalized = {_normalize_column_name(column): str(column) for column in columns}
    for candidate in candidates:
        match = normalized.get(_normalize_column_name(candidate))
        if match:
            return match

    for candidate in candidates:
        candidate_norm = _normalize_column_name(candidate)
        for column in columns:
            if candidate_norm and candidate_norm in _normalize_column_name(column):
                return str(column)
    return None


def suggest_column_setup(
    reference_columns: Sequence[str],
    sensor_columns: Sequence[str],
    configured_timestamp: str = "timestamp",
    configured_target: str = "pm25",
) -> tuple[str, str]:
    """Suggest timestamp and target columns from available CSV headers."""
    reference_columns = [str(column) for column in reference_columns]
    sensor_columns = [str(column) for column in sensor_columns]

    common_columns = [column for column in reference_columns if column in set(sensor_columns)]
    timestamp = configured_timestamp
    if configured_timestamp not in common_columns:
        timestamp = (
            _first_candidate(common_columns, TIMESTAMP_CANDIDATES)
            or _first_candidate(reference_columns, TIMESTAMP_CANDIDATES)
            or configured_timestamp
        )

    target = configured_target
    if configured_target not in reference_columns:
        target = _first_candidate(reference_columns, TARGET_CANDIDATES) or configured_target

    return timestamp, target
