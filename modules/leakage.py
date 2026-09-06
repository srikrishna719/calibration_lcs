"""Detection of predictors that encode the calibration target.

Co-location exports often ship derived columns alongside the raw channels --
``pm25_sensor_minus_reference``, residuals, error terms. Handed to a model as
predictors they reconstruct the reference value algebraically, so the model
scores near-perfectly while having learned nothing transferable.

Correlation does not find them. In the 2025 co-location dataset
``pm25_sensor_minus_reference`` has a univariate R2 of 0.15 against the target,
which looks unremarkable, yet paired with ``sensor_pm25`` it reproduces the
target to 1.0. The signal is joint, so the scan is pairwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# An exact algebraic identity lands at 1.0 up to floating-point noise. A genuine
# calibration does not: the best honest pair in the reference dataset reaches
# 0.39, and even a leakage-adjacent pair only 0.99. The gap is wide, so the
# threshold sits well above any plausible real fit.
DEFAULT_R2_THRESHOLD = 0.9999

# Pairwise scanning is quadratic. Base merged frames are narrow, but engineered
# frames are not, so refuse rather than stall.
DEFAULT_MAX_COLUMNS = 60

# Naming conventions for derived columns. Used only to decide which member of a
# flagged pair to drop -- never to find leakage in the first place, so a column
# following no convention is still caught.
_DERIVED_MARKERS = (
    "minus", "_diff", "diff_", "_delta", "delta_",
    "residual", "_error", "error_", "_bias", "bias_", "_offset",
)


@dataclass
class LeakageReport:
    """Which predictors reconstruct the target, and what to do about them."""

    excluded: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)
    exact_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    solo: List[Tuple[str, float]] = field(default_factory=list)
    scanned_columns: int = 0
    skipped_reason: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.excluded)

    def summary(self) -> str:
        """One line per excluded column, suitable for a UI warning or a log."""
        return "\n".join(f"{name}: {self.reasons[name]}" for name in self.excluded)


def _looks_derived(name: str) -> bool:
    lowered = str(name).lower()
    return any(marker in lowered for marker in _DERIVED_MARKERS)


def _r2(design: np.ndarray, y: np.ndarray) -> float:
    """R2 of an ordinary least squares fit of ``y`` on ``design`` plus intercept."""
    matrix = np.column_stack([design, np.ones(len(design))])
    total = float(((y - y.mean()) ** 2).sum())
    if total <= 0:
        return float("nan")
    try:
        beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
        return float("nan")
    residual = float(((y - matrix @ beta) ** 2).sum())
    return 1.0 - residual / total


def find_target_encoding_columns(
    dataframe: pd.DataFrame,
    target_column: str,
    candidates: Optional[Iterable[str]] = None,
    *,
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    min_rows: int = 10,
    subject: str = "the target",
) -> LeakageReport:
    """Find predictors that reproduce ``target_column`` algebraically.

    Two passes. The first catches a column that is the target under another
    name; the second catches pairs whose linear combination is the target,
    which is the shape a ``sensor - reference`` difference column takes.

    A flagged pair is only dangerous together, so where possible just one member
    is excluded: the one whose name follows a derived-column convention, most
    implicated pairs first. Where no name gives that away, both are excluded --
    guessing would risk keeping the leak, and the reason text tells the user
    which one to re-enable.
    """
    report = LeakageReport()

    if target_column not in dataframe.columns:
        report.skipped_reason = f"target column '{target_column}' is not in the dataset"
        return report

    numeric = dataframe.select_dtypes(include="number")
    pool = [
        str(c) for c in (candidates if candidates is not None else numeric.columns)
        if str(c) in numeric.columns and str(c) != target_column
    ]
    if not pool:
        return report

    if len(pool) > max_columns:
        report.skipped_reason = (
            f"{len(pool)} candidate columns exceeds the {max_columns}-column scan limit; "
            "run the check on the base merged columns instead of engineered features"
        )
        return report

    frame = numeric[[target_column, *pool]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < min_rows:
        report.skipped_reason = f"only {len(frame)} complete rows; need at least {min_rows}"
        return report

    y = frame[target_column].to_numpy(dtype=float)
    usable = [c for c in pool if frame[c].nunique() > 1]
    report.scanned_columns = len(usable)

    # Pass 1 -- a column that is the target rescaled.
    solo_flagged: List[str] = []
    for column in usable:
        score = _r2(frame[[column]].to_numpy(dtype=float), y)
        if np.isfinite(score) and score >= r2_threshold:
            solo_flagged.append(column)
            report.solo.append((column, float(score)))
            report.reasons[column] = (
                f"reproduces {subject} on its own (R2={score:.6f}); it is {subject} "
                "under another name, not a predictor"
            )
    remaining = [c for c in usable if c not in solo_flagged]

    # Pass 2 -- pairs whose linear combination is the target.
    pairs: List[Tuple[str, str, float]] = []
    for i, first in enumerate(remaining):
        for second in remaining[i + 1:]:
            score = _r2(frame[[first, second]].to_numpy(dtype=float), y)
            if np.isfinite(score) and score >= r2_threshold:
                pairs.append((first, second, float(score)))
    report.exact_pairs = pairs

    dropped: List[str] = []
    outstanding = list(pairs)

    def _partners(name: str, pool: List[Tuple[str, str, float]]) -> List[str]:
        return sorted({a if b == name else b for a, b, _ in pool if name in (a, b)})

    def _best_score(name: str, pool: List[Tuple[str, str, float]]) -> float:
        return max(score for a, b, score in pool if name in (a, b))

    # Drop derived-looking columns first, most-implicated first: one removal can
    # clear several pairs at once.
    while outstanding:
        counts: Dict[str, int] = {}
        for first, second, _ in outstanding:
            counts[first] = counts.get(first, 0) + 1
            counts[second] = counts.get(second, 0) + 1

        derived = [name for name in counts if _looks_derived(name)]
        if not derived:
            break

        victim = sorted(derived, key=lambda n: (-counts[n], n))[0]
        dropped.append(victim)
        report.reasons[victim] = (
            f"combined with {', '.join(_partners(victim, outstanding))} it reconstructs "
            f"{subject} exactly (R2={_best_score(victim, outstanding):.6f}); it carries the "
            "reference value, so a model using it cannot be applied to a sensor on its own"
        )
        outstanding = [p for p in outstanding if victim not in (p[0], p[1])]

    # Whatever is left has no naming signal to say which member is the artefact.
    # Guessing risks keeping the leak, so drop both and let the reason explain.
    for first, second, score in outstanding:
        for name in (first, second):
            if name in dropped:
                continue
            other = second if name is first else first
            dropped.append(name)
            report.reasons[name] = (
                f"together with {other} it reconstructs {subject} exactly (R2={score:.6f}). "
                "Neither name identifies which is the derived column, so both are excluded; "
                "re-enable the one you know is an independent measurement"
            )

    report.excluded = solo_flagged + dropped
    return report


def drop_target_encoding_columns(
    dataframe: pd.DataFrame,
    target_column: str,
    candidates: Optional[Sequence[str]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, LeakageReport]:
    """Return the frame without target-encoding predictors, plus the report."""
    report = find_target_encoding_columns(dataframe, target_column, candidates, **kwargs)
    if not report.excluded:
        return dataframe, report
    return dataframe.drop(columns=report.excluded, errors="ignore"), report


def find_reference_encoding_columns(
    dataframe: pd.DataFrame,
    reference_columns: Sequence[str],
    candidates: Optional[Iterable[str]] = None,
    **kwargs,
) -> LeakageReport:
    """Find predictors that carry reference-instrument readings.

    A ``sensor_x - reference_x`` difference column is not target leakage when
    the target is a different pollutant -- it does not reconstruct PM2.5 -- but
    it still embeds a reading only the reference station has, so a model using
    it cannot run on a deployed sensor. Detected by treating each reference
    column as the thing to reconstruct, which finds these regardless of naming.
    """
    combined = LeakageReport()
    pool = list(candidates) if candidates is not None else [
        c for c in dataframe.select_dtypes(include="number").columns
        if c not in set(reference_columns)
    ]

    for reference_column in reference_columns:
        scoped = [c for c in pool if c != reference_column]
        report = find_target_encoding_columns(
            dataframe, reference_column, scoped, subject=f"'{reference_column}'", **kwargs
        )
        combined.scanned_columns = max(combined.scanned_columns, report.scanned_columns)
        combined.skipped_reason = combined.skipped_reason or report.skipped_reason
        combined.exact_pairs.extend(report.exact_pairs)
        combined.solo.extend(report.solo)
        for name in report.excluded:
            if name not in combined.reasons:
                combined.excluded.append(name)
                combined.reasons[name] = report.reasons[name]

    return combined
