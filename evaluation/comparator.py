"""Model comparison utilities — leaderboard and best-model selection."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from models.train import TrainingResult


# Columns displayed on the leaderboard, in order.
LEADERBOARD_COLUMNS = [
    "rank",
    "model_name",
    "rmse",
    "mae",
    "r2",
    "mape",
    "bias",
    "pearson_r",
    "slope",
    "intercept",
]


def create_leaderboard(
    training_results: List[TrainingResult],
    config: Dict[str, object],
) -> pd.DataFrame:
    """Build a sortable leaderboard from training outputs.

    Parameters
    ----------
    training_results:
        Output from the training stage.
    config:
        Evaluation configuration with ``sort_by`` and ``ascending`` keys.

    Returns
    -------
    pd.DataFrame
        Ranked model leaderboard.
    """
    rows = []
    for result in training_results:
        row = {"model_name": result.model_name}
        row.update(result.metrics)
        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    # A diagnostic about MAPE's coverage, not a score to rank models by.
    leaderboard = leaderboard.drop(columns=["mape_excluded_fraction"], errors="ignore")
    if leaderboard.empty:
        raise ValueError("No model training results were available for comparison.")

    sort_by = str(config.get("sort_by", "rmse"))
    ascending = bool(config.get("ascending", True))
    if sort_by not in leaderboard.columns:
        sort_by = "rmse"

    leaderboard = leaderboard.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))

    # Reorder columns for presentation — keep extras at the end
    ordered = [col for col in LEADERBOARD_COLUMNS if col in leaderboard.columns]
    extras = [col for col in leaderboard.columns if col not in ordered]
    leaderboard = leaderboard[ordered + extras]

    return leaderboard


def select_best_model(
    training_results: List[TrainingResult],
    leaderboard: pd.DataFrame,
) -> TrainingResult:
    """Return the best-ranked model according to the leaderboard.

    Parameters
    ----------
    training_results:
        Full list of training outputs.
    leaderboard:
        Sorted leaderboard from ``create_leaderboard``.

    Returns
    -------
    TrainingResult
        The top-ranked training result.
    """
    best_model_name = leaderboard.iloc[0]["model_name"]
    for result in training_results:
        if result.model_name == best_model_name:
            return result
    raise ValueError(f"Best model '{best_model_name}' was not found in training results.")
