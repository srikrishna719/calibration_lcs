"""Model comparison helpers."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from models.train import TrainingResult


def create_leaderboard(
    training_results: List[TrainingResult],
    config: Dict[str, object],
) -> pd.DataFrame:
    """Build a sortable leaderboard from training outputs.

    Parameters
    ----------
    training_results:
        Outputs from model training.
    config:
        Evaluation settings.

    Returns
    -------
    pd.DataFrame
        Sorted leaderboard.
    """
    leaderboard = pd.DataFrame(
        [
            {"model": result.model_name, **result.metrics}
            for result in training_results
        ]
    )

    if leaderboard.empty:
        raise ValueError("No model training results were available for comparison.")

    sort_by = str(config.get("sort_by", "rmse"))
    ascending = bool(config.get("ascending", True))
    if sort_by not in leaderboard.columns:
        sort_by = "rmse"

    return leaderboard.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)


def select_best_model(
    training_results: List[TrainingResult],
    leaderboard: pd.DataFrame,
) -> TrainingResult:
    """Return the best-ranked model according to the leaderboard.

    Parameters
    ----------
    training_results:
        Outputs from model training.
    leaderboard:
        Sorted leaderboard.

    Returns
    -------
    TrainingResult
        Top-ranked model result.
    """
    best_model_name = leaderboard.iloc[0]["model"]
    for result in training_results:
        if result.model_name == best_model_name:
            return result
    raise ValueError(f"Best model '{best_model_name}' was not found in training results.")
