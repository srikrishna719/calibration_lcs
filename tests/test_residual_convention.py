"""Residuals must mean the same thing everywhere.

Regression cover: TrainingResult.residuals was actual - predicted while
evaluation.metrics.bias, the drift analysis and every residual plot used
predicted - actual, so the stored series had the opposite sign to everything
that consumed it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import bias
from modules.drift_analysis import compute_rolling_errors
from pipeline.run_pipeline import train_on_prepared_dataset

TARGET, TIMESTAMP = "reference_pm25", "timestamp"


@pytest.fixture(scope="module")
def result(merged, _base_config):
    from copy import deepcopy

    cfg = deepcopy(_base_config)
    cfg["training"]["selected_models"] = ["ridge"]
    out = train_on_prepared_dataset(merged, TARGET, cfg)
    return out["training_results"]["ridge"]


class TestSignConvention:
    def test_training_residuals_are_predicted_minus_actual(self, result):
        predictions = result.validation_predictions
        expected = predictions["predicted"].to_numpy() - predictions["actual"].to_numpy()
        assert np.allclose(result.residuals.to_numpy(), expected)

    def test_mean_residual_equals_reported_bias(self, result):
        """bias() is predicted - actual, so the mean residual must match it."""
        predictions = result.validation_predictions
        assert np.isclose(
            result.residuals.mean(),
            bias(predictions["actual"], predictions["predicted"]),
        )

    def test_drift_analysis_uses_the_same_convention(self, result):
        rolling = compute_rolling_errors(result.validation_predictions, window=6)
        expected = (
            result.validation_predictions["predicted"] - result.validation_predictions["actual"]
        )
        assert np.allclose(rolling["residual"].to_numpy(), expected.to_numpy())

    def test_over_prediction_gives_a_positive_residual(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "actual": [10.0, 10.0, 10.0],
            "predicted": [12.0, 12.0, 12.0],
        })
        assert (compute_rolling_errors(frame, window=2)["residual"] > 0).all()
        assert bias(frame["actual"], frame["predicted"]) > 0
