"""Residuals must mean the same thing everywhere.

Regression cover: TrainingResult.residuals was actual - predicted while
evaluation.metrics.bias and every residual plot used predicted - actual, so the
stored series had the opposite sign to everything that consumed it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import bias
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

    def test_the_residual_plots_use_the_same_convention(self, result):
        """The Residual Analysis step plots predicted - actual directly."""
        predictions = result.validation_predictions
        plotted = predictions["predicted"] - predictions["actual"]
        assert np.allclose(result.residuals.to_numpy(), plotted.to_numpy())

    def test_over_prediction_gives_a_positive_residual(self):
        actual = pd.Series([10.0, 10.0, 10.0])
        predicted = pd.Series([12.0, 12.0, 12.0])
        assert ((predicted - actual) > 0).all()
        assert bias(actual, predicted) > 0
