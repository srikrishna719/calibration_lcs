"""Scaling as part of the model rather than a prior transform of the dataset.

Regression cover for the worst defect found: the scaler was fit across every
row before any split and then discarded. The exported model therefore could not
be applied to raw sensor readings -- the documented use -- and validation-fold
statistics fed the fit behind the reported metrics.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from models.train import (
    build_estimator,
    final_estimator,
    fitted_scaler,
    prepare_training_matrices,
    _clone_model,
    _make_cv_splitter,
)
from modules.exporter import build_metadata, export_model_bytes
from modules.normalization import build_scaler, normalize_dataset
from pipeline.run_pipeline import run_modeling_stage, train_on_prepared_dataset

TARGET, TIMESTAMP = "reference_pm25", "timestamp"
MODELS = ["ols_regression", "multiple_linear_regression", "ridge", "lasso", "random_forest", "xgboost"]


@pytest.fixture
def scaled_config(config_factory):
    def _make(models=None, method="standard"):
        cfg = config_factory(models or ["ridge"])
        cfg["normalization"]["method"] = method
        return cfg

    return _make


class TestScalerFactory:
    @pytest.mark.parametrize("method", ["standard", "minmax", "robust", "StandardScaler", "MinMaxScaler"])
    def test_supported_methods_build_a_scaler(self, method):
        assert build_scaler(method) is not None

    @pytest.mark.parametrize("method", [None, "none", "", "None"])
    def test_no_scaling_returns_none(self, method):
        assert build_scaler(method) is None

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            build_scaler("quantile")

    def test_scaler_preserves_feature_names(self, merged):
        """The statsmodels wrapper downstream relies on column names surviving."""
        scaler = build_scaler("standard")
        frame = merged[["sensor_pm25_raw", "sensor_humidity"]]
        out = scaler.fit_transform(frame)
        assert isinstance(out, pd.DataFrame)
        assert out.columns.tolist() == frame.columns.tolist()


class TestEstimatorComposition:
    @pytest.mark.parametrize("model_name", MODELS)
    def test_scaling_puts_the_scaler_inside_the_model(self, merged, scaled_config, model_name):
        out = train_on_prepared_dataset(merged, TARGET, scaled_config([model_name]))
        estimator = out["training_results"][model_name].model
        assert isinstance(estimator, Pipeline)
        assert fitted_scaler(estimator) is not None
        assert out["scaler"] is not None

    def test_no_scaling_leaves_a_bare_estimator(self, merged, scaled_config):
        out = train_on_prepared_dataset(merged, TARGET, scaled_config(["ridge"], method="none"))
        assert not isinstance(out["training_results"]["ridge"].model, Pipeline)
        assert out["scaler"] is None

    def test_unwrapping_helpers(self):
        bare = Ridge()
        assert build_estimator(bare, "none") is bare
        assert final_estimator(bare) is bare
        assert fitted_scaler(bare) is None

        wrapped = build_estimator(Ridge(), "standard")
        assert isinstance(wrapped, Pipeline)
        assert isinstance(final_estimator(wrapped), Ridge)


class TestNoLeakage:
    def test_each_fold_fits_its_own_scaler(self, merged):
        features, target, _ = prepare_training_matrices(merged, TARGET, TIMESTAMP)
        splitter = _make_cv_splitter("timeseriessplit", len(features), 5, 42)

        means = []
        for train_idx, _ in splitter.split(features):
            fold = _clone_model(build_estimator(Ridge(), "standard"))
            fold.fit(features.iloc[train_idx], target.iloc[train_idx])
            means.append(fitted_scaler(fold).mean_[0])

        assert len(set(np.round(means, 9))) > 1, "folds shared a single scaler"
        full_mean = features.iloc[:, 0].mean()
        assert not any(np.isclose(m, full_mean) for m in means[:-1]), "a fold used full-data statistics"

    def test_cloning_a_pipeline_yields_an_unfitted_scaler(self):
        fitted = build_estimator(Ridge(), "standard")
        fitted.fit(pd.DataFrame({"a": [1.0, 2, 3]}), pd.Series([1.0, 2, 3]))
        assert hasattr(fitted_scaler(fitted), "mean_")
        assert not hasattr(fitted_scaler(_clone_model(fitted)), "mean_")


class TestExportedModelIsUsable:
    def test_pickle_predicts_from_raw_unscaled_features(self, merged, scaled_config):
        out = train_on_prepared_dataset(merged, TARGET, scaled_config(["ridge"]))
        result = out["training_results"]["ridge"]
        raw = merged[result.feature_names]

        loaded = pickle.loads(export_model_bytes(result.model))
        assert np.allclose(loaded.predict(raw), result.model.predict(raw))

    def test_a_bare_model_on_raw_data_would_have_been_badly_wrong(self, merged, scaled_config):
        """What the old export produced: an estimator with no scaler attached."""
        out = train_on_prepared_dataset(merged, TARGET, scaled_config(["ridge"]))
        result = out["training_results"]["ridge"]
        raw = merged[result.feature_names]

        correct = result.model.predict(raw)
        without_scaler = final_estimator(result.model).predict(raw)
        reference_range = merged[TARGET].max() - merged[TARGET].min()
        assert np.abs(without_scaler - correct).max() > reference_range

    def test_metadata_states_the_input_contract(self, merged, scaled_config):
        cfg = scaled_config(["ridge"])
        out = train_on_prepared_dataset(merged, TARGET, cfg)
        result = out["training_results"]["ridge"]

        meta = build_metadata("ridge", result.feature_names, result.metrics, cfg)
        assert meta["model_pickle_contains_scaler"] is True
        assert meta["pipeline_config"]["normalization_method"] == "standard"
        assert "Raw (unscaled)" in meta["model_input_expectation"]

    def test_metadata_without_scaling(self, merged, scaled_config):
        cfg = scaled_config(["ridge"], method="none")
        meta = build_metadata("ridge", ["sensor_pm25_raw"], {"rmse": 1.0}, cfg)
        assert meta["model_pickle_contains_scaler"] is False


class TestDiagnosticsThroughThePipeline:
    def test_ols_coefficients_survive_wrapping(self, merged, scaled_config):
        out = train_on_prepared_dataset(merged, TARGET, scaled_config(["ols_regression"]))
        result = out["training_results"]["ols_regression"]
        assert result.coefficient_table is not None and not result.coefficient_table.empty
        assert result.p_values and result.standard_errors
        named = list(result.coefficient_table["Variable"])
        assert named[0] == "const"
        assert named[1:] == result.feature_names

    def test_tree_importances_survive_wrapping(self, merged, scaled_config):
        out = train_on_prepared_dataset(merged, TARGET, scaled_config(["random_forest"]))
        result = out["training_results"]["random_forest"]
        assert result.feature_importance
        assert len(result.feature_importance) == len(result.feature_names)


class TestPreviewMatchesReality:
    def test_preview_lists_exactly_the_scaled_columns(self, merged, scaled_config):
        cfg = scaled_config(["ridge"])
        cfg["feature_engineering"]["add_time_features"] = True
        cfg["feature_engineering"]["time_feature_flags"] = {
            "hour_of_day": True, "unix_timestamp": True, "julian_date": True,
        }
        out = run_modeling_stage(merged, cfg)
        previewed = list(out["normalization_outputs"]["summary"]["column"])
        actually_scaled = list(fitted_scaler(out["best_model"]).feature_names_in_)
        assert previewed == actually_scaled

    def test_time_features_are_scaled_like_any_other_predictor(self, merged, scaled_config):
        cfg = scaled_config(["ridge"])
        cfg["feature_engineering"]["add_time_features"] = True
        cfg["feature_engineering"]["time_feature_flags"] = {"unix_timestamp": True, "calendar_date": True}
        out = run_modeling_stage(merged, cfg)

        summary = out["normalization_outputs"]["summary"].set_index("column")
        for column in ("unix_timestamp", "calendar_date"):
            assert abs(summary.loc[column, "after_mean"]) < 1e-6
            assert abs(summary.loc[column, "after_std"] - 1) < 0.05
        assert summary.loc["unix_timestamp", "before_mean"] > 1e6

    def test_no_scaling_produces_no_preview(self, merged, scaled_config):
        out = run_modeling_stage(merged, scaled_config(["ridge"], method="none"))
        assert out["normalization_outputs"] is None
        assert out["scaler"] is None


class TestStandaloneNormalizeDataset:
    def test_only_requested_numeric_columns_change(self, merged_copy):
        merged_copy["site"] = "A"
        out = normalize_dataset(merged_copy, ["sensor_pm25_raw", "site"], "standard")
        assert abs(out["sensor_pm25_raw"].mean()) < 1e-9
        assert (out["site"] == "A").all()
        assert np.allclose(out[TARGET], merged_copy[TARGET])

    def test_returns_the_scaler_on_request(self, merged):
        _, scaler = normalize_dataset(merged, ["sensor_pm25_raw"], "standard", return_scaler=True)
        assert hasattr(scaler, "mean_")
