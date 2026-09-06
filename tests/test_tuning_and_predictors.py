"""Hyperparameter search and which columns are allowed to be predictors.

Regression cover for two defects:

* tuning searched once on the first 80% of rows, then metrics were computed by
  cross-validating over all rows, so parameters were scored on the same data
  that selected them;
* reference-instrument columns were offered -- and pre-selected -- as
  predictors, producing models that cannot be applied to a deployed sensor.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from evaluation.metrics import calculate_regression_metrics
from models.train import build_estimator, generate_validation_predictions, prepare_training_matrices
from models.model_registry import build_model_registry
from pipeline.run_pipeline import (
    deployable_feature_subset,
    run_modeling_stage,
    train_on_prepared_dataset,
)

TARGET, TIMESTAMP = "reference_pm25", "timestamp"


@pytest.fixture
def tuned_config(config_factory):
    def _make(models=("ridge",), n_iter=6, **overrides):
        cfg = config_factory(list(models), **overrides)
        cfg["training"]["tuning"] = {m: {"enabled": True, "n_iter": n_iter} for m in models}
        return cfg

    return _make


class TestNestedTuning:
    def test_a_search_runs_inside_every_outer_fold(self, merged, tuned_config):
        out = train_on_prepared_dataset(merged, TARGET, tuned_config())
        result = out["training_results"]["ridge"]
        folds = int(tuned_config()["training"]["cross_validation_folds"])
        assert result.nested_best_params is not None
        assert len(result.nested_best_params) == folds

    def test_folds_may_disagree_and_that_is_recorded(self, merged, tuned_config):
        out = train_on_prepared_dataset(merged, TARGET, tuned_config())
        chosen = [f["alpha"] for f in out["training_results"]["ridge"].nested_best_params]
        assert len(set(chosen)) > 1, (
            "folds all agreed, so this dataset cannot demonstrate the nesting; "
            "the single up-front search was invalid precisely because they differ"
        )

    def test_final_model_is_searched_over_all_rows(self, merged, tuned_config):
        out = train_on_prepared_dataset(merged, TARGET, tuned_config())
        result = out["training_results"]["ridge"]
        assert result.best_params and "alpha" in result.best_params

    def test_reported_params_are_not_pipeline_prefixed(self, merged, tuned_config):
        cfg = tuned_config()
        cfg["normalization"]["method"] = "standard"
        out = train_on_prepared_dataset(merged, TARGET, cfg)
        result = out["training_results"]["ridge"]
        assert isinstance(result.model, Pipeline)
        assert not any(k.startswith("model__") for k in result.best_params)
        assert all(not any(k.startswith("model__") for k in fold) for fold in result.nested_best_params)

    def test_nested_predictions_differ_from_the_leaky_procedure(self, merged, tuned_config):
        """The old path tuned once on the first 80%, then scored every fold."""
        from models.train import tune_hyperparameters

        features, target, timestamps = prepare_training_matrices(merged, TARGET, TIMESTAMP)
        split = max(1, int(len(features) * 0.8))

        leaky_model, _ = tune_hyperparameters(
            "ridge", build_estimator(build_model_registry()["ridge"], "none"),
            features.iloc[:split], target.iloc[:split], 6, 5, 42, "timeseriessplit",
        )
        leaky_preds, leaky_folds = generate_validation_predictions(
            leaky_model, features, target, timestamps, 5, "timeseriessplit", 42
        )
        assert leaky_folds == [], "no nesting requested, so no per-fold params"

        nested_preds, nested_folds = generate_validation_predictions(
            build_estimator(build_model_registry()["ridge"], "none"),
            features, target, timestamps, 5, "timeseriessplit", 42,
            tuning={"model_name": "ridge", "n_iter": 6},
        )
        assert len(nested_folds) == 5
        leaky = calculate_regression_metrics(leaky_preds["actual"], leaky_preds["predicted"])["rmse"]
        nested = calculate_regression_metrics(nested_preds["actual"], nested_preds["predicted"])["rmse"]
        assert not np.isclose(leaky, nested), "nesting made no difference; the test is not discriminating"

    def test_holdout_tunes_once_and_does_not_nest(self, merged, tuned_config):
        out = train_on_prepared_dataset(merged, TARGET, tuned_config(validation_method="holdout"))
        result = out["training_results"]["ridge"]
        assert result.best_params and "alpha" in result.best_params
        assert result.nested_best_params is None

    def test_untuned_runs_record_nothing(self, merged, config_factory):
        out = train_on_prepared_dataset(merged, TARGET, config_factory(["ridge"]))
        result = out["training_results"]["ridge"]
        assert result.best_params is None
        assert result.nested_best_params is None

    def test_models_without_a_grid_are_left_alone(self, merged, tuned_config):
        out = train_on_prepared_dataset(merged, TARGET, tuned_config(models=("ols_regression",)))
        result = out["training_results"]["ols_regression"]
        assert result.best_params is None
        assert result.coefficient_table is not None and not result.coefficient_table.empty


class TestDeployablePredictors:
    def test_reference_columns_are_excluded_by_default(self, merged, config):
        subset = deployable_feature_subset(merged, TIMESTAMP, TARGET, config)
        assert subset
        assert not any(c.startswith("reference_") for c in subset)
        assert set(subset) == {
            "sensor_pm25_raw", "sensor_temperature", "sensor_humidity", "sensor_voc",
        }

    def test_opt_in_restores_them(self, merged, config):
        config["training"]["include_reference_predictors"] = True
        assert deployable_feature_subset(merged, TIMESTAMP, TARGET, config) is None

    def test_training_never_sees_reference_columns_by_default(self, merged, config_factory):
        out = run_modeling_stage(merged, config_factory(["ridge"]))
        features = out["training_results"]["ridge"].feature_names
        assert not any(c.startswith("reference_") for c in features)

    def test_reference_columns_flatter_the_metrics(self, merged, config_factory):
        excluded = run_modeling_stage(merged, config_factory(["ridge"]))
        cfg = config_factory(["ridge"])
        cfg["training"]["include_reference_predictors"] = True
        included = run_modeling_stage(merged, cfg)

        assert (
            included["training_results"]["ridge"].metrics["rmse"]
            < excluded["training_results"]["ridge"].metrics["rmse"]
        ), "reference columns should look better; that is exactly why they mislead"

    def test_an_explicit_subset_from_the_caller_wins(self, merged, config_factory):
        out = run_modeling_stage(merged, config_factory(["ridge"]), feature_subset=["sensor_pm25_raw"])
        assert out["training_results"]["ridge"].feature_names == ["sensor_pm25_raw"]

    def test_non_standard_sensor_names_are_still_filtered(self, merged_copy, config):
        renamed = merged_copy.rename(
            columns={c: c.replace("sensor_", "lcs_") for c in merged_copy.columns}
        )
        subset = deployable_feature_subset(renamed, TIMESTAMP, TARGET, config)
        assert "lcs_pm25_raw" in subset
        assert not any(c.startswith("reference_") for c in subset)

    def test_all_reference_frame_falls_back_rather_than_failing(self, merged, config):
        only_reference = merged[[TIMESTAMP, TARGET, "reference_temperature", "reference_humidity"]]
        assert deployable_feature_subset(only_reference, TIMESTAMP, TARGET, config) is None

    def test_demo_workflow_applies_the_same_rule(self, reference_csv, sensor_csv, config_factory):
        import pandas as pd
        from ui.demo_workflow import build_sample_demo_state

        state = build_sample_demo_state(
            pd.read_csv(reference_csv), pd.read_csv(sensor_csv), config_factory(["ridge"])
        )
        assert not any(c.startswith("reference_") for c in state["selected_predictors"])
