"""End-to-end runs and the UI workflow helpers.

Replaces the old tests/smoke_test.py script: same ground, but assertions the
runner can report individually.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules.plots import create_scatter_with_fit
from pipeline.run_pipeline import build_export_bundle, run_full_pipeline
from ui.demo_workflow import build_sample_demo_state, history_as_dataframe, make_run_history_entry
from ui.workflow import next_step, normalize_current_step, suggest_column_setup, visible_steps

TARGET = "reference_pm25"


class TestWorkflowHelpers:
    def test_column_setup_is_suggested_from_headers(self, reference_csv, sensor_csv):
        ref = pd.read_csv(reference_csv)
        sen = pd.read_csv(sensor_csv)
        timestamp, target = suggest_column_setup(
            ref.columns.tolist(), sen.columns.tolist(),
            configured_timestamp="not_the_timestamp", configured_target="not_the_target",
        )
        assert timestamp == "timestamp"
        assert target == "pm25"

    def test_configured_names_are_kept_when_they_exist(self, reference_csv, sensor_csv):
        ref = pd.read_csv(reference_csv)
        sen = pd.read_csv(sensor_csv)
        assert suggest_column_setup(ref.columns.tolist(), sen.columns.tolist()) == ("timestamp", "pm25")

    @pytest.mark.parametrize(
        "mode,expected",
        [("Basic", ["Upload", "Export"]), ("Advanced", ["Upload", "Diagnostics", "Export"])],
    )
    def test_advanced_only_steps_are_hidden_in_basic_mode(self, mode, expected):
        steps = ["Upload", "Diagnostics", "Export"]
        keys = ["upload", "statistical_diagnostics", "export"]
        assert visible_steps(steps, keys, mode) == expected

    def test_navigation_skips_hidden_steps(self):
        steps = ["Upload", "Diagnostics", "Export"]
        keys = ["upload", "statistical_diagnostics", "export"]
        assert next_step("Upload", steps, keys, "Basic") == "Export"
        assert next_step("Upload", steps, keys, "Advanced") == "Diagnostics"

    def test_a_hidden_current_step_is_normalised_away(self):
        steps = ["Upload", "Diagnostics", "Export"]
        keys = ["upload", "statistical_diagnostics", "export"]
        assert normalize_current_step("Diagnostics", steps, keys, "Basic") == "Upload"


class TestDemoWorkflow:
    @pytest.fixture(scope="class")
    def demo(self, reference_csv, sensor_csv, _base_config):
        from copy import deepcopy

        cfg = deepcopy(_base_config)
        cfg["training"]["selected_models"] = ["ols_regression", "ridge", "random_forest"]
        cfg["training"]["cross_validation_folds"] = 3
        cfg["training"]["model_params"]["random_forest"]["n_estimators"] = 20
        cfg["training"]["model_params"]["random_forest"]["max_depth"] = 6
        return build_sample_demo_state(pd.read_csv(reference_csv), pd.read_csv(sensor_csv), cfg)

    def test_produces_a_ranked_leaderboard_and_calibrated_data(self, demo):
        outputs = demo["modeling_outputs"]
        assert outputs["best_model_name"]
        assert len(outputs["leaderboard"]) == 3
        assert not outputs["calibrated_dataset"].empty
        assert outputs["calibrated_dataset"]["calibrated_value"].notna().all()

    def test_leaderboard_is_sorted_by_rmse(self, demo):
        rmse = demo["modeling_outputs"]["leaderboard"]["rmse"].tolist()
        assert rmse == sorted(rmse)

    def test_every_stage_output_is_present(self, demo):
        for key in (
            "data_outputs", "preprocessing_outputs", "alignment_outputs", "eda_outputs",
            "variable_selection_outputs", "feature_engineering_outputs",
            "normalization_outputs", "modeling_outputs",
        ):
            assert demo[key] is not None, key
        assert demo["normalization_outputs"]["modelling_dataset"].shape[0] > 0

    def test_charts_render_to_png(self, demo):
        outputs = demo["modeling_outputs"]
        best = outputs["training_results"][outputs["best_model_name"]]
        figure = create_scatter_with_fit(best.validation_predictions, model_name=outputs["best_model_name"])
        assert len(figure.to_image(format="png")) > 1000

    def test_run_history_records_the_best_model(self, demo):
        entry = make_run_history_entry(demo["modeling_outputs"], demo["config"], "test run")
        frame = history_as_dataframe([entry])
        assert frame.loc[0, "model_name"] == demo["modeling_outputs"]["best_model_name"]

    def test_export_bundle_builds_from_demo_state(self, demo):
        outputs = demo["modeling_outputs"]
        best = outputs["training_results"][outputs["best_model_name"]]
        bundle = build_export_bundle(
            calibrated_dataset=outputs["calibrated_dataset"], selected_model=outputs["best_model"],
            model_name=outputs["best_model_name"], metrics=outputs["best_model_metrics"],
            feature_names=best.feature_names, config=demo["config"],
            leaderboard=outputs["leaderboard"], training_results=outputs["training_results"],
            prepared_dataset=outputs["featured_data"], selected_target=demo["selected_target"],
            selected_predictors=demo["selected_predictors"],
        )
        assert len(bundle["all_model_metrics_json"]) > 1000
        if "research_report_pdf" in bundle:
            assert len(bundle["research_report_pdf"]) > 1000


@pytest.mark.slow
class TestFullPipeline:
    @pytest.fixture(scope="class")
    def result(self, reference_csv, sensor_csv, _base_config):
        from copy import deepcopy

        cfg = deepcopy(_base_config)
        cfg["training"]["model_params"]["random_forest"]["n_estimators"] = 25
        return run_full_pipeline(reference_csv, sensor_csv, cfg), cfg

    def test_runs_every_stage_and_ranks_all_models(self, result):
        out, cfg = result
        assert len(out["leaderboard"]) == len(cfg["training"]["selected_models"])
        assert out["best_model_name"] in out["training_results"]

    def test_calibrated_output_is_complete(self, result):
        out, _ = result
        assert out["calibrated_dataset"]["calibrated_value"].notna().all()

    def test_no_reference_columns_leak_into_the_model(self, result):
        out, _ = result
        features = out["training_results"][out["best_model_name"]].feature_names
        assert not any(c.startswith("reference_") for c in features)

    def test_metrics_are_finite(self, result):
        out, _ = result
        for name, value in out["best_model_metrics"].items():
            if isinstance(value, float):
                assert np.isfinite(value), name

    def test_export_artefacts_are_all_present(self, result):
        out, _ = result
        produced = {k for k in out if k.endswith(("_csv", "_json", "_yaml", "_pdf", "_pickle"))}
        assert produced >= {
            "calibrated_dataset_csv", "model_pickle", "metrics_json",
            "config_json", "config_yaml", "metadata_json",
        }

    def test_reproducible_across_runs(self, reference_csv, sensor_csv, _base_config):
        from copy import deepcopy

        cfg = deepcopy(_base_config)
        cfg["training"]["selected_models"] = ["ridge", "random_forest"]
        cfg["training"]["model_params"]["random_forest"]["n_estimators"] = 20
        first = run_full_pipeline(reference_csv, sensor_csv, deepcopy(cfg))
        second = run_full_pipeline(reference_csv, sensor_csv, deepcopy(cfg))
        assert first["best_model_name"] == second["best_model_name"]
        assert np.isclose(first["best_model_metrics"]["rmse"], second["best_model_metrics"]["rmse"])
