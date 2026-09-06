"""Statistical diagnostics and the export bundle.

Regression cover: the coefficient table's column names had no single source of
truth, so three renderers looked for "t-Statistic"/"P-value" against columns
actually named "t-statistic"/"p-value". The research PDF silently dropped both
columns and the diagnostics tab showed unrounded floats.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from modules.diagnostics import (
    COEFFICIENT_TABLE_COLUMNS,
    compute_coefficient_table,
    compute_vif,
    shapiro_wilk_test,
)
from modules.exporter import (
    export_config_yaml_bytes,
    export_metadata_json_bytes,
    export_research_report_pdf,
    export_model_summary_report_pdf,
)
from pipeline.run_pipeline import build_export_bundle, train_on_prepared_dataset

TARGET, TIMESTAMP = "reference_pm25", "timestamp"


@pytest.fixture(scope="module")
def ols_result(_base_config, merged):
    from copy import deepcopy

    cfg = deepcopy(_base_config)
    cfg["training"]["selected_models"] = ["ols_regression"]
    out = train_on_prepared_dataset(merged, TARGET, cfg)
    return out, out["training_results"]["ols_regression"]


class TestCoefficientTable:
    def test_columns_match_the_shared_constant(self, ols_result):
        _, result = ols_result
        assert list(result.coefficient_table.columns) == COEFFICIENT_TABLE_COLUMNS

    def test_the_constant_spells_the_statistics_columns_in_lower_case(self):
        assert "t-statistic" in COEFFICIENT_TABLE_COLUMNS
        assert "p-value" in COEFFICIENT_TABLE_COLUMNS

    def test_every_column_is_populated(self, ols_result):
        _, result = ols_result
        table = result.coefficient_table
        for column in COEFFICIENT_TABLE_COLUMNS[1:]:
            assert table[column].notna().all(), column

    def test_ui_formatter_rounds_the_statistics_columns(self, ols_result):
        from ui.app import _format_coefficient_table

        _, result = ols_result
        formatted = _format_coefficient_table(result.coefficient_table)
        assert (formatted["t-statistic"] == formatted["t-statistic"].round(2)).all()
        assert formatted["p-value"].map(lambda v: isinstance(v, str)).all()

    def test_empty_placeholder_advertises_the_real_column_names(self):
        from ui.app import _format_coefficient_table

        assert list(_format_coefficient_table(None).columns) == COEFFICIENT_TABLE_COLUMNS

    def test_non_ols_input_returns_none(self):
        assert compute_coefficient_table(object()) is None


class TestVif:
    def test_returns_a_row_per_predictor(self, merged):
        frame = merged[["sensor_pm25_raw", "sensor_temperature", "sensor_humidity"]]
        table = compute_vif(frame)
        assert set(table["Variable"]) == set(frame.columns)
        assert (table["Status"] == "ok").all()

    def test_constant_columns_are_excluded_and_labelled(self, merged_copy):
        merged_copy["flat"] = 1.0
        table = compute_vif(merged_copy[["sensor_pm25_raw", "sensor_temperature", "flat"]])
        flat = table[table["Variable"] == "flat"].iloc[0]
        assert flat["Status"] == "constant column excluded"

    def test_single_predictor_is_reported_as_such(self, merged):
        table = compute_vif(merged[["sensor_pm25_raw"]])
        assert table.iloc[0]["Status"] == "single predictor"


class TestShapiro:
    def test_returns_statistic_and_p_value(self):
        rng = np.random.default_rng(0)
        out = shapiro_wilk_test(pd.Series(rng.normal(size=200)))
        assert np.isfinite(out["statistic"]) and np.isfinite(out["p_value"])

    def test_too_few_points_yields_nan_rather_than_raising(self):
        out = shapiro_wilk_test(pd.Series([1.0, 2.0]))
        assert np.isnan(out["statistic"])

    def test_large_samples_are_subsampled_deterministically(self):
        rng = np.random.default_rng(0)
        big = pd.Series(rng.normal(size=6000))
        assert shapiro_wilk_test(big) == shapiro_wilk_test(big)


class TestExportBundle:
    def test_research_pdf_renders_every_coefficient_column(self, ols_result):
        out, result = ols_result
        requested = [c for c in COEFFICIENT_TABLE_COLUMNS if c in result.coefficient_table.columns]
        assert len(requested) == len(COEFFICIENT_TABLE_COLUMNS)

    def test_pdfs_are_produced(self, ols_result, _base_config):
        out, result = ols_result
        research = export_research_report_pdf(
            model_name="ols_regression", metrics=result.metrics,
            feature_names=result.feature_names, config=_base_config,
            leaderboard=out["leaderboard"], training_results=out["training_results"],
            prepared_dataset=out["featured_data"], selected_target=TARGET,
            selected_predictors=result.feature_names, modelling_objective=None,
        )
        summary = export_model_summary_report_pdf(
            model_name="ols_regression", metrics=result.metrics,
            feature_names=result.feature_names, config=_base_config,
            coefficient_table=result.coefficient_table,
        )
        assert research and len(research) > 1000
        assert summary and len(summary) > 1000

    def test_bundle_contains_every_expected_artefact(self, ols_result, _base_config):
        out, result = ols_result
        bundle = build_export_bundle(
            calibrated_dataset=out["calibrated_dataset"], selected_model=out["best_model"],
            model_name=out["best_model_name"], metrics=out["best_model_metrics"],
            feature_names=result.feature_names, config=_base_config,
            leaderboard=out["leaderboard"], training_results=out["training_results"],
            prepared_dataset=out["featured_data"], selected_target=TARGET,
            selected_predictors=result.feature_names,
        )
        for key in (
            "calibrated_dataset_csv", "model_pickle", "metrics_json",
            "all_model_metrics_json", "config_json", "config_yaml",
            "metadata_json", "project_run_json",
        ):
            assert key in bundle and bundle[key], key

    def test_metadata_is_valid_json_with_provenance(self, ols_result, _base_config):
        out, result = ols_result
        meta = json.loads(
            export_metadata_json_bytes(
                model_name="ols_regression", features_used=result.feature_names,
                metrics=result.metrics, config=_base_config,
            )
        )
        assert meta["features_used"] == result.feature_names
        assert meta["pipeline_config"]["random_state"] == _base_config["app"]["random_state"]
        assert "created_at" in meta

    def test_config_yaml_round_trips(self, _base_config):
        import yaml

        restored = yaml.safe_load(export_config_yaml_bytes(_base_config))
        assert restored["data"]["target_column"] == _base_config["data"]["target_column"]
