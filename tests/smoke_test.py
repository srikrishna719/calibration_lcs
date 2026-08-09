"""Fast smoke test for the CaliSenseAQ Streamlit workflow."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.plots import create_scatter_with_fit
from pipeline.run_pipeline import build_export_bundle, load_config
from ui.demo_workflow import build_sample_demo_state, history_as_dataframe, make_run_history_entry
from ui.workflow import suggest_column_setup, visible_steps


def _smoke_config() -> dict:
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    config = deepcopy(config)
    config["training"]["selected_models"] = ["ols_regression", "ridge", "random_forest"]
    config["training"]["cross_validation_folds"] = 3
    config["training"]["model_params"]["random_forest"]["n_estimators"] = 20
    config["training"]["model_params"]["random_forest"]["max_depth"] = 6
    return config


def main() -> None:
    ref_path = PROJECT_ROOT / "sample_data" / "reference_dataset.csv"
    sen_path = PROJECT_ROOT / "sample_data" / "low_cost_sensor_dataset.csv"
    ref_df = pd.read_csv(ref_path)
    sen_df = pd.read_csv(sen_path)

    ts_col, target_col = suggest_column_setup(
        ref_df.columns.tolist(),
        sen_df.columns.tolist(),
        configured_timestamp="not_the_timestamp",
        configured_target="not_the_target",
    )
    assert ts_col == "timestamp", ts_col
    assert target_col == "pm25", target_col

    basic_steps = visible_steps(["Upload", "Diagnostics", "Export"], ["upload", "statistical_diagnostics", "export"], "Basic")
    advanced_steps = visible_steps(["Upload", "Diagnostics", "Export"], ["upload", "statistical_diagnostics", "export"], "Advanced")
    assert basic_steps == ["Upload", "Export"], basic_steps
    assert advanced_steps == ["Upload", "Diagnostics", "Export"], advanced_steps

    demo_state = build_sample_demo_state(ref_df, sen_df, _smoke_config())
    modeling_outputs = demo_state["modeling_outputs"]
    assert modeling_outputs["best_model_name"]
    assert not modeling_outputs["leaderboard"].empty
    assert not modeling_outputs["calibrated_dataset"].empty
    assert demo_state["normalization_outputs"]["normalized_dataset"].shape[0] > 0

    best = modeling_outputs["training_results"][modeling_outputs["best_model_name"]]
    fig = create_scatter_with_fit(best.validation_predictions, model_name=modeling_outputs["best_model_name"])
    png_bytes = fig.to_image(format="png")
    assert len(png_bytes) > 1_000

    history_entry = make_run_history_entry(modeling_outputs, demo_state["config"], "smoke test")
    history_df = history_as_dataframe([history_entry])
    assert history_df.loc[0, "model_name"] == modeling_outputs["best_model_name"]

    export_bundle = build_export_bundle(
        calibrated_dataset=modeling_outputs["calibrated_dataset"],
        selected_model=modeling_outputs["best_model"],
        model_name=modeling_outputs["best_model_name"],
        metrics=modeling_outputs["best_model_metrics"],
        feature_names=best.feature_names,
        config=demo_state["config"],
        leaderboard=modeling_outputs["leaderboard"],
        training_results=modeling_outputs["training_results"],
        prepared_dataset=modeling_outputs["featured_data"],
        selected_target=demo_state["selected_target"],
        selected_predictors=demo_state["selected_predictors"],
    )
    assert len(export_bundle["all_model_metrics_json"]) > 1_000
    if "research_report_pdf" in export_bundle:
        assert len(export_bundle["research_report_pdf"]) > 1_000

    print("smoke ok")
    print(f"best_model={modeling_outputs['best_model_name']}")
    print(f"leaderboard_rows={len(modeling_outputs['leaderboard'])}")
    print(f"calibrated_rows={len(modeling_outputs['calibrated_dataset'])}")
    print(f"png_bytes={len(png_bytes)}")


if __name__ == "__main__":
    main()
