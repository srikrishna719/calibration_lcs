"""Streamlit application for air quality sensor calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from models.predict import predict_with_model
from pipeline.run_pipeline import export_model_bytes, load_config, run_calibration_pipeline


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes for download.

    Parameters
    ----------
    dataframe:
        Dataset to serialize.

    Returns
    -------
    bytes
        CSV-encoded bytes.
    """
    return dataframe.to_csv(index=False).encode("utf-8")


def load_uploaded_config(uploaded_file: Any) -> Dict[str, Any]:
    """Load configuration from an uploaded file or fall back to default config.

    Parameters
    ----------
    uploaded_file:
        Uploaded config file from Streamlit.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration.
    """
    if uploaded_file is None:
        return load_config(DEFAULT_CONFIG_PATH)

    suffix = Path(uploaded_file.name).suffix.lower()
    text_content = uploaded_file.getvalue().decode("utf-8")

    if suffix in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(text_content)

    if suffix == ".json":
        return json.loads(text_content)

    raise ValueError("Config upload must be a YAML, YML, or JSON file.")


def render_summary_cards(model_name: str, metrics: Dict[str, Any]) -> None:
    """Render top-level KPI cards for the selected model.

    Parameters
    ----------
    model_name:
        Model name shown in the summary.
    metrics:
        Metric dictionary for the selected model.
    """
    columns = st.columns(4)
    columns[0].metric("Model", model_name)
    columns[1].metric("RMSE", f"{metrics['rmse']:.4f}")
    columns[2].metric("MAE", f"{metrics['mae']:.4f}")
    columns[3].metric("R2", f"{metrics['r2']:.4f}")


def render_prediction_chart(predictions: pd.DataFrame, model_name: str) -> None:
    """Render actual versus predicted comparison chart.

    Parameters
    ----------
    predictions:
        Timestamped actual and predicted values.
    model_name:
        Model name shown in the chart title.
    """
    chart_data = predictions.melt(
        id_vars="timestamp",
        value_vars=["actual", "predicted"],
        var_name="series",
        value_name="value",
    )
    figure = px.line(
        chart_data,
        x="timestamp",
        y="value",
        color="series",
        title=f"{model_name}: Actual vs Predicted",
    )
    st.plotly_chart(figure, use_container_width=True)


def build_calibrated_dataset(results: Dict[str, Any], config: Dict[str, Any], selected_model: str) -> pd.DataFrame:
    """Create a calibrated dataset for the selected model.

    Parameters
    ----------
    results:
        Pipeline output dictionary.
    config:
        Active pipeline configuration.
    selected_model:
        Model name chosen in the UI.

    Returns
    -------
    pd.DataFrame
        Calibrated dataset with reference and predicted values.
    """
    timestamp_column = config["data"]["timestamp_column"]
    reference_target_column = (
        f"{config['data']['reference_prefix']}_{config['data']['target_column']}"
    )
    selected_result = results["training_results"][selected_model]
    predictions = predict_with_model(
        model=selected_result.model,
        dataframe=results["featured_data"],
        target_column=reference_target_column,
        timestamp_column=timestamp_column,
    )
    calibrated_dataset = results["featured_data"][
        [timestamp_column, reference_target_column]
    ].merge(
        predictions,
        on=timestamp_column,
        how="left",
    ).rename(
        columns={
            reference_target_column: "reference_value",
            "prediction": "calibrated_prediction",
        }
    )
    return calibrated_dataset


def main() -> None:
    """Run the Streamlit calibration UI."""
    st.set_page_config(page_title="Air Quality Sensor Calibration Lab", layout="wide")
    st.title("Air Quality Sensor Calibration Lab")
    st.caption("Upload reference-grade and low-cost sensor datasets to build a calibration model.")

    with st.sidebar:
        st.header("Inputs")
        reference_file = st.file_uploader("Reference CSV", type=["csv"])
        sensor_file = st.file_uploader("Sensor CSV", type=["csv"])
        config_file = st.file_uploader("Optional Config", type=["yaml", "yml", "json"])
        run_pipeline_clicked = st.button("Run Calibration", type="primary")

    if not run_pipeline_clicked:
        st.info("Upload both CSV files and run the calibration pipeline.")
        return

    if reference_file is None or sensor_file is None:
        st.error("Both reference and sensor CSV files are required.")
        return

    try:
        config = load_uploaded_config(config_file)
        results = run_calibration_pipeline(reference_file, sensor_file, config)
    except Exception as exc:  # pragma: no cover - UI runtime path
        st.exception(exc)
        return

    st.subheader("Leaderboard")
    st.dataframe(results["leaderboard"], use_container_width=True)

    model_options = results["leaderboard"]["model"].tolist()
    selected_model = st.selectbox(
        "Selected model for export",
        options=model_options,
        index=model_options.index(results["best_model_name"]),
    )
    selected_result = results["training_results"][selected_model]
    selected_calibrated_dataset = build_calibrated_dataset(results, config, selected_model)

    render_summary_cards(selected_model, selected_result.metrics)
    st.write(f"Current export model: `{selected_model}`")
    st.json(selected_result.metrics)

    st.subheader("Alignment Metadata")
    st.json(results["alignment_metadata"])

    st.subheader("Prediction Comparison")
    render_prediction_chart(selected_result.predictions, selected_model)

    st.subheader("Calibrated Dataset Preview")
    st.dataframe(selected_calibrated_dataset.head(50), use_container_width=True)

    st.download_button(
        label="Download Calibrated Dataset",
        data=dataframe_to_csv_bytes(selected_calibrated_dataset),
        file_name="calibrated_dataset.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download Trained Model",
        data=export_model_bytes(selected_result.model),
        file_name=f"{selected_model}_model.pkl",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
