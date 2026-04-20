# Air Quality Sensor Calibration Lab

Modular Python application for calibrating low-cost air quality sensor measurements against reference-grade data using multiple machine learning models.

## Overview

The application supports:

- Uploading reference and sensor CSV datasets
- Automated preprocessing and timestamp alignment
- Training and comparing multiple ML calibration models
- Exporting calibrated data and trained models
- Reproducible execution through a config-driven pipeline

## Structure

- `modules/`
  - `data_loader.py`: dataset loading and validation
  - `preprocessing.py`: missing values, outlier removal, normalization
  - `alignment.py`: resampling, timestamp alignment, lag detection
  - `feature_engineering.py`: lag features, rolling statistics, time features
- `models/`
  - `model_registry.py`: supported regressors
  - `train.py`: training and evaluation flow
  - `predict.py`: prediction helpers
- `evaluation/`
  - `metrics.py`: RMSE, MAE, R2, MAPE
  - `comparator.py`: leaderboard creation and best-model selection
- `pipeline/`
  - `run_pipeline.py`: end-to-end orchestration
- `ui/`
  - `app.py`: Streamlit UI
- `config/`
  - `default.yaml`: default pipeline configuration

## Expected Input Schema

Both CSV files must contain:

- `timestamp`: parseable datetime column

Reference CSV must also contain:

- target column configured in `config/default.yaml` (`pm25` by default)

Sensor CSV should contain one or more numeric sensor features, for example:

- `pm25_raw`
- `temperature`
- `humidity`
- `voc`

## Supported Models

- Linear Regression
- Multilinear Regression
- Ridge
- Lasso
- Elastic Net
- Random Forest
- Gradient Boosting
- Extra Trees
- XGBoost when installed

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run ui/app.py
```

## Default Flow

1. Upload reference and sensor CSV files.
2. Optionally upload a custom YAML or JSON config.
3. Run calibration.
4. Review the leaderboard.
5. Select a model for export.
6. Download the calibrated dataset and trained model.

## Notes

- `xgboost` is optional at runtime. If unavailable, the rest of the models still work.
- The reference target column is not normalized, so exported calibrated predictions stay in the target's native unit scale.
- Alignment uses resampling plus correlation-based lag detection before merging the datasets.
