# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Air Quality Sensor Calibration Lab — a research-grade Streamlit application that calibrates
low-cost sensor (LCS) measurements against reference-grade data using a reproducible,
config-driven ML pipeline. Pure Python; no compiled components.

## Commands

Windows environment; the shell used by tooling here is bash (use `/` paths, `/dev/null`, etc.).

```bash
# Setup (PowerShell venv already exists at .venv / .venv-1)
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell activation
pip install -r requirements.txt
pip install xgboost                 # optional — gracefully skipped if absent

# Run the Streamlit UI (primary entry point)
streamlit run ui/app.py
streamlit run ui/app.py --server.port 8502        # alternate port

# Run the full pipeline non-interactively
python pipeline/run_pipeline.py
```

Tests are pytest, under `tests/`, with fixtures in `tests/conftest.py`:

```bash
pytest                    # full suite
pytest -m "not slow"      # skip multi-model pipeline runs
pytest -m realdata        # only the 2025 co-location tests
```

Two markers matter. `slow` covers full-pipeline runs. `realdata` needs
`data/processed_2025_trends/`, which is gitignored, so those tests skip when it is
absent — they exist because the bundled sample data is synthetic and tidy, while the
real file carries text columns, all-NaN channels, duplicate timestamps across
co-located devices, and derived columns that encode the target.

No linter is configured. `COMMANDS.md` has the full command reference.

## Architecture

The system has two layers that share the same staged functions: a **non-interactive
pipeline** (`pipeline/run_pipeline.py`) and an **interactive Streamlit UI** (`ui/app.py`).
Both call the same stage functions, so business logic lives in `pipeline/` + `modules/` +
`models/` + `evaluation/`, never in the UI.

### Config-driven design
Everything is driven by a single config dict loaded from `config/default.yaml`
(`load_config` accepts YAML or JSON). The dict is threaded through every stage; there is no
global state. Top-level keys: `app`, `data`, `preprocessing` (with independent `reference`
and `sensor` sub-configs), `alignment`, `feature_engineering`, `normalization`, `training`
(includes `selected_models`, `model_params`, `validation_method`, optional `tuning`),
`evaluation`. When adding a feature, add its knobs here first.

### Pipeline stages (`pipeline/run_pipeline.py`)
Discrete functions, each taking the config dict, runnable independently or via
`run_full_pipeline`:
1. `load_input_data` → `modules/data_loader.py` (CSV parse, timestamp/timezone normalization)
2. `run_preprocessing_stage` → `modules/preprocessing.py` (reference & sensor cleaned with
   separate settings; reference outlier removal off by default to keep target in native units)
3. `run_alignment_stage` → `modules/alignment.py` (resample, cross-correlation lag detection, merge)
4. `run_eda_stage` → `modules/eda.py`
5. `run_modeling_stage` → `modules/feature_engineering.py` + `modules/normalization.py` +
   `models/train.py` + `evaluation/comparator.py` (engineer features, optional scaling, train,
   rank leaderboard, produce calibrated dataset)
6. `build_export_bundle` → `modules/exporter.py` (CSV, model `.pkl`, metrics JSON, config
   YAML/JSON, metadata JSON, optional PDF report)

Note the target column naming convention: after alignment the reference target is prefixed,
so modelling uses `f"{reference_prefix}_{target_column}"` (e.g. `reference_pm25`).

### Models (`models/`)
- `model_registry.py` — registry of sklearn regressors (Linear, Ridge, Lasso, RandomForest,
  XGBoost). Grouped for UI via `MODEL_GROUPS`/`MODEL_DISPLAY_NAMES`/`MODEL_INFO`. XGBoost import
  is optional. `get_selected_models` applies `model_params` overrides and coerces JSON-stringified
  numbers back to int/float.
- `train.py` — the core training engine. Key things to know:
  - **OLS regression is special**: `ols_regression` and `multiple_linear_regression` are routed
    to `StatsmodelsOLSRegressor` (a sklearn-like wrapper over statsmodels) to get p-values,
    std errors, t-stats. The registry entry is just a placeholder; the real swap happens in
    `train_models`.
  - **Chronological, leakage-safe splitting** (`split_train_test_by_time`) — never random shuffle on the time split.
  - Three validation methods normalized via `_canonical_validation_method`: `timeseriessplit`
    (default), `kfold`, `holdout`. Metrics computed on out-of-fold predictions (or holdout test set).
  - Optional `RandomizedSearchCV` tuning per model via `tune_hyperparameters` (grids in `_PARAM_GRIDS`).
  - Returns `List[TrainingResult]` — the central dataclass carrying model, metrics,
    predictions, feature importance/coefficients, OLS diagnostics, residuals.

### Evaluation (`evaluation/`)
`metrics.py` computes the calibration metric suite (RMSE, MAE, R², MAPE, Bias, Pearson r,
slope, intercept). `comparator.py` builds the ranked leaderboard and selects the best model
(sorted per `config["evaluation"]`, default by RMSE ascending).

### Streamlit UI (`ui/app.py`)
Single large module — the only entry point. Patterns to follow when editing:
- **13-step wizard** defined by `STEPS` / `STEP_KEYS` (Upload → Preprocessing → Alignment →
  EDA → Variable Selection → Feature Engineering → Normalization → Modelling →
  Validation & Results → Statistical Diagnostics → Residual Analysis → Export → README).
  (The README.md still says "8 steps" — the UI has since expanded.)
- State lives in `st.session_state`, initialized in `init_state()`. Each step has a
  `render_*` function and clears downstream outputs when an upstream input changes.
- **Caching**: `cached_*` wrappers (`@st.cache_data`) re-key on a serialized config **string**
  (`_cfg_to_json` / config text), because dicts aren't hashable. When changing what a stage
  consumes, make sure the relevant `cached_*` signature includes it so the cache invalidates.
- Plotting helpers live in `modules/plots.py`; download
  buttons in `modules/download_helpers.py`; diagnostics (VIF, Shapiro-Wilk, coefficient
  tables) in `modules/diagnostics.py`. Keep computation out of the UI.

### Reproducibility
`random_state` (default 42, from `config["app"]`) flows into the registry, tuning, and CV.
Exports embed provenance metadata. Keep the seed plumbing intact when touching training.
