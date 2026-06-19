# v1.0 — Multi-Model Basic Pipeline

**Date**: pre-2026-04-20  
**Status**: Superseded by v2.0

---

## Goal
Extend the v0 prototype with multiple models, structured pipeline stages, and basic evaluation metrics.

## Features Added over v0

### Models
- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression

### Preprocessing
- Forward-fill missing values
- Basic outlier removal

### Alignment
- Fixed-frequency resample (e.g., hourly)
- Inner join merge on timestamp

### Evaluation
- RMSE, MAE, R²
- Side-by-side model comparison table

### UI
- Streamlit tabs per model
- Upload + download CSV
- Basic configuration sliders

## Architecture
```
app.py              — Streamlit UI
preprocessing.py    — Basic cleaning
models.py           — LR, Ridge, Lasso
metrics.py          — RMSE, MAE, R²
```

## Limitations
- No cross-validation
- No lag correction
- No feature engineering
- No drift analysis
- No model explainability
- No YAML/JSON config system
- No Zenodo export
