# v0.0 — Original Prototype

**Date**: pre-2026-04-20  
**Status**: Superseded by v1.0

---

## Goal
Proof-of-concept single-model Streamlit app for calibrating one low-cost sensor CSV against one reference CSV using Linear Regression.

## Features
- Upload reference CSV and sensor CSV via Streamlit file uploader
- Parse timestamps, merge datasets on timestamp
- Train a single Linear Regression model
- Compute RMSE
- Display prediction vs actual scatter plot
- Download calibrated output as CSV

## Architecture (single-file)
```
app.py   — monolithic single-file Streamlit app
```

## Limitations
- No preprocessing (missing values, outliers)
- No time alignment or lag correction
- No cross-validation
- No model selection
- Single metric only (RMSE)
- No configurable parameters
- No export beyond calibrated CSV
