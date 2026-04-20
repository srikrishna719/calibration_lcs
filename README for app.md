# 🌍 Air Quality Sensor Calibration Lab

## 📌 Overview

This project is a modular, research-grade calibration framework for aligning and calibrating low-cost air quality sensors against reference-grade instruments.

The application enables researchers to:

* Upload reference and sensor datasets
* Perform automated preprocessing and timestamp alignment
* Train and compare multiple machine learning models
* Evaluate calibration performance using standard metrics
* Export reproducible results for research (Zenodo-ready)

---

## 🎯 Objectives

* Provide an end-to-end calibration pipeline
* Ensure reproducibility via config-driven workflows
* Benchmark multiple ML models transparently
* Support research and publication use-cases

---

## 🧱 Architecture

```
app/
│
├── ui/                  # Streamlit UI
├── pipeline/            # Pipeline orchestrator
├── modules/             # Data processing modules
├── models/              # ML models + registry
├── evaluation/          # Metrics and comparison
├── config/              # YAML/JSON configs
├── exports/             # Output artifacts
└── utils/               # Helper functions
```

---

## ⚙️ Pipeline Flow

1. Data Ingestion
2. Preprocessing
3. Timestamp Alignment
4. Feature Engineering
5. Model Training
6. Evaluation & Comparison
7. Export (Zenodo-ready)

---

## 📊 Supported Models

* Linear Regression
* Ridge / Lasso
* Random Forest
* XGBoost / LightGBM
* Support Vector Regression

---

## 📈 Evaluation Metrics

* RMSE
* MAE
* R²
* MAPE

---

## 📦 Outputs

* Calibrated dataset (CSV)
* Model files (.pkl)
* Metrics (JSON)
* Config file (for reproducibility)

---

## 🧪 Research Contributions

* Automated timestamp alignment with lag detection
* Config-driven reproducible ML pipeline
* Multi-model benchmarking for calibration
* Modular architecture for extensibility

---

## 🚀 Getting Started

```bash
git clone https://github.com/srikrishna719/calibration_lcs.git
cd calibration_lcs
pip install -r requirements.txt
streamlit run ui/app.py
```

---

## 📌 Future Enhancements

* AutoML (Optuna)
* Drift detection for sensor degradation
* Edge deployment (Raspberry Pi)
* Transfer learning across locations

---

## 📄 License

MIT License

---
