# Commands Reference — Air Quality Sensor Calibration Lab

Quick reference for all commands used to set up, run, and maintain the project.

---

## 📦 Environment Setup

### Create Virtual Environment

```powershell
# Create venv
python -m venv .venv
```

### Activate Virtual Environment

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\.venv\Scripts\activate.bat
```

### Deactivate Virtual Environment

```powershell
deactivate
```

---

## 📥 Install Dependencies

### Install all requirements

```powershell
pip install -r requirements.txt
```

### Install a specific package

```powershell
pip install <package-name>
```

### Install XGBoost (optional)

```powershell
pip install xgboost
```

### Freeze current packages to requirements.txt

```powershell
pip freeze > requirements.txt
```

### List installed packages

```powershell
pip list
```

---

## 🚀 Run the App

### Launch Streamlit UI

```powershell
streamlit run ui/app.py
```

### Launch on a specific port

```powershell
streamlit run ui/app.py --server.port 8502
```

### Launch without opening the browser automatically

```powershell
streamlit run ui/app.py --server.headless true
```

---

## 🔬 Run the Pipeline (CLI)

### Run the full pipeline directly

```powershell
python pipeline/run_pipeline.py
```

---

## 🗂️ Project Navigation

### List project structure

```powershell
ls
```

### View a specific module

```powershell
cat modules\data_loader.py
cat modules\preprocessing.py
cat modules\alignment.py
cat modules\eda.py
cat modules\feature_engineering.py
cat modules\drift_analysis.py
cat modules\exporter.py
```

### View models

```powershell
cat models\model_registry.py
cat models\train.py
cat models\predict.py
```

### View evaluation

```powershell
cat evaluation\metrics.py
cat evaluation\comparator.py
```

### View config

```powershell
cat config\default.yaml
```

---

## 🧪 Testing & Debugging

### Run a specific Python script

```powershell
python modules\feature_engineering.py
python models\train.py
```

### Check Python version

```powershell
python --version
```

### Check pip version

```powershell
pip --version
```

### Check if a package is installed

```powershell
pip show streamlit
pip show xgboost
pip show scikit-learn
```

---

## 📁 Git Commands

### Initialize repository

```powershell
git init
```

### Check status

```powershell
git status
```

### Stage all changes

```powershell
git add .
```

### Commit changes

```powershell
git commit -m "your message here"
```

### View commit history

```powershell
git log --oneline -10
```

### Create a new branch

```powershell
git checkout -b feature/your-feature-name
```

### Switch branches

```powershell
git checkout main
```

### Push to remote

```powershell
git push origin main
```

### Pull latest changes

```powershell
git pull origin main
```

---

## 🔧 Utility Commands

### Check running Streamlit processes

```powershell
Get-Process | Where-Object { $_.Name -like "*python*" }
```

### Kill a process by port (e.g., 8501)

```powershell
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Clear Python cache files

```powershell
Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
```

---

## 📄 Sample Data

Located in `sample_data/`:

| File | Description |
|------|-------------|
| `reference_dataset.csv` | 7-day reference-grade demo dataset (168 rows) |
| `low_cost_sensor_dataset.csv` | 7-day LCS demo dataset (168 rows) |

---

## 🌐 Default Streamlit URL

Once the app is running, open in browser:

```
http://localhost:8501
```

---

## 📝 Notes

- Always activate the virtual environment before running any commands.
- `xgboost` is optional — the app runs without it.
- The config file is at `config/default.yaml` and controls all pipeline defaults.
- Exports (CSV, PKL, JSON, YAML) are generated in the UI's Export step.
