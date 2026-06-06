# 📖 PROJECT DOCUMENTATION INDEX

## 🎯 START HERE

### For First-Time Users

1. **[QUICK_START.md](QUICK_START.md)** - 5-minute setup guide
   - One-command execution
   - Troubleshooting tips
   - File structure overview

### For Detailed Understanding

2. **[README.md](README.md)** - Complete project documentation
   - What, why, and how of the project
   - Technology stack with integration details
   - Installation, data pipeline, model performance
   - Configuration and customization

---

## 🚀 EXECUTION METHODS

### Fastest (Automated)

```bash
python run.py
```

✅ Single command runs all 4 stages  
✅ Auto-launches dashboard  
✅ Colored progress output  
✅ Error handling & recovery

### Windows (Double-Click)

```
run.bat
```

✅ No terminal knowledge needed  
✅ Auto-detects Python  
✅ Logs output

### Linux/macOS

```bash
./run.sh
```

✅ Bash automation  
✅ Dependency checking  
✅ Python3 compatibility

### Manual Stage-by-Stage

```bash
# Stage 1: Generate data
python src/data_ingestion.py

# Stage 2: Create features
python src/feature_engineering.py

# Stage 3: Train models
python src/train.py

# Stage 4: Launch dashboard
streamlit run src/dashboard.py
```

---

## 📁 DIRECTORY STRUCTURE

```
c:\projects\EV\
│
├── EXECUTION FILES (Use these!)
│   ├── run.py              ← Main Python runner
│   ├── run.bat             ← Windows batch runner
│   └── run.sh              ← Linux/macOS runner
│
├── DOCUMENTATION
│   ├── README.md           ← Full technical docs (15 sections)
│   ├── QUICK_START.md      ← Quick reference (this file!)
│   └── PROJECT_INDEX.md    ← Documentation map
│
├── CONFIGURATION
│   └── requirements.txt    ← Python dependencies (13 packages)
│
├── SOURCE CODE (Python modules)
│   └── src/
│       ├── data_ingestion.py       (500+ lines) - Weather + demand simulation
│       ├── feature_engineering.py  (300+ lines) - ML feature creation
│       ├── train.py                (400+ lines) - XGBoost + Prophet training
│       └── dashboard.py            (600+ lines) - Streamlit web interface
│
├── GENERATED DATA
│   └── data/
│       ├── raw_demand_data.csv           (438.6K rows, 16 columns)
│       ├── processed_demand_data.csv     (434.4K rows, 75 features)
│       └── predictions_test_set.csv      (86.9K rows, 12 columns)
│
├── TRAINED MODELS
│   └── models/
│       ├── xgboost_model.json            (1.2 MB - Primary model)
│       ├── prophet_Bengaluru_*.json      (25 KB each - 25 total)
│       └── [25 zone-specific Prophet models]
│
└── EVALUATION REPORTS
    └── reports/
        ├── metrics.csv                   (2 models × 4 metrics)
        ├── cross_validation_metrics.csv  (3 folds × 4 metrics)
        └── shap_summary.png              (Feature importance visualization)
```

---

## 📊 WHAT EACH FILE DOES

### Execution & Configuration

| File               | Purpose              | User Action                       |
| ------------------ | -------------------- | --------------------------------- |
| `run.py`           | Main orchestrator    | `python run.py`                   |
| `run.bat`          | Windows launcher     | Double-click or `run.bat`         |
| `run.sh`           | Linux/macOS launcher | `./run.sh`                        |
| `requirements.txt` | Dependencies list    | `pip install -r requirements.txt` |

### Documentation

| File               | Content                  | Read Time |
| ------------------ | ------------------------ | --------- |
| `README.md`        | Everything (15 sections) | 20-30 min |
| `QUICK_START.md`   | Quick reference          | 5 min     |
| `PROJECT_INDEX.md` | This file (navigation)   | 3 min     |

### Source Code

| File                     | Lines | Purpose                         | Key Output                  |
| ------------------------ | ----- | ------------------------------- | --------------------------- |
| `data_ingestion.py`      | 500+  | Weather API + demand simulation | `raw_demand_data.csv`       |
| `feature_engineering.py` | 300+  | Lag/rolling/cyclical features   | `processed_demand_data.csv` |
| `train.py`               | 400+  | XGBoost + 25 Prophet models     | Models + metrics            |
| `dashboard.py`           | 600+  | Interactive Streamlit interface | Web app @ :8501             |

### Generated Outputs

| File                        | Size       | Rows    | Purpose                 |
| --------------------------- | ---------- | ------- | ----------------------- |
| `raw_demand_data.csv`       | ~100 MB    | 438,600 | Raw weather + demand    |
| `processed_demand_data.csv` | ~200 MB    | 434,400 | ML-ready features       |
| `predictions_test_set.csv`  | ~30 MB     | 86,900  | Model predictions       |
| `xgboost_model.json`        | 1.2 MB     | -       | Primary ML model        |
| `prophet_*.json`            | 25 × 25 KB | -       | Zone-specific forecasts |

---

## 🔄 DATA FLOW DIAGRAM

```
┌──────────────────────┐
│  run.py (orchestrator)
└──────────┬───────────┘
           │
    ┌──────▼─────────────────────────────────────┐
    │ STEP 1: DATA INGESTION (data_ingestion.py) │
    │ • Fetch weather from Open-Meteo API        │
    │ • Query charging stations (OpenStreetMap)  │
    │ • Simulate 2-year demand curves            │
    │ • Output: 438,600 records                  │
    └──────────┬──────────────────────────────────┘
               │ raw_demand_data.csv
               │
    ┌──────────▼────────────────────────────────────────┐
    │ STEP 2: FEATURE ENGINEERING (feature_engineering)│
    │ • Time features (hour, day_of_week, month, etc.)  │
    │ • Lag features (1h, 24h, 168h historical)        │
    │ • Rolling statistics (6h, 12h, 24h, 1week)       │
    │ • Cyclical encoding (sin/cos transforms)         │
    │ • One-hot encoding (city, zone, zone_type)       │
    │ • Output: 434,400 records, 75 features            │
    └──────────┬─────────────────────────────────────────┘
               │ processed_demand_data.csv
               │
    ┌──────────▼────────────────────────────────────────┐
    │ STEP 3: MODEL TRAINING (train.py)                │
    │ • Time-series split (80/20 train/test)           │
    │ • XGBoost: Global model (250 estimators)         │
    │ • Prophet: 25 zone-specific models               │
    │ • Cross-validation: 3-fold TS-CV                 │
    │ • SHAP explainability (300 samples)              │
    │ • Output: Models + metrics                        │
    └──────────┬─────────────────────────────────────────┘
               │ xgboost_model.json
               │ prophet_*.json (25 files)
               │ metrics.csv
               │
    ┌──────────▼────────────────────────────────────────┐
    │ STEP 4: DASHBOARD (dashboard.py)                 │
    │ • Streamlit web interface                        │
    │ • Interactive city/zone filtering                │
    │ • Real-time forecasts & maps                     │
    │ • SHAP explanations                              │
    │ • Scenario simulators                            │
    │ • Output: http://localhost:8501                  │
    └────────────────────────────────────────────────────┘
```

---

## 🎓 LEARNING PATH

### Beginner

1. Read [QUICK_START.md](QUICK_START.md) - 5 min
2. Run `python run.py` - 20 min
3. Explore dashboard - 10 min
4. Play with filters and scenarios - 10 min

### Intermediate

1. Read [README.md](README.md) sections 1-5 - 15 min
2. Review data flow diagram - 5 min
3. Look at generated CSV files - 10 min
4. Check model metrics in `reports/metrics.csv` - 5 min

### Advanced

1. Read full [README.md](README.md) - 30 min
2. Study `src/data_ingestion.py` - 10 min
3. Study `src/feature_engineering.py` - 10 min
4. Study `src/train.py` - 10 min
5. Modify hyperparameters and re-run - 20 min

### Expert

1. Customize data generation parameters
2. Add new features (temperature interaction, day-of-month, etc.)
3. Experiment with different models (LSTM, RandomForest, etc.)
4. Deploy dashboard to cloud (Heroku, AWS, GCP)
5. Integrate with real DISCOM data

---

## ✅ VERIFICATION CHECKLIST

Before running the project:

### System Setup

- [ ] Python 3.8+ installed (`python --version`)
- [ ] Internet connection available
- [ ] 2GB disk space free
- [ ] 4GB RAM minimum

### Project Files

- [ ] `run.py` exists in root
- [ ] `requirements.txt` exists in root
- [ ] `src/` folder with 4 Python files
- [ ] All source files have 0 syntax errors

### Dependencies

- [ ] Run: `pip install -r requirements.txt`
- [ ] Verify: `python -c "import pandas, xgboost, prophet, streamlit"`

### Ready to Execute

- [ ] `python run.py` starts without errors
- [ ] Dashboard launches at http://localhost:8501
- [ ] Can select different cities/zones
- [ ] All metrics display correctly

---

## 🔗 QUICK REFERENCE

### One-Liners

```bash
# Run entire pipeline
python run.py

# Just data generation
python src/data_ingestion.py

# Just features
python src/feature_engineering.py

# Just models
python src/train.py

# Just dashboard
streamlit run src/dashboard.py

# Check dependencies
pip show pandas xgboost prophet streamlit

# View model metrics
cat reports/metrics.csv

# View feature importance
cat reports/cross_validation_metrics.csv
```

### Dashboard URLs

- Local: `http://localhost:8501`
- Network: `http://<YOUR-IP>:8501`
- Custom port: `streamlit run src/dashboard.py --server.port 8502`

### File Locations

- Data: `data/*.csv`
- Models: `models/*.json`
- Metrics: `reports/*.csv`
- Plots: `reports/*.png`

---

## 📞 HELP & TROUBLESHOOTING

| Problem              | Solution                                            |
| -------------------- | --------------------------------------------------- |
| Python not found     | Install from https://www.python.org                 |
| Module not found     | `pip install -r requirements.txt`                   |
| API timeout          | Check internet; script auto-simulates               |
| Port in use          | `streamlit run src/dashboard.py --server.port 8502` |
| Memory error         | Reduce `n_estimators` in train.py                   |
| Dashboard won't load | Check firewall; try `http://127.0.0.1:8501`         |

---

## 🎯 NEXT STEPS

1. **Execute**: `python run.py`
2. **Explore**: Dashboard at http://localhost:8501
3. **Understand**: Read README.md sections 1-3
4. **Customize**: Modify parameters in src/ files
5. **Deploy**: Share dashboard with team

---

**Version**: 1.0  
**Created**: May 27, 2026  
**Status**: Production Ready ✅  
**Support**: See README.md for detailed docs

