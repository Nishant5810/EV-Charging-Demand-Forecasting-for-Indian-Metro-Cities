# 🚀 Quick Start Guide

## One-Command Execution

Run your entire EV forecasting project with a single command:

### 🪟 Windows

```bash
python run.py
```

Or double-click: `run.bat`

### 🐧 Linux / 🍎 macOS

```bash
python3 run.py
```

Or:

```bash
chmod +x run.sh
./run.sh
```

---

## ⏱️ Expected Timeline

| Stage                   | Duration    | Description                           |
| ----------------------- | ----------- | ------------------------------------- |
| **Verification**        | 10s         | Check dependencies and directories    |
| **Data Ingestion**      | 5-10 min    | Fetch weather, generate demand curves |
| **Feature Engineering** | 2-3 min     | Create ML features                    |
| **Model Training**      | 5-8 min     | XGBoost + 25 Prophet models           |
| **Dashboard Launch**    | 5s          | Start Streamlit web app               |
| **TOTAL**               | **~20 min** | (First run with API calls)            |

**Subsequent runs:** ~8 minutes (cached data)

---

## 📊 What Gets Generated

### Data Files

- ✅ `data/raw_demand_data.csv` - 438,600 weather-demand records
- ✅ `data/processed_demand_data.csv` - 434,400 engineered features
- ✅ `data/predictions_test_set.csv` - 86,900 model predictions

### Trained Models

- ✅ `models/xgboost_model.json` - Main gradient boosting model
- ✅ `models/prophet_*.json` - 25 zone-specific forecasts

### Reports & Metrics

- ✅ `reports/metrics.csv` - XGBoost vs Prophet comparison
- ✅ `reports/cross_validation_metrics.csv` - 3-fold CV results
- ✅ `reports/shap_summary.png` - Feature importance plot

---

## 🌐 Access Dashboard

Once `run.py` completes, your dashboard launches automatically:

### Local Access

```
http://localhost:8501
```

### Network Access

```
http://<YOUR-IP>:8501
```

### Dashboard Features

- 🎛️ City & zone selection
- 📈 Hourly demand forecasts
- 🗺️ Interactive overload maps
- 🧠 SHAP explainability
- 📊 Model comparisons
- 🎲 Scenario simulators

---

## 📋 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 2GB free space
- **Internet**: Required for API calls (first run only)

### Check Python Version

```bash
python --version  # Windows
python3 --version # Linux/macOS
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚠️ Troubleshooting

### Issue: "Python not found"

```bash
# Windows: Add Python to PATH, or use full path
C:\Python313\python.exe run.py

# Linux/macOS: Use python3 instead of python
python3 run.py
```

### Issue: "Module not found"

```bash
pip install -r requirements.txt
```

### Issue: "API timeout"

- Network might be slow
- Script automatically falls back to simulation
- Check internet connection

### Issue: "Port 8501 already in use"

```bash
# Windows PowerShell
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Linux/macOS
lsof -i :8501
kill -9 <PID>

# Or use different port:
streamlit run src/dashboard.py --server.port 8502
```

### Issue: "Memory error"

Reduce model complexity:

1. Edit `src/train.py`
2. Change `n_estimators=250` to `n_estimators=100`
3. Rerun: `python run.py`

---

## 📚 Project Structure

```
EV/
├── run.py              ← Main execution script (USE THIS!)
├── run.bat             ← Windows batch runner
├── run.sh              ← Linux/macOS shell runner
├── requirements.txt    ← Python dependencies
├── README.md           ← Full documentation
├── QUICK_START.md      ← This file
│
├── data/               ← Generated datasets
│   ├── raw_demand_data.csv
│   ├── processed_demand_data.csv
│   └── predictions_test_set.csv
│
├── models/             ← Trained ML models
│   ├── xgboost_model.json
│   └── prophet_*.json
│
├── reports/            ← Metrics & visualizations
│   ├── metrics.csv
│   ├── cross_validation_metrics.csv
│   └── shap_summary.png
│
└── src/                ← Python source code
    ├── data_ingestion.py
    ├── feature_engineering.py
    ├── train.py
    └── dashboard.py
```

---

## 🎯 Next Steps

1. **Run the pipeline**

   ```bash
   python run.py
   ```

2. **Open dashboard** (auto-launched or manually)

   ```
   http://localhost:8501
   ```

3. **Explore features**
   - Select different cities and zones
   - View hourly forecasts and peak windows
   - Check overload maps
   - Analyze feature importance (SHAP)
   - Compare model performance
   - Test growth and smart charging scenarios

4. **View detailed documentation**

   ```bash
   # Open in text editor or browser
   README.md
   ```

5. **Customize for your needs**
   - Edit data ingestion parameters
   - Adjust model hyperparameters
   - Add new forecast horizons
   - Modify dashboard colors/styling

---

## 🔗 File Usage Matrix

| Task                  | Run Command                         | Files Used           | Output                      |
| --------------------- | ----------------------------------- | -------------------- | --------------------------- |
| **Complete Pipeline** | `python run.py`                     | All source files     | All outputs                 |
| **Just Data**         | `python src/data_ingestion.py`      | Weather/OSM APIs     | `raw_demand_data.csv`       |
| **Just Features**     | `python src/feature_engineering.py` | Raw data CSV         | `processed_demand_data.csv` |
| **Just Training**     | `python src/train.py`               | Processed data       | Models + metrics            |
| **Just Dashboard**    | `streamlit run src/dashboard.py`    | Models + predictions | Web app @ :8501             |

---

## 💾 First Run Checklist

- [ ] Python 3.8+ installed
- [ ] Internet connection available
- [ ] `requirements.txt` in project root
- [ ] 2GB free disk space
- [ ] Run: `python run.py`
- [ ] Wait 15-20 minutes for completion
- [ ] Open dashboard: http://localhost:8501
- [ ] Explore city/zone data
- [ ] ✅ Done!

---

## 📞 Support

**If pipeline fails:**

1. Check error message in terminal
2. Verify internet connection
3. Ensure all dependencies installed: `pip install -r requirements.txt`
4. Check disk space: `df -h` (Linux/Mac) or `dir` (Windows)
5. Try again with: `python run.py`

**For detailed documentation:**

- See `README.md` for comprehensive guide
- Each source file (`src/*.py`) has inline comments

**Expected output on success:**

```
✓ SUCCESS: Raw demand data generated successfully
✓ SUCCESS: Features engineered successfully
✓ SUCCESS: Models trained successfully
✓ SUCCESS: All dependencies installed

You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

---

**Version**: 1.0  
**Last Updated**: May 27, 2026  
**Status**: Production Ready ✅

