# Quick Start Guide

## One-Command Execution

Run your entire EV forecasting project with a single command:

### Windows

```bash
python run.py
```

Or double-click: `run.bat`

### Linux / macOS

```bash
python3 run.py
```

Or:

```bash
chmod +x run.sh
./run.sh
```

---

## Expected Timeline

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

## What Gets Generated

### Data Files

- `data/raw_demand_data.csv` - 438,600 weather-demand records
- `data/processed_demand_data.csv` - 434,400 engineered features
- `data/predictions_test_set.csv` - Model predictions on test set

### Trained Models

- `models/xgboost_model.json` - Primary forecasting model
- `models/prophet_[City]_[Zone].json` - 25 Prophet zone-specific models

### Reports

- `reports/metrics.csv` - XGBoost evaluation metrics
- `reports/cross_validation_metrics.csv` - Prophet cross-validation results

### Dashboard

- Automatically opens at `http://localhost:8501`
- Interactive charts, maps, and scenario planning

---

## Troubleshooting

### Issue: "Module not found" error

**Solution:**

```bash
pip install -r requirements.txt
```

### Issue: Dashboard won't open

**Solution:**

```bash
# Make sure Streamlit is installed
pip install streamlit

# Try with explicit port
streamlit run src/dashboard.py --server.port 8501
```

### Issue: API rate limit exceeded

**Solution:**

```bash
# Get free API key from openweathermap.org
# Add to .env file:
WEATHER_API_KEY=your_api_key_here
```

### Issue: Out of memory during training

**Solution:**
Edit `src/train.py` and reduce batch size:

```python
BATCH_SIZE = 5000  # Reduce from default
```

---

## File Structure Overview

```
project/
├── run.py, run.bat, run.sh     - Entry point scripts
├── README.md                    - Full documentation
├── QUICK_START.md              - This file
├── requirements.txt            - Python dependencies
├── src/
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── dashboard.py
├── data/                       - Generated datasets
├── models/                     - Trained models
└── reports/                    - Evaluation metrics
```

---

## Next Steps

1. Run `python run.py` to execute the complete pipeline
2. Wait for the dashboard to launch
3. Explore the interactive visualizations
4. Check the model performance metrics
5. Modify configurations in `src/train.py` to experiment

---

## Documentation

- **Full Documentation**: See [README.md](README.md) for complete details
- **Project Index**: See [PROJECT_INDEX.md](PROJECT_INDEX.md) for documentation map
- **Installation Guide**: See [README.md](README.md#installation) for detailed setup

---

For more information, visit: https://github.com/Nishant5810/EV-Charging-Demand-Forecasting-for-Indian-Metro-Cities
