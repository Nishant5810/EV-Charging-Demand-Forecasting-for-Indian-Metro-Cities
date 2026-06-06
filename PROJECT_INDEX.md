# PROJECT DOCUMENTATION INDEX

## START HERE

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

## EXECUTION METHODS

### Fastest (Automated)

```bash
python run.py
```

Single command runs all 4 stages
Auto-launches dashboard
Colored progress output
Error handling & recovery

### Windows (Double-Click)

```
run.bat
```

No terminal knowledge needed
Auto-detects Python
Logs output

### Linux/macOS

```bash
./run.sh
```

Bash automation
Dependency checking
Python3 compatibility

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

## DIRECTORY STRUCTURE

```
c:\projects\EV\

├── EXECUTION FILES (Use these!)
│   ├── run.py              <- Main Python runner
│   ├── run.bat             <- Windows batch runner
│   └── run.sh              <- Linux/macOS runner
│
├── DOCUMENTATION
│   ├── README.md           <- Full technical docs (15 sections)
│   ├── QUICK_START.md      <- Quick reference (this file!)
│   └── PROJECT_INDEX.md    <- Documentation map
│
├── CONFIGURATION
│   └── requirements.txt    <- Python dependencies (13 packages)
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
│       ├── processed_demand_data.csv     (434.4K rows, 50+ columns)
│       └── predictions_test_set.csv      (Test predictions)
│
├── TRAINED MODELS
│   └── models/
│       ├── xgboost_model.json           (3.2 MB - Primary model)
│       ├── prophet_Bengaluru_*.json     (25 Prophet models)
│       └── ... (5 per city)
│
└── REPORTS & OUTPUTS
    └── reports/
        ├── metrics.csv                  (XGBoost evaluation)
        ├── cross_validation_metrics.csv (Prophet validation)
        └── shap_summary.png             (Feature importance)
```

---

## QUICK LINKS

### Setup & Configuration

- [Installation Steps](README.md#installation)
- [Environment Setup](README.md#prerequisites)
- [Dependency List](requirements.txt)

### Usage & Execution

- [Quick Start](QUICK_START.md)
- [Running the Project](README.md#quick-start)
- [Manual Execution](README.md#option-2-manual-stage-by-stage)

### Understanding the Project

- [Project Overview](README.md#project-overview)
- [Problem Statement](README.md#motivation--problem-statement)
- [Features & Capabilities](README.md#features--capabilities)
- [Technology Stack](README.md#technology-stack)

### Architecture & Models

- [How It Works](README.md#how-it-works)
- [Model Architecture](README.md#model-architecture)
- [XGBoost Details](README.md#xgboost-primary-model)
- [Prophet Details](README.md#prophet-baseline--validation)

### Results & Performance

- [Model Performance](README.md#results--performance)
- [Evaluation Metrics](README.md#model-evaluation-metrics)
- [Feature Importance](README.md#feature-importance-shap-analysis)

### Advanced Topics

- [Configuration Options](README.md#configuration)
- [Customization Guide](README.md#customization)
- [Troubleshooting Guide](README.md#troubleshooting)

### Contributing & Support

- [Contributing Guidelines](README.md#contributing)
- [License Information](README.md#license)
- [Contact & Support](README.md#contact--support)

---

## EXECUTION TIMELINE

### First Run (Complete Pipeline)

| Step          | Time        | What Happens                                |
| ------------- | ----------- | ------------------------------------------- |
| 1. Start      | 0s          | Execute `python run.py`                     |
| 2. Verify     | 10s         | Check Python, dependencies, directories     |
| 3. Data Fetch | 5-10m       | Download weather data via OpenWeather API   |
| 4. Data Gen   | 2-3m        | Generate synthetic demand data for 25 zones |
| 5. Features   | 2-3m        | Create 50+ engineered features              |
| 6. XGBoost    | 3-5m        | Train primary forecasting model             |
| 7. Prophet    | 2-3m        | Train 25 zone-specific Prophet models       |
| 8. Reports    | 1m          | Generate evaluation metrics                 |
| 9. Dashboard  | 5s          | Launch Streamlit web app                    |
| **TOTAL**     | **~20 min** | Complete pipeline execution                 |

### Subsequent Runs (Cached)

Since data is cached after first run, subsequent executions complete in ~8-10 minutes.

---

## COMMON WORKFLOWS

### I want to see the dashboard immediately

```bash
# Fastest way to see visualizations
python run.py

# Or if models/data already exist:
streamlit run src/dashboard.py
```

### I want to retrain models with new data

```bash
# Delete cache to force data regeneration
rm data/processed_demand_data.csv

# Rerun pipeline
python run.py
```

### I want to modify model parameters

Edit `src/train.py`:

```python
XGBOOST_PARAMS = {
    'n_estimators': 200,  # Increase from 150
    'max_depth': 8,       # Increase from 7
}
```

Then run:

```bash
python run.py
```

### I want to add a new city/zone

Edit `src/data_ingestion.py`:

```python
CITIES_ZONES = {
    'NewCity': ['Zone1', 'Zone2']
}
```

Then run:

```bash
python run.py
```

---

## PROJECT COMPONENTS

### Data Pipeline (src/data_ingestion.py)

- Fetches historical weather via OpenWeather API
- Generates realistic EV charging demand curves
- Combines weather + demand into raw dataset
- Output: data/raw_demand_data.csv

### Feature Engineering (src/feature_engineering.py)

- Temporal encoding: Hour, day, month, year
- Cyclical encoding: sin/cos for circular features
- Lag features: Previous hour/day demand
- Rolling statistics: 7-day, 30-day averages
- Holiday indicators: Indian holidays
- Output: data/processed_demand_data.csv

### Model Training (src/train.py)

- XGBoost gradient boosting model
- 25 Prophet time-series models (one per zone)
- Cross-validation: Walk-forward validation
- Hyperparameter tuning
- Output: models/_.json, reports/_.csv

### Dashboard (src/dashboard.py)

- Real-time demand charts
- Geographic heatmaps
- Model performance metrics
- Feature importance (SHAP)
- Scenario simulation
- Interactive controls

---

## PERFORMANCE EXPECTATIONS

### Model Accuracy

- Mean Absolute Error (MAE): 15-25 kW
- Root Mean Squared Error (RMSE): 20-35 kW
- Mean Absolute Percentage Error (MAPE): 8-15%
- R² Score: 0.85-0.92

### Execution Speed

- Data generation: 5-10 minutes (first run only)
- Feature engineering: 2-3 minutes
- Model training: 5-8 minutes
- Dashboard loading: <5 seconds

### Data Volume

- Raw records: 438,600 demand samples
- Features per sample: 50+
- Cities covered: 5
- Zones per city: 5
- Total zones: 25

---

## REQUIREMENTS

### System Requirements

- Python 3.8 or higher
- 2+ GB RAM (3GB+ for comfortable operation)
- 1 GB disk space for data and models
- Internet connection (for weather API)

### Software Dependencies

- See [requirements.txt](requirements.txt) for complete list
- 13 Python packages including:
  - XGBoost (ML model)
  - Prophet (Time-series)
  - Streamlit (Dashboard)
  - Pandas, NumPy (Data processing)
  - SHAP (Explainability)

### Optional

- Git (for version control)
- Text editor or IDE for code modification

---

## GETTING HELP

### Common Issues

**"Python not found"**

- Install Python 3.8+ from python.org
- Add Python to system PATH

**"Module not found"**

- Run: `pip install -r requirements.txt`
- Ensure virtual environment is activated

**"API limit exceeded"**

- Get free OpenWeatherMap key from openweathermap.org
- Add to .env file: `WEATHER_API_KEY=your_key`

**"Dashboard won't open"**

- Check if port 8501 is in use
- Try: `streamlit run src/dashboard.py --server.port 8502`

### More Help

- See [Troubleshooting Section](README.md#troubleshooting) in README.md
- Check GitHub Issues: https://github.com/Nishant5810/EV-Charging-Demand-Forecasting-for-Indian-Metro-Cities/issues
- Review code comments in src/ files

---

## NEXT STEPS AFTER FIRST RUN

1. Explore the dashboard visualizations
2. Check model performance metrics in reports/
3. Review feature importance analysis
4. Modify parameters to improve accuracy
5. Add new cities or zones
6. Experiment with scenario planning

---

## Project Repository

https://github.com/Nishant5810/EV-Charging-Demand-Forecasting-for-Indian-Metro-Cities

For updates, issues, and contributions, visit the repository above.
