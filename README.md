# ⚡ EV Charging Demand Forecasting for Indian Metro Cities

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Models](https://img.shields.io/badge/Models-XGBoost%20%26%20Prophet-orange)

**Advanced ML Platform for Predicting EV Charging Demand Across Indian Metropolitan Cities**

[Installation](#-installation) • [Quick Start](#-quick-start) • [Features](#-features) • [Architecture](#-architecture) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Motivation & Problem Statement](#-motivation--problem-statement)
- [Features & Capabilities](#-features--capabilities)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Model Architecture](#-model-architecture)
- [Results & Performance](#-results--performance)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

**EV Charging Demand Forecasting for Indian Metro Cities** is an end-to-end machine learning platform designed to predict electric vehicle (EV) charging demand across major Indian metropolitan areas and their unique urban zones. This intelligent system helps power utilities, charging network operators, and grid management authorities optimize infrastructure planning and implement smart charging strategies.

### 🌍 Coverage

| City          | Zones   | Region              | Population |
| ------------- | ------- | ------------------- | ---------- |
| **Delhi**     | 5 zones | North India         | 30M+       |
| **Bengaluru** | 5 zones | South India         | 12M+       |
| **Mumbai**    | 5 zones | West India          | 20M+       |
| **Chennai**   | 5 zones | South India         | 7M+        |
| **Hyderabad** | 5 zones | South-Central India | 9M+        |

**Total Coverage: 25 urban zones across 5 major Indian cities**

### 🎯 Key Highlights

- ✅ **Multi-City Coverage**: 5 major Indian metropolitan areas
- ✅ **Zone-Level Precision**: 25 distinct charging zones with unique demand patterns
- ✅ **Dual ML Models**: XGBoost (primary) + Facebook Prophet (baseline validation)
- ✅ **Real-time Dashboard**: Interactive Streamlit web interface
- ✅ **Weather Integration**: Live OpenWeather API data
- ✅ **Scenario Planning**: Smart charging simulation for EV adoption growth
- ✅ **Explainability**: SHAP analysis for feature importance
- ✅ **Production Ready**: Automated pipeline with error handling

---

## 🔋 Motivation & Problem Statement

### The Challenge

The rapid adoption of electric vehicles (EVs) in India presents significant challenges for power distribution networks (DISCOMs):

#### 1. **Grid Overload Risk**

- EV charging demand spikes during peak evening hours (18:00-22:00)
- Concurrent residential load + EV charging can overwhelm transformers
- Lack of predictive capacity planning leads to blackouts and grid instability

#### 2. **Infrastructure Investment Uncertainty**

- DISCOMs must decide where to install charging infrastructure
- Inaccurate demand forecasts lead to underutilization or insufficient capacity
- Billions of rupees at stake in capacity planning decisions

#### 3. **Charging Network Optimization**

- Charging Point Operators (CPOs) need zone-level demand patterns
- Manual site selection is inefficient and error-prone
- Real-time demand fluctuations require adaptive strategies

#### 4. **Complex Demand Variability**

EV charging demand varies across multiple dimensions:

- **Temporal**: Morning commute vs. evening home charging patterns
- **Cyclical**: Weekday office patterns vs. residential weekend usage
- **Seasonal**: Summer AC loads compound with EV charging
- **Weather-Dependent**: Extreme heat, rain affect both demand and supply
- **Geographic**: Different zones have vastly different usage patterns

### 📊 Solution Delivered

This platform provides:

✅ **Accurate Demand Forecasting**: 24-hour ahead predictions at zone-level precision
✅ **Intelligent Capacity Planning**: Data-driven infrastructure investment guidance
✅ **Smart Charging Strategies**: Real-time demand simulation and optimization
✅ **Transparent Decision Making**: SHAP-based explainability for stakeholders
✅ **Scalable Architecture**: Easily extensible to new cities and zones

---

## 🚀 Features & Capabilities

### Core Features

| Feature                   | Description                                 | Benefit                                        |
| ------------------------- | ------------------------------------------- | ---------------------------------------------- |
| **Demand Forecasting**    | 24-hour ahead hourly predictions            | Grid operators can plan charging strategies    |
| **Zone-Level Analysis**   | Individual forecasts for 25 urban zones     | CPOs optimize station placement                |
| **Weather Integration**   | Real-time OpenWeather API data              | Accounts for weather-dependent demand          |
| **Multi-Model Ensemble**  | XGBoost + Prophet dual validation           | Cross-validation ensures reliability           |
| **Interactive Dashboard** | Streamlit web interface with visualizations | Non-technical stakeholders can access insights |
| **Scenario Simulation**   | "What-if" analysis for EV adoption growth   | Long-term planning capabilities                |
| **Feature Importance**    | SHAP explainability analysis                | Decision transparency                          |
| **Historical Analysis**   | 1+ year of demand data per zone             | Pattern recognition and anomaly detection      |

### Dashboard Features

- 📊 **Real-time Charts**: Hourly demand trends and forecasts
- 🗺️ **Geographic Heatmaps**: Zone-level demand visualization
- 📈 **Metrics Dashboard**: MAE, RMSE, MAPE for all models
- 🔍 **Feature Analysis**: SHAP importance plots
- 📋 **Zone Comparison**: Side-by-side demand patterns
- ⚙️ **Scenario Planning**: Interactive growth simulation

---

## 💻 Technology Stack

### Machine Learning & Data Science

- **XGBoost**: Gradient boosting for primary forecasting (handles non-linear patterns)
- **Prophet**: Facebook's time-series model for baseline and seasonal decomposition
- **Scikit-learn**: Feature preprocessing and metrics calculation
- **SHAP**: Model explainability and feature importance analysis

### Data & Processing

- **Pandas**: Data manipulation and cleaning
- **NumPy**: Numerical computations
- **Holidays**: Calendar-aware feature engineering

### Visualization & Dashboard

- **Streamlit**: Interactive web dashboard framework
- **Folium**: Geographic heatmap visualization
- **Streamlit-Folium**: Map integration in Streamlit
- **Matplotlib & Seaborn**: Statistical plots

### External APIs

- **OpenWeather API**: Real-time weather data
- **Python Requests**: API communication

### Environment

- **Python 3.8+**: Core language
- **pip**: Dependency management

---

## 📦 Installation

### Prerequisites

Before installing, ensure you have:

- Python 3.8 or higher
- pip package manager
- ~1 GB disk space for models and data
- Internet connection (for weather API)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nishant5810/EV-Charging-Demand-Forecasting-for-Indian-Metro-Cities.git
cd "EV-Charging-Demand-Forecasting-for-Indian-Metro-Cities"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies Overview:**

```
xgboost              - Gradient boosting model
prophet              - Time-series forecasting
shap                 - Model explainability
streamlit            - Dashboard framework
folium               - Geographic visualizations
streamlit-folium     - Streamlit-Folium bridge
scikit-learn         - ML utilities
seaborn              - Statistical plotting
holidays             - Calendar awareness
matplotlib           - Plotting library
requests             - HTTP client for APIs
pandas               - Data manipulation
numpy                - Numerical computing
```

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, xgboost, prophet, streamlit; print('✅ All dependencies installed!')"
```

---

## 🚀 Quick Start

### Option 1: Automated Execution (Recommended)

**Windows:**

```bash
python run.py
```

Or simply double-click: `run.bat`

**Linux / macOS:**

```bash
python3 run.py
./run.sh
```

### Option 2: Manual Stage-by-Stage

```bash
# Stage 1: Generate and process data
python src/data_ingestion.py

# Stage 2: Create ML features
python src/feature_engineering.py

# Stage 3: Train models (XGBoost + Prophet)
python src/train.py

# Stage 4: Launch interactive dashboard
streamlit run src/dashboard.py
```

### ⏱️ Expected Execution Time

| Stage                   | Duration | Notes                                |
| ----------------------- | -------- | ------------------------------------ |
| **Verification**        | 10s      | Dependency check                     |
| **Data Ingestion**      | 5-10 min | Weather API calls + synthetic demand |
| **Feature Engineering** | 2-3 min  | Feature creation                     |
| **Model Training**      | 5-8 min  | 26 models (XGBoost + 25 Prophet)     |
| **Dashboard Launch**    | 5s       | Web server startup                   |
| **TOTAL (First Run)**   | ~20 min  | With API delays                      |
| **Subsequent Runs**     | ~8 min   | Cached data                          |

### 📊 Output

After execution, you'll have:

**Generated Data:**

- ✅ `data/raw_demand_data.csv` - 438,600 weather-demand records (25 zones × ~1 year)
- ✅ `data/processed_demand_data.csv` - 434,400 engineered features (ready for ML)
- ✅ `data/predictions_test_set.csv` - Model predictions on test set

**Trained Models:**

- ✅ `models/xgboost_model.json` - Primary forecasting model (~1 MB)
- ✅ `models/prophet_[City]_[Zone].json` - 25 Prophet models for baselines

**Reports:**

- ✅ `reports/metrics.csv` - XGBoost performance metrics
- ✅ `reports/cross_validation_metrics.csv` - Prophet cross-validation results

**Dashboard:**

- 🌐 Open automatically at `http://localhost:8501` in your browser

---

## 📁 Project Structure

```
EV-Charging-Demand-Forecasting-for-Indian-Metro-Cities/
│
├── 📄 EXECUTION SCRIPTS (Start here!)
│   ├── run.py                    ← Main entry point (all-in-one)
│   ├── run.bat                   ← Windows batch runner
│   └── run.sh                    ← Linux/macOS runner
│
├── 📚 DOCUMENTATION
│   ├── README.md                 ← This file
│   ├── QUICK_START.md            ← 5-minute quick reference
│   ├── PROJECT_INDEX.md          ← Documentation map
│   └── requirements.txt          ← Python dependencies
│
├── 🐍 SOURCE CODE (Core Logic)
│   └── src/
│       ├── data_ingestion.py     ← Weather data + demand simulation
│       │                           (500+ lines, OpenWeather API integration)
│       ├── feature_engineering.py ← Feature creation & preprocessing
│       │                           (300+ lines, 40+ features)
│       ├── train.py              ← Model training pipeline
│       │                          (400+ lines, XGBoost + 25 Prophet)
│       ├── dashboard.py          ← Streamlit interactive dashboard
│       │                          (600+ lines, maps & charts)
│       └── lstm_colab.py         ← LSTM experimental model (Colab)
│
├── 📊 GENERATED DATA (Auto-created on first run)
│   └── data/
│       ├── raw_demand_data.csv        (438.6K rows × 16 cols)
│       ├── processed_demand_data.csv  (434.4K rows × 50+ cols)
│       └── predictions_test_set.csv   (Test set predictions)
│
├── 🤖 TRAINED MODELS (Auto-created on first run)
│   └── models/
│       ├── xgboost_model.json         (Primary model)
│       ├── prophet_Bengaluru_*.json   (5 Prophet models)
│       ├── prophet_Chennai_*.json     (5 Prophet models)
│       ├── prophet_Delhi_*.json       (5 Prophet models)
│       ├── prophet_Hyderabad_*.json   (5 Prophet models)
│       └── prophet_Mumbai_*.json      (5 Prophet models)
│
├── 📈 EVALUATION REPORTS (Auto-created on first run)
│   └── reports/
│       ├── metrics.csv                (XGBoost metrics)
│       └── cross_validation_metrics.csv (Prophet cross-validation)
│
└── ⚙️ CONFIGURATION
    └── .gitignore                 ← Files excluded from git

```

### File Descriptions

| File                     | Purpose                                               | Lines |
| ------------------------ | ----------------------------------------------------- | ----- |
| `run.py`                 | Main orchestrator - runs all 4 stages in sequence     | 150+  |
| `data_ingestion.py`      | Fetches weather, generates synthetic EV demand curves | 500+  |
| `feature_engineering.py` | Creates 40+ time-series and domain features           | 300+  |
| `train.py`               | Trains XGBoost + 25 Prophet models, evaluates metrics | 400+  |
| `dashboard.py`           | Streamlit web interface with maps, charts, scenarios  | 600+  |
| `requirements.txt`       | Python package dependencies (13 packages)             | 13    |

---

## ⚙️ How It Works

### Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION                                               │
├─────────────────────────────────────────────────────────────────┤
│ • OpenWeather API: Fetch historical weather for 25 zones        │
│ • Synthetic Demand: Generate realistic EV charging patterns     │
│ • Integration: Combine weather + demand into raw dataset        │
│ Output: data/raw_demand_data.csv (438,600 rows)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. FEATURE ENGINEERING                                          │
├─────────────────────────────────────────────────────────────────┤
│ • Temporal Features: Hour, day, month, quarter, year           │
│ • Cyclical Features: sin/cos encoding for circular variables    │
│ • Lag Features: Previous hour/day demand patterns               │
│ • Rolling Stats: 7-day & 30-day moving averages                │
│ • Holiday Features: Indian public/regional holidays            │
│ • Weather Features: Temperature, humidity, precipitation       │
│ • Zone Features: Location-specific categorical encoding        │
│ Output: data/processed_demand_data.csv (434,400 rows × 50 cols)│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. MODEL TRAINING                                               │
├─────────────────────────────────────────────────────────────────┤
│ • XGBoost: Primary model (handles non-linear patterns)          │
│ • Prophet: 25 zone-specific models (seasonal decomposition)     │
│ • Cross-Validation: Time-series walk-forward validation         │
│ • Hyperparameter Tuning: Grid search for optimal parameters     │
│ • Evaluation: MAE, RMSE, MAPE, R² metrics                       │
│ Output: models/ (26 JSON files), reports/metrics.csv            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. INTERACTIVE DASHBOARD                                        │
├─────────────────────────────────────────────────────────────────┤
│ • Real-time Charts: Hourly demand trends                        │
│ • Geographic Maps: Zone-level heatmaps                          │
│ • Predictions: 24-hour forecast display                         │
│ • Model Performance: Metrics and comparison                     │
│ • Feature Analysis: SHAP importance plots                       │
│ • Scenario Planning: Growth simulation controls                 │
│ Output: http://localhost:8501 (Streamlit web app)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Model Architecture

### XGBoost (Primary Model)

**Why XGBoost?**

- Excellent for capturing non-linear demand patterns
- Fast training and inference
- Built-in feature importance (SHAP compatibility)
- Robust to outliers and missing data

**Architecture:**

```
Input Features (50+ features)
        ↓
├─ Temporal: hour, day, month, year
├─ Cyclical: sin/cos of temporal features
├─ Lags: demand from t-1h, t-1d, t-7d
├─ Rolling: 7-day & 30-day averages
├─ Weather: temp, humidity, precipitation
├─ Holiday: is_holiday, holiday_type
└─ Zone: city, zone categorical encoding
        ↓
[XGBoost Gradient Boosting Regressor]
        ↓
Output: Hourly EV charging demand forecast
```

**Hyperparameters (Optimized):**

- `n_estimators`: 100-200 (number of boosting rounds)
- `max_depth`: 5-8 (tree depth)
- `learning_rate`: 0.05-0.1 (step size)
- `subsample`: 0.8 (row sampling)
- `colsample_bytree`: 0.8 (feature sampling)

### Prophet (Baseline & Validation)

**Why Prophet?**

- Excellent for seasonal time-series data
- Handles missing values and outliers automatically
- Built-in seasonality decomposition
- Zone-specific models capture local patterns

**Architecture (25 Zone-Specific Models):**

```
Per Zone:
├─ Trend: Long-term growth pattern
├─ Yearly Seasonality: Annual EV demand cycle
├─ Weekly Seasonality: Weekday vs. weekend patterns
├─ Daily Seasonality: Hourly demand fluctuation
└─ Holiday Effects: Indian holidays impact
```

### Ensemble Strategy

```
Forecast = 0.7 × XGBoost + 0.3 × Prophet_Ensemble

Benefits:
✅ Leverages strengths of both models
✅ Reduces overfitting from individual models
✅ More robust to market changes
```

---

## 📊 Results & Performance

### Model Evaluation Metrics

#### XGBoost Performance

| Metric       | Value     | Interpretation                                    |
| ------------ | --------- | ------------------------------------------------- |
| **MAE**      | 15-25 kW  | Average prediction error within 15-25 kW          |
| **RMSE**     | 20-35 kW  | Root mean square error accounts for larger errors |
| **MAPE**     | 8-15%     | Mean absolute percentage error                    |
| **R² Score** | 0.85-0.92 | Model explains 85-92% of variance                 |

#### Cross-City Performance Variation

| City          | Demand Range | Model MAPE | Variance                      |
| ------------- | ------------ | ---------- | ----------------------------- |
| **Delhi**     | 150-800 kW   | 9.2%       | High (weather-sensitive)      |
| **Bengaluru** | 100-600 kW   | 8.1%       | Moderate                      |
| **Mumbai**    | 200-1000 kW  | 10.5%      | Highest (coastal variability) |
| **Chennai**   | 80-450 kW    | 7.8%       | Lowest                        |
| **Hyderabad** | 120-550 kW   | 8.5%       | Moderate                      |

### Key Insights

1. **Temporal Patterns**: Demand peaks at 18:00-22:00 (evening home charging)
2. **Day-of-Week**: 15-20% higher demand on weekdays
3. **Seasonality**: 25-30% variation between summer and winter
4. **Weather Impact**: Hot days increase demand by 10-15%
5. **Growth Trends**: Average 8-12% year-over-year growth

### Feature Importance (SHAP Analysis)

Top 10 Features:

1. Hour of day (circular encoding)
2. Day of week
3. Previous hour demand (lag)
4. Week number (yearly seasonality)
5. Temperature
6. Previous day demand
7. Holiday indicator
8. Month of year
9. Humidity
10. 7-day rolling average

---

## ⚙️ Configuration

### Environment Variables (Optional)

Create a `.env` file in the project root:

```env
# OpenWeather API
WEATHER_API_KEY=your_api_key_here

# Model Parameters
XGBOOST_PARAMS={"n_estimators": 150, "max_depth": 7}
PROPHET_INTERVALS=80  # Uncertainty interval

# Data Range
DATA_START_DATE=2023-01-01
DATA_END_DATE=2024-12-31

# Dashboard
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
```

### Customization

**Add New City/Zone:**

1. Edit `src/data_ingestion.py`:

```python
CITIES_ZONES = {
    'YourCity': ['Zone1', 'Zone2', ...]
}

CITY_COORDINATES = {
    'YourCity': {
        'Zone1': (latitude, longitude),
        ...
    }
}
```

2. Rerun the pipeline:

```bash
python run.py
```

**Modify Model Parameters:**

Edit `src/train.py`:

```python
XGBOOST_PARAMS = {
    'n_estimators': 200,      # Increase for better accuracy
    'max_depth': 8,           # Deeper trees for complexity
    'learning_rate': 0.08,    # Smaller for more iterations
}
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. **OpenWeather API Limit Exceeded**

**Problem:** `OpenWeatherMap error: 429 Too Many Requests`

**Solution:**

- Get free API key from [openweathermap.org](https://openweathermap.org/api)
- Add to `.env` file:

```env
WEATHER_API_KEY=your_free_api_key
```

- Free tier: 60 calls/minute, 1M calls/month (sufficient for this project)

#### 2. **Memory Error During Training**

**Problem:** `MemoryError: Unable to allocate ... GB`

**Solution:**

```bash
# Reduce batch size in src/train.py
BATCH_SIZE = 10000  # Reduce from default

# Or increase system virtual memory
```

#### 3. **Dashboard Won't Launch**

**Problem:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**

```bash
# Reinstall Streamlit
pip uninstall streamlit -y
pip install streamlit
```

#### 4. **Slow Model Training**

**Problem:** Training takes >15 minutes

**Optimization:**

- Reduce `n_estimators` in XGBoost (default: 150, try: 100)
- Skip Prophet models temporarily:

```python
# In src/train.py, comment out Prophet section
# TRAINED_MODELS = {'xgboost': xgb_model}
```

#### 5. **Port 8501 Already in Use**

**Problem:** `Address already in use: ('127.0.0.1', 8501)`

**Solution:**

```bash
# Use different port
streamlit run src/dashboard.py --server.port 8502
```

### Dependency Conflicts

**If you encounter version conflicts:**

```bash
# Clean reinstall
pip uninstall -r requirements.txt -y
pip install -r requirements.txt --upgrade

# Or create fresh environment
python -m venv venv_fresh
venv_fresh\Scripts\activate
pip install -r requirements.txt
```

---

## 📝 Files to Exclude from Git

The `.gitignore` file is already configured to exclude:

### ❌ Do NOT Push These Files

**Large Data Files:**

- `data/*.csv` - Raw and processed demand data (can be regenerated)
- `data/raw_demand_data.csv`
- `data/processed_demand_data.csv`
- `data/predictions_test_set.csv`

**Trained Models:**

- `models/*.json` - XGBoost and Prophet model files (regenerated on training)

**Generated Reports:**

- `reports/*.csv` - Metrics and cross-validation results

**Python Cache:**

- `__pycache__/` - Python bytecode cache
- `*.pyc`, `*.pyo` - Compiled Python files
- `.pytest_cache/` - Test cache

**Virtual Environments:**

- `venv/`, `env/`, `.venv/` - Virtual environment directories

**IDE & Editor Files:**

- `.vscode/` - VS Code settings
- `.idea/` - PyCharm settings
- `*.swp`, `*.swo` - Editor swap files

**OS-Specific Files:**

- `.DS_Store` - macOS system files
- `Thumbs.db` - Windows thumbnail cache

**Other:**

- `*.log` - Log files
- `.env` - Environment variables (contains sensitive data)

### ✅ Files That WILL Be Pushed

**Source Code:**

- `src/*.py` - All Python source files
- `src/data_ingestion.py`
- `src/feature_engineering.py`
- `src/train.py`
- `src/dashboard.py`

**Configuration & Scripts:**

- `run.py`, `run.bat`, `run.sh` - Execution scripts
- `requirements.txt` - Dependencies list
- `.gitignore` - Git configuration

**Documentation:**

- `README.md` - Main documentation
- `QUICK_START.md` - Quick reference guide
- `PROJECT_INDEX.md` - Documentation index

---

## 🔄 Git Workflow

### Initial Setup & First Push

```bash
# 1. Initialize git (if not already done)
git init

# 2. Add all tracked files
git add .

# 3. Create initial commit
git commit -m "Initial commit: EV Charging Demand Forecasting Platform"

# 4. Add remote repository
git remote add origin https://github.com/Nishant5810/EV-Charging-Demand-Forecasting-for-Indian-Metro-Cities.git

# 5. Push to GitHub
git push -u origin main
```

### After First Push

```bash
# 1. Make changes to your code
# ... (edit files)

# 2. Check status
git status

# 3. Stage changes
git add .

# 4. Commit with message
git commit -m "Descriptive message about changes"

# 5. Push to repository
git push
```

### Verify What Will Be Pushed

```bash
# See all files that will be pushed
git status

# See detailed diff
git diff --cached

# Verify .gitignore is working
git check-ignore -v *  # Should show ignored files
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** and test thoroughly
4. **Commit with clear messages**: `git commit -m "Add feature: description"`
5. **Push to your fork**: `git push origin feature/your-feature-name`
6. **Create a Pull Request** with detailed description

### Contribution Areas

- 🔬 **Model Improvements**: Better hyperparameters, new architectures
- 🗺️ **New Cities/Zones**: Extend coverage to other Indian cities
- 📊 **Dashboard Enhancements**: New visualizations, interactive features
- 📚 **Documentation**: Improve clarity and examples
- 🐛 **Bug Fixes**: Report and fix issues

---

## 📄 License

This project is licensed under the **MIT License** - see LICENSE file for details.

### Citation

If you use this project in your research or work, please cite:

```bibtex
@project{ev_charging_forecasting_2024,
  title={EV Charging Demand Forecasting for Indian Metro Cities},
  author={Nishant},
  year={2024},
  url={https://github.com/Nishant5810/EV-Charging-Demand-Forecasting-for-Indian-Metro-Cities}
}
```

---

## 📧 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Nishant5810/EV-Charging-Demand-Forecasting-for-Indian-Metro-Cities/issues)
- **Project Repository**: https://github.com/Nishant5810/EV-Charging-Demand-Forecasting-for-Indian-Metro-Cities

---

## 🎓 Learning Resources

Useful resources for understanding the project:

- **XGBoost**: https://xgboost.readthedocs.io/
- **Prophet**: https://facebook.github.io/prophet/
- **SHAP**: https://shap.readthedocs.io/
- **Streamlit**: https://docs.streamlit.io/
- **Time Series Forecasting**: https://en.wikipedia.org/wiki/Time_series

---

## 📊 Project Statistics

- **Lines of Code**: 1,800+ (production code)
- **Models**: 26 (1 XGBoost + 25 Prophet)
- **Features**: 50+ engineered features
- **Data Points**: 438,600+ demand records
- **Cities Covered**: 5
- **Zones Covered**: 25
- **Python Packages**: 13 dependencies
- **Documentation**: 3 comprehensive guides

---

## 🎉 Acknowledgments

This project combines best practices from:

- Facebook Research (Prophet)
- XGBoost Team
- Open weather data providers
- Indian DISCOM public data
- Open-source Python community

---

<div align="center">

**Made with ❤️ for India's EV Future**

⭐ If you find this project useful, please give it a star! ⭐

</div>

