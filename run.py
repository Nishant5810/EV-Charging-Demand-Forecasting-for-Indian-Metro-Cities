#!/usr/bin/env python
"""
================================================================================
 EV CHARGING DEMAND FORECASTING & GRID LOAD OPTIMIZER - MAIN EXECUTION SCRIPT
================================================================================

Complete automated pipeline to:
1. Ingest weather and infrastructure data
2. Engineer machine learning features
3. Train XGBoost and Prophet models
4. Launch interactive Streamlit dashboard

Usage:
    python run.py

Expected Duration: ~20 minutes (first run with API calls)
Subsequent Runs: ~8 minutes (uses cached data)

================================================================================
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(title):
    """Print formatted section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}\n")


def print_step(step_num, step_name, description=""):
    """Print formatted step indicator."""
    print(f"{Colors.BOLD}{Colors.BLUE}[STEP {step_num}]{Colors.ENDC} {step_name}")
    if description:
        print(f"  {Colors.YELLOW}→ {description}{Colors.ENDC}")


def print_success(message):
    """Print success message."""
    print(f"{Colors.GREEN}✓ SUCCESS: {message}{Colors.ENDC}")


def print_error(message):
    """Print error message."""
    print(f"{Colors.RED}✗ ERROR: {message}{Colors.ENDC}")


def print_warning(message):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ WARNING: {message}{Colors.ENDC}")


def print_info(message):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ {message}{Colors.ENDC}")


def verify_directories():
    """Verify and create required directories."""
    print_step(0, "Verifying Project Structure", "Creating directories if needed")
    
    required_dirs = [
        "data",
        "models",
        "reports",
        "src"
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✓ {dir_name}/ exists")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created {dir_name}/")
    
    print()


def verify_dependencies():
    """Verify that all required Python packages are installed."""
    print_step(1, "Verifying Python Dependencies")
    
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'xgboost': 'XGBoost model',
        'prophet': 'Facebook Prophet model',
        'sklearn': 'Scikit-learn utilities',
        'shap': 'Model explainability',
        'streamlit': 'Web dashboard',
        'folium': 'Map visualization',
        'streamlit_folium': 'Streamlit Folium integration',
        'matplotlib': 'Plotting library',
        'seaborn': 'Statistical visualization',
        'holidays': 'Holiday detection',
        'requests': 'HTTP requests'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {package:20} - {description}")
        except ImportError:
            print(f"  ✗ {package:20} - {description} [MISSING]")
            missing_packages.append(package)
    
    if missing_packages:
        print_warning(f"Missing packages: {', '.join(missing_packages)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("\nOr install all dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    print_success("All dependencies installed")
    print()


def run_data_ingestion():
    """Execute data ingestion stage."""
    print_step(2, "DATA INGESTION", 
               "Fetching weather data and generating EV demand curves")
    
    print_info("This stage will:")
    print("  • Fetch 2 years of hourly weather data (Open-Meteo API)")
    print("  • Query EV charging stations (OpenStreetMap Overpass API)")
    print("  • Generate realistic EV demand patterns for 25 zones")
    print("  • Output: 438,600 records → data/raw_demand_data.csv\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "src/data_ingestion.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print_error("Data ingestion failed")
            print(result.stderr)
            return False
        
        # Verify output file exists
        if Path("data/raw_demand_data.csv").exists():
            print_success("Raw demand data generated successfully")
            print()
            return True
        else:
            print_error("Output file not created")
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Data ingestion timed out (exceeded 5 minutes)")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return False


def run_feature_engineering():
    """Execute feature engineering stage."""
    print_step(3, "FEATURE ENGINEERING",
               "Creating ML features: lags, rolling stats, cyclical encodings")
    
    print_info("This stage will:")
    print("  • Create time-based features (hour, day_of_week, month, etc.)")
    print("  • Engineer lag features (1h, 24h, 168h historical demand)")
    print("  • Calculate rolling statistics (6h, 12h, 24h, 1week)")
    print("  • Encode cyclical features with sin/cos transformations")
    print("  • One-hot encode city, zone, and zone_type identities")
    print("  • Output: 434,400 records, 75 features → data/processed_demand_data.csv\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "src/feature_engineering.py"],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print_error("Feature engineering failed")
            print(result.stderr)
            return False
        
        # Verify output file exists
        if Path("data/processed_demand_data.csv").exists():
            print_success("Features engineered successfully")
            print()
            return True
        else:
            print_error("Output file not created")
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Feature engineering timed out (exceeded 3 minutes)")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return False


def run_model_training():
    """Execute model training stage."""
    print_step(4, "MODEL TRAINING",
               "Training XGBoost + 25 Prophet models with cross-validation")
    
    print_info("This stage will:")
    print("  • Perform 3-fold time-series cross-validation")
    print("  • Train global XGBoost regressor (250 estimators)")
    print("  • Train 25 zone-specific Prophet models")
    print("  • Generate SHAP explainability plots")
    print("  • Create evaluation metrics and predictions")
    print("  • Output: Models in /models, metrics in /reports\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "src/train.py"],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print_error("Model training failed")
            print(result.stderr)
            return False
        
        # Verify key output files
        required_files = [
            "models/xgboost_model.json",
            "reports/metrics.csv",
            "data/predictions_test_set.csv"
        ]
        
        missing_files = [f for f in required_files if not Path(f).exists()]
        
        if missing_files:
            print_error(f"Missing output files: {missing_files}")
            return False
        
        print_success("Models trained successfully")
        print()
        return True
        
    except subprocess.TimeoutExpired:
        print_error("Model training timed out (exceeded 10 minutes)")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return False


def print_performance_summary():
    """Print model performance summary."""
    print_step(5, "PERFORMANCE SUMMARY",
               "Model evaluation metrics on test set")
    
    metrics_file = Path("reports/metrics.csv")
    
    if metrics_file.exists():
        import pandas as pd
        try:
            metrics = pd.read_csv(metrics_file)
            print(metrics.to_string(index=False))
            print()
            
            xgb_r2 = metrics[metrics['Model'] == 'XGBoost']['R2_score'].values[0]
            prophet_r2 = metrics[metrics['Model'] == 'Facebook Prophet']['R2_score'].values[0]
            
            print_success(f"XGBoost R² = {xgb_r2:.4f} (Best model)")
            print_success(f"Prophet R² = {prophet_r2:.4f} (Baseline)")
            print()
        except Exception as e:
            print_warning(f"Could not read metrics: {e}")
    else:
        print_warning("Metrics file not found")


def launch_dashboard():
    """Launch Streamlit dashboard."""
    print_step(6, "LAUNCHING DASHBOARD",
               "Starting interactive Streamlit web application")
    
    print_info("Dashboard will be available at:")
    print(f"  {Colors.BOLD}{Colors.GREEN}Local URL: http://localhost:8501{Colors.ENDC}")
    print(f"  {Colors.BOLD}{Colors.GREEN}Network URL: http://<your-ip>:8501{Colors.ENDC}\n")
    
    print_info("Dashboard Features:")
    print("  • 📊 Interactive city/zone filtering")
    print("  • 📈 Hourly demand forecasts")
    print("  • 🗺️  Zone overload visualization")
    print("  • 🧠 SHAP explainability plots")
    print("  • 📉 Model performance comparisons")
    print("  • 🎛️  EV growth and smart charging simulators\n")
    
    print_warning("Press Ctrl+C to stop the dashboard\n")
    
    time.sleep(2)
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "src/dashboard.py", 
             "--logger.level=error"],
            check=False
        )
    except KeyboardInterrupt:
        print_info("\nDashboard stopped")
    except Exception as e:
        print_error(f"Failed to launch dashboard: {e}")


def print_completion_summary():
    """Print completion summary."""
    print_header("EXECUTION COMPLETE ✓")
    
    print(f"{Colors.GREEN}{Colors.BOLD}All pipeline stages completed successfully!{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Generated Artifacts:{Colors.ENDC}")
    print(f"  📊 data/raw_demand_data.csv        - 438,600 raw records")
    print(f"  📊 data/processed_demand_data.csv  - 434,400 engineered features")
    print(f"  📊 data/predictions_test_set.csv   - 86,900 predictions\n")
    
    print(f"{Colors.BOLD}Trained Models:{Colors.ENDC}")
    print(f"  🤖 models/xgboost_model.json       - Primary gradient boosting model")
    print(f"  🤖 models/prophet_*.json           - 25 zone-specific time-series models\n")
    
    print(f"{Colors.BOLD}Reports & Metrics:{Colors.ENDC}")
    print(f"  📈 reports/metrics.csv             - Model comparison (MAE, RMSE, R²)")
    print(f"  📈 reports/cross_validation_metrics.csv - 3-fold CV results")
    print(f"  📈 reports/shap_summary.png        - Feature importance plot\n")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print(f"  1. View dashboard at {Colors.GREEN}http://localhost:8501{Colors.ENDC}")
    print(f"  2. Select city and zone from sidebar filters")
    print(f"  3. Explore forecasts, maps, and explainability tabs")
    print(f"  4. Test smart charging and growth simulation scenarios\n")
    
    print(f"{Colors.BOLD}To Run Again:{Colors.ENDC}")
    print(f"  python run.py\n")


def main():
    """Main execution flow."""
    
    print_header("EV CHARGING DEMAND FORECASTING - AUTOMATED PIPELINE")
    
    print(f"  Project: EV Charging Demand Forecasting & Grid Load Optimizer")
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Working Directory: {Path.cwd()}\n")
    
    # Stage 0: Verify setup
    verify_directories()
    verify_dependencies()
    
    # Stage 1: Data Ingestion
    if not run_data_ingestion():
        print_error("Pipeline failed at data ingestion stage")
        sys.exit(1)
    
    # Stage 2: Feature Engineering
    if not run_feature_engineering():
        print_error("Pipeline failed at feature engineering stage")
        sys.exit(1)
    
    # Stage 3: Model Training
    if not run_model_training():
        print_error("Pipeline failed at model training stage")
        sys.exit(1)
    
    # Stage 4: Performance Summary
    print_performance_summary()
    
    # Completion
    print_completion_summary()
    
    # Stage 5: Launch Dashboard
    print_info(f"Launching dashboard in 3 seconds...\n")
    time.sleep(3)
    launch_dashboard()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
