#!/bin/bash

################################################################################
# EV CHARGING DEMAND FORECASTING - LINUX/MAC SHELL RUNNER
################################################################################
#
# This script automatically runs the complete EV forecasting pipeline:
# 1. Data Ingestion
# 2. Feature Engineering  
# 3. Model Training
# 4. Dashboard Launch
#
# Usage: chmod +x run.sh && ./run.sh
#
################################################################################

set -e

echo ""
echo "============================================================================="
echo " EV CHARGING DEMAND FORECASTING & GRID LOAD OPTIMIZER"
echo "============================================================================="
echo ""
echo "Starting complete automated pipeline..."
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "  macOS: brew install python3"
    echo "  Or download from https://www.python.org"
    exit 1
fi

# Check Python version
python3_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Using Python: $python3_version"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found"
    echo "This script must be run from the project root directory"
    exit 1
fi

# Check if run.py exists
if [ ! -f "run.py" ]; then
    echo "ERROR: run.py not found"
    echo "This script must be run from the project root directory"
    exit 1
fi

# Verify dependencies
echo ""
echo "Checking dependencies..."
python3 -c "import pandas; print('  ✓ pandas')" || { echo "ERROR: pandas not installed"; exit 1; }
python3 -c "import numpy; print('  ✓ numpy')" || { echo "ERROR: numpy not installed"; exit 1; }
python3 -c "import xgboost; print('  ✓ xgboost')" || { echo "ERROR: xgboost not installed"; exit 1; }
python3 -c "import prophet; print('  ✓ prophet')" || { echo "ERROR: prophet not installed"; exit 1; }
python3 -c "import streamlit; print('  ✓ streamlit')" || { echo "ERROR: streamlit not installed"; exit 1; }

# Run the main pipeline
echo ""
echo "Executing pipeline with: python3 run.py"
echo ""

python3 run.py

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "ERROR: Pipeline execution failed with exit code $exit_code"
    echo "Check the error messages above"
    exit $exit_code
fi

exit 0
