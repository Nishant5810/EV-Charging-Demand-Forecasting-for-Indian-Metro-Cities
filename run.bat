@echo off
REM ============================================================================
REM  EV CHARGING DEMAND FORECASTING - WINDOWS BATCH RUNNER
REM ============================================================================
REM
REM This script automatically runs the complete EV forecasting pipeline:
REM 1. Data Ingestion
REM 2. Feature Engineering  
REM 3. Model Training
REM 4. Dashboard Launch
REM
REM Usage: Double-click this file or run from command prompt
REM
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo  EV CHARGING DEMAND FORECASTING ^& GRID LOAD OPTIMIZER
echo ============================================================================
echo.
echo Starting complete automated pipeline...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    echo This script must be run from the project root directory
    pause
    exit /b 1
)

REM Check if run.py exists
if not exist "run.py" (
    echo ERROR: run.py not found
    echo This script must be run from the project root directory
    pause
    exit /b 1
)

REM Run the main pipeline
echo Executing pipeline with: python run.py
echo.
python run.py

if errorlevel 1 (
    echo.
    echo ERROR: Pipeline execution failed
    echo Check the error messages above
    pause
    exit /b 1
)

pause
