@echo off
setlocal enabledelayedexpansion

echo ========================================
echo GodotGS Local CI Pipeline
echo ========================================

cd /d "%~dp0..\.."

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ and add to PATH.
    pause
    exit /b 1
)

:: Install requirements if needed
if not exist "ci\.venv" (
    echo Creating virtual environment...
    python -m venv ci\.venv
)

:: Activate virtual environment
call "ci\.venv\Scripts\activate.bat"

:: Install/update requirements
pip install -r ci\requirements.txt

:: Run the pipeline
echo.
echo Running CI Pipeline...
echo.
python ci\simple_pipeline.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo CI PIPELINE FAILED!
    echo ========================================
    pause
    exit /b 1
) else (
    echo.
    echo ========================================
    echo CI PIPELINE PASSED!
    echo ========================================

    :: Open the latest report
    if exist "ci\reports\latest.html" (
        echo Opening latest report...
        start ci\reports\latest.html
    )

    pause
    exit /b 0
)