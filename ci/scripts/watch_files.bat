@echo off
setlocal enabledelayedexpansion

echo ========================================
echo GodotGS File Watcher
echo ========================================

cd /d "%~dp0..\.."

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ and add to PATH.
    pause
    exit /b 1
)

:: Create virtual environment if needed
if not exist "ci\.venv" (
    echo Creating virtual environment...
    python -m venv ci\.venv
)

:: Activate virtual environment
call "ci\.venv\Scripts\activate.bat"

:: Install/update requirements
pip install -r ci\requirements.txt

:: Run the file watcher
echo.
echo Starting file watcher...
echo Press Ctrl+C to stop.
echo.
python ci\watch.py