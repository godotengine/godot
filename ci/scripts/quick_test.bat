@echo off
setlocal enabledelayedexpansion

echo ========================================
echo GodotGS Quick Test (for pre-commit)
echo ========================================

cd /d "%~dp0..\.."

:: Activate virtual environment if it exists
if exist "ci\.venv\Scripts\activate.bat" (
    call "ci\.venv\Scripts\activate.bat"
) else (
    echo Warning: Virtual environment not found. Using system Python.
)

:: Run quick validation
python ci\simple_pipeline.py --quick

if errorlevel 1 (
    echo.
    echo ========================================
    echo QUICK TEST FAILED!
    echo ========================================
    exit /b 1
) else (
    echo.
    echo ========================================
    echo QUICK TEST PASSED!
    echo ========================================
    exit /b 0
)