@echo off
setlocal

cd /d "%~dp0\..\.."

echo ========================================
echo Legacy Phase 0 Verification Redirect
echo ========================================
echo.
echo This helper now runs the tests\ci baseline QA pipeline.
echo Historical Phase 0 assets are available in tests\archive\ for context.
echo.

set "REQUESTED_BIN=%~1"
if not "%REQUESTED_BIN%"=="" (
    set "GODOT_BINARY=%REQUESTED_BIN%"
) else if defined GODOT_BINARY (
    set "REQUESTED_BIN=%GODOT_BINARY%"
) else (
    set "REQUESTED_BIN=bin\godot.windows.editor.x86_64.exe"
)

set "GODOT_BINARY=%REQUESTED_BIN%"

if exist "%GODOT_BINARY%" (
    echo Using Godot binary: %GODOT_BINARY%
) else (
    echo WARNING: Could not find Godot binary at "%GODOT_BINARY%".
    echo Provide the path as the first argument or pre-set the GODOT_BINARY environment variable.
)

echo.
echo Running baseline QA suite from tests\ci\ ...
python tests\ci\run_baseline_qa.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo ✅ tests\ci baseline QA completed successfully.
) else (
    echo ❌ tests\ci baseline QA reported failures (exit code %EXIT_CODE%).
)

echo.
endlocal & exit /b %EXIT_CODE%
