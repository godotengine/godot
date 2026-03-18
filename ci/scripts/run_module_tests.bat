@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\..\.."

set "ERR=0"
set "PYTHON_EXE=python"
set "IN_GODOT_SOURCE=0"

echo ========================================
echo Quick Test Runner for Gaussian Splatting
echo ========================================
echo.

%PYTHON_EXE% --version >nul 2>&1
if errorlevel 1 (
    echo Python is required to run CI guards and module tests.
    set "ERR=1"
    goto :cleanup
)

echo Running renderer CI guards...
%PYTHON_EXE% tests\ci\run_module_tests.py --guard-only
if errorlevel 1 (
    echo Renderer CI guards failed!
    set "ERR=1"
    goto :cleanup
)

echo.
echo Building with tests enabled (incremental)...
pushd . >nul
set "IN_GODOT_SOURCE=1"
scons platform=windows tools=yes tests=yes -j4
if errorlevel 1 (
    echo Build failed!
    set "ERR=1"
    goto :cleanup
)

echo.
echo Running Gaussian Splatting tests...
echo ========================================
set "GODOT_BINARY=%CD%\bin\godot.windows.editor.x86_64.exe"
popd >nul
set "IN_GODOT_SOURCE=0"
%PYTHON_EXE% tests\ci\run_module_tests.py --godot-binary "%GODOT_BINARY%" --skip-render-guards
set TEST_ERRORLEVEL=!ERRORLEVEL!
if !TEST_ERRORLEVEL! NEQ 0 (
    echo.
    echo Gaussian Splatting tests failed!
    set "ERR=!TEST_ERRORLEVEL!"
    goto :cleanup
)

echo.
echo Test run complete!

:cleanup
if "%IN_GODOT_SOURCE%"=="1" popd >nul
endlocal & exit /b %ERR%
