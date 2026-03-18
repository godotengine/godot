@echo off
setlocal

cd /d "%~dp0\..\.."
set "TEST_ERRORLEVEL=0"
set "GODOT_BINARY=bin\godot.windows.editor.x86_64.console.exe"
set "TEST_PROJECT=tests\examples\godot\test_project"
set "TEST_SCENE=%TEST_PROJECT%\hello_splat_visual.tscn"

echo ====================================
echo HELLO SPLAT VISUAL TEST
echo ====================================
echo.

if not exist "%GODOT_BINARY%" (
    echo ERROR: Missing Godot console binary: %GODOT_BINARY%
    set "TEST_ERRORLEVEL=1"
    goto :done
)

if not exist "%TEST_PROJECT%" (
    echo ERROR: Missing test project directory: %TEST_PROJECT%
    set "TEST_ERRORLEVEL=1"
    goto :done
)

if not exist "%TEST_SCENE%" (
    echo ERROR: Missing test scene: %TEST_SCENE%
    set "TEST_ERRORLEVEL=1"
    goto :done
)

echo Running Godot with test project...
pushd . >nul
bin\godot.windows.editor.x86_64.console.exe --path tests\examples\godot\test_project --scene hello_splat_visual.tscn 2>&1 ^
    | findstr /i "hello splat gaussian error warning"
set TEST_ERRORLEVEL=%ERRORLEVEL%
popd >nul

if %TEST_ERRORLEVEL% NEQ 0 (
    echo.
    echo Godot exited with error level %TEST_ERRORLEVEL%.
)

:done
echo.
echo ====================================
echo Test complete. Check if you see:
echo 1. 100 colored dots/billboards
echo 2. Console messages about splat initialization
echo 3. No crash or errors
echo ====================================
pause
endlocal & exit /b %TEST_ERRORLEVEL%
