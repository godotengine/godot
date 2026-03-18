@echo off
setlocal

cd /d "%~dp0\..\.."
set "TEST_ERRORLEVEL=0"
set "PYTHON_EXE=python"
set "GODOT_BINARY=bin\godot.windows.editor.x86_64.exe"
set "TEST_PROJECT=tests\examples\godot\test_project"
set "TEST_SCENE=%TEST_PROJECT%\hello_splat_visual.tscn"

echo ====================================
echo HELLO SPLAT RENDERING TEST
echo ====================================
echo.

%PYTHON_EXE% --version >nul 2>&1
if errorlevel 1 (
    echo Python is required to run renderer CI guards.
    set "TEST_ERRORLEVEL=1"
    goto :done
)

echo Running renderer CI guards...
%PYTHON_EXE% tests\ci\run_module_tests.py --guard-only
if errorlevel 1 (
    echo.
    echo Renderer CI guards failed.
    set "TEST_ERRORLEVEL=1"
    goto :done
)

if not exist "%GODOT_BINARY%" (
    echo ERROR: Missing Godot editor binary: %GODOT_BINARY%
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

echo.
echo Starting Godot editor with test project...
echo.

pushd . >nul
bin\godot.windows.editor.x86_64.exe --path tests\examples\godot\test_project --scene hello_splat_visual.tscn
set TEST_ERRORLEVEL=%ERRORLEVEL%
popd >nul

if %TEST_ERRORLEVEL% NEQ 0 (
    echo.
    echo Godot exited with error level %TEST_ERRORLEVEL%.
)

:done
echo.
echo ====================================
echo Visual test complete!
echo.
echo You should see:
echo 1. 100 colored billboard quads in the scene
echo 2. Console messages about splat initialization
echo 3. No errors or crashes
echo.
echo If you don't see splats, check:
echo - Look for "GaussianSplatRenderer" in the scene tree
echo - Check if it has 100 MeshInstance3D children
echo - Verify the meshes are visible
echo ====================================
pause
endlocal & exit /b %TEST_ERRORLEVEL%
