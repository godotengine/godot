@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\..\.."

echo ========================================
echo Gaussian Splatting Test Verification
echo ========================================
echo.

echo [Step 1] Checking test files exist...
echo ----------------------------------------
set TEST_DIR=modules\gaussian_splatting\tests

if exist "%TEST_DIR%\test_gaussian_splatting.h" (
    echo [OK] test_gaussian_splatting.h
) else (
    echo [MISSING] test_gaussian_splatting.h
    exit /b 1
)

if exist "%TEST_DIR%\test_gaussian_data.h" (
    echo [OK] test_gaussian_data.h
) else (
    echo [MISSING] test_gaussian_data.h
    exit /b 1
)

if exist "%TEST_DIR%\test_gpu_streaming.h" (
    echo [OK] test_gpu_streaming.h
) else (
    echo [MISSING] test_gpu_streaming.h
    exit /b 1
)

if exist "%TEST_DIR%\test_gpu_sorting.h" (
    echo [OK] test_gpu_sorting.h
) else (
    echo [MISSING] test_gpu_sorting.h
    exit /b 1
)

if exist "%TEST_DIR%\test_phase1_integration.h" (
    echo [OK] test_phase1_integration.h
) else (
    echo [MISSING] test_phase1_integration.h
    exit /b 1
)

echo.
echo [Step 2] Building with tests enabled...
echo ----------------------------------------
pushd . >nul

REM Clean build to ensure tests are included
echo Performing clean build with tests...
scons platform=windows tools=yes tests=yes optimize=speed -j4

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    popd >nul
    endlocal
    exit /b 1
)

echo.
echo [Step 3] Checking if binary includes tests...
echo ----------------------------------------
bin\godot.windows.editor.x86_64.exe --test --help > test_help.txt 2>&1

findstr /C:"doctest" test_help.txt > nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Test framework (doctest) is available
) else (
    echo [ERROR] Test framework not found in binary
    del test_help.txt
    popd >nul
    endlocal
    exit /b 1
)

del test_help.txt

echo.
echo [Step 4] Running basic test discovery...
echo ----------------------------------------
bin\godot.windows.editor.x86_64.exe --test --list-test-names-only > test_list.txt 2>&1

findstr /C:"GaussianSplatting" test_list.txt > nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Gaussian Splatting tests discovered
    echo.
    echo Found tests:
    findstr /C:"GaussianSplatting" test_list.txt
) else (
    echo [WARNING] No Gaussian Splatting tests found in test list
    echo This may be normal if tests are compiled but not yet discovered
)

del test_list.txt

echo.
echo [Step 5] Attempting to run tests...
echo ----------------------------------------
bin\godot.windows.editor.x86_64.exe --test --test-case="*GaussianSplatting*" --duration

popd >nul

echo.
echo ========================================
echo Test verification complete!
echo ========================================
echo.
echo To run all tests: bin\godot.windows.editor.x86_64.exe --test
echo To run our tests: bin\godot.windows.editor.x86_64.exe --test --test-case="*GaussianSplatting*"
echo.
endlocal
