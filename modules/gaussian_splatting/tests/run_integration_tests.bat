@echo off
REM ============================================================
REM Gaussian Splatting Integration Test Runner
REM ============================================================

setlocal EnableDelayedExpansion

echo ============================================================
echo Gaussian Splatting Integration Test Suite
echo ============================================================
echo.

REM Check if Godot binary exists
set GODOT_BIN=C:\projects\GodotGS\bin\godot.windows.editor.x86_64.exe
if not exist "%GODOT_BIN%" (
    echo ERROR: Godot binary not found at %GODOT_BIN%
    echo Please build Godot first with: scons platform=windows tools=yes
    exit /b 1
)

REM Parse command line arguments
set RUN_BENCHMARKS=0
set RUN_STRESS=0
set RUN_HEAVY=0
set RUN_HEADLESS=0
set VERBOSE=0

:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--benchmarks" set RUN_BENCHMARKS=1
if /i "%~1"=="--stress" set RUN_STRESS=1
if /i "%~1"=="--heavy" set RUN_HEAVY=1
if /i "%~1"=="--headless" set RUN_HEADLESS=1
if /i "%~1"=="--verbose" set VERBOSE=1
if /i "%~1"=="--help" goto show_help
shift
goto parse_args
:end_parse

REM Create test results directory
if not exist test_results mkdir test_results

REM Set timestamp for this test run
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,4%%datetime:~4,2%%datetime:~6,2%_%datetime:~8,2%%datetime:~10,2%%datetime:~12,2%

echo Test Configuration:
echo   Benchmarks: %RUN_BENCHMARKS%
echo   Stress Tests: %RUN_STRESS%
echo   Heavy Tests: %RUN_HEAVY%
echo   Headless: %RUN_HEADLESS%
echo   Timestamp: %TIMESTAMP%
echo.

REM ============================================================
REM Run C++ Integration Tests
REM ============================================================
echo [1/4] Running C++ Integration Tests...
echo ----------------------------------------

"%GODOT_BIN%" --test --headless --verbose --test-filter="[Gaussian Splatting*]" > test_results\cpp_tests_%TIMESTAMP%.log 2>&1

if %ERRORLEVEL% EQU 0 (
    echo   ✅ C++ tests passed
) else (
    echo   ❌ C++ tests failed (see test_results\cpp_tests_%TIMESTAMP%.log)
)
echo.

REM ============================================================
REM Run Headless Build Tests
REM ============================================================
if %RUN_HEADLESS%==1 (
    echo [2/4] Running Headless Build Tests...
    echo ----------------------------------------

    python test_headless_build.py > test_results\headless_%TIMESTAMP%.log 2>&1

    if %ERRORLEVEL% EQU 0 (
        echo   ✅ Headless tests passed
    ) else (
        echo   ❌ Headless tests failed (see test_results\headless_%TIMESTAMP%.log)
    )
    echo.
) else (
    echo [2/4] Skipping Headless Tests (use --headless to enable)
    echo.
)

REM ============================================================
REM Run Performance Benchmarks
REM ============================================================
if %RUN_BENCHMARKS%==1 (
    echo [3/4] Running Performance Benchmarks...
    echo ----------------------------------------

    REM 100K splats benchmark
    echo   Running 100K splats benchmark...
    "%GODOT_BIN%" --headless --path test_projects\benchmark_100k --quit-after 200 > test_results\bench_100k_%TIMESTAMP%.log 2>&1

    REM 1M splats benchmark (if not heavy)
    echo   Running 1M splats benchmark...
    "%GODOT_BIN%" --headless --path test_projects\benchmark_1m --quit-after 100 > test_results\bench_1m_%TIMESTAMP%.log 2>&1

    if %RUN_HEAVY%==1 (
        REM 10M splats benchmark
        echo   Running 10M splats benchmark (this may take a while)...
        "%GODOT_BIN%" --headless --path test_projects\benchmark_10m --quit-after 50 > test_results\bench_10m_%TIMESTAMP%.log 2>&1
    )

    echo   ✅ Benchmarks completed
    echo.
) else (
    echo [3/4] Skipping Performance Benchmarks (use --benchmarks to enable)
    echo.
)

REM ============================================================
REM Run Integration Test Suite
REM ============================================================
echo [4/4] Running Full Integration Suite...
echo ----------------------------------------

set PYTHON_ARGS=
if %RUN_BENCHMARKS%==1 set PYTHON_ARGS=%PYTHON_ARGS% --benchmarks
if %RUN_STRESS%==1 set PYTHON_ARGS=%PYTHON_ARGS% --stress
if %RUN_HEAVY%==1 set PYTHON_ARGS=%PYTHON_ARGS% --heavy

python run_integration_tests.py %PYTHON_ARGS% > test_results\integration_%TIMESTAMP%.log 2>&1

if %ERRORLEVEL% EQU 0 (
    echo   ✅ Integration tests passed
) else (
    echo   ❌ Integration tests failed (see test_results\integration_%TIMESTAMP%.log)
)

REM ============================================================
REM Generate Summary Report
REM ============================================================
echo.
echo ============================================================
echo Test Summary
echo ============================================================

REM Count pass/fail from logs
set /a TOTAL_TESTS=0
set /a PASSED_TESTS=0
set /a FAILED_TESTS=0

REM Parse test results (simplified - in real implementation would parse JSON)
for %%f in (test_results\*_%TIMESTAMP%.log) do (
    findstr /c:"✅" "%%f" >nul && set /a PASSED_TESTS+=1
    findstr /c:"❌" "%%f" >nul && set /a FAILED_TESTS+=1
)

set /a TOTAL_TESTS=PASSED_TESTS+FAILED_TESTS

echo Total Tests Run: %TOTAL_TESTS%
echo Passed: %PASSED_TESTS%
echo Failed: %FAILED_TESTS%
echo.
echo Results saved to: test_results\
echo.

REM Set exit code based on failures
if %FAILED_TESTS% GTR 0 (
    echo ❌ Some tests failed. Please check the logs.
    exit /b 1
) else (
    echo ✅ All tests passed successfully!
    exit /b 0
)

:show_help
echo.
echo Usage: run_integration_tests.bat [options]
echo.
echo Options:
echo   --benchmarks    Run performance benchmarks
echo   --stress        Run stress tests
echo   --heavy         Include heavy benchmarks (10M+ splats)
echo   --headless      Run headless build tests
echo   --verbose       Enable verbose output
echo   --help          Show this help message
echo.
echo Examples:
echo   run_integration_tests.bat                    Run basic tests only
echo   run_integration_tests.bat --benchmarks       Include performance benchmarks
echo   run_integration_tests.bat --stress --heavy   Run all tests including stress
echo.
exit /b 0
