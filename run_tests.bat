@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "CANONICAL_RUNNER=%SCRIPT_DIR%ci\scripts\run_module_tests.bat"

if not exist "%CANONICAL_RUNNER%" (
    echo ERROR: Canonical module test runner not found at "%CANONICAL_RUNNER%".
    endlocal & exit /b 1
)

call "%CANONICAL_RUNNER%" %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
