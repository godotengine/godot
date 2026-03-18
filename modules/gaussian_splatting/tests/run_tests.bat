@echo off
setlocal

set "ROOT_RUNNER=%~dp0..\..\..\run_tests.bat"

if not exist "%ROOT_RUNNER%" (
    echo ERROR: Root test runner not found at "%ROOT_RUNNER%".
    endlocal & exit /b 1
)

call "%ROOT_RUNNER%" %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
