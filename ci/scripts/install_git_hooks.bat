@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Installing Git Pre-commit Hook
echo ========================================

cd /d "%~dp0..\.."

:: Check if .git directory exists
if not exist ".git" (
    echo ERROR: Not in a git repository
    pause
    exit /b 1
)

:: Create hooks directory if it doesn't exist
if not exist ".git\hooks" (
    mkdir ".git\hooks"
)

:: Create the pre-commit hook
echo Creating pre-commit hook...
(
echo #!/bin/sh
echo # GodotGS Pre-commit Hook
echo # Runs quick validation before allowing commit
echo.
echo echo "Running GodotGS quick validation..."
echo.
echo # Change to project root
echo cd "$(git rev-parse --show-toplevel)"
echo.
echo # Run quick test
echo if command -v python ^>/dev/null 2^>^&1; then
echo     python ci/simple_pipeline.py --quick
echo elif command -v python3 ^>/dev/null 2^>^&1; then
echo     python3 ci/simple_pipeline.py --quick
echo else
echo     echo "ERROR: Python not found. Cannot run pre-commit validation."
echo     exit 1
echo fi
echo.
echo if [ $? -ne 0 ]; then
echo     echo "Pre-commit validation failed. Commit aborted."
echo     echo "Run 'ci/scripts/run_ci.bat' to see detailed results."
echo     exit 1
echo fi
echo.
echo echo "Pre-commit validation passed!"
) > ".git\hooks\pre-commit"

:: Make the hook executable (on Windows, this is less critical)
echo Pre-commit hook installed!
echo.
echo The hook will run quick validation before each commit.
echo To bypass the hook (not recommended), use: git commit --no-verify
echo.
pause