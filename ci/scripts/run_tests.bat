@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\..\.."

set "ERR=0"

echo ========================================
echo Gaussian Splatting Module Test Runner
echo ========================================
echo.

echo [1/3] Building Godot with tests enabled...
pushd . >nul
call scons platform=windows tools=yes tests=yes optimize=speed -j4
if errorlevel 1 (
    echo.
    echo ERROR: Build failed! Check the build output above.
    set "ERR=1"
    goto :cleanup
)

echo.
echo [2/3] Build successful! Running tests...
echo ========================================
echo.
bin\godot.windows.editor.x86_64.exe --test --verbose
set ALL_TEST_ERRORLEVEL=!ERRORLEVEL!
if !ALL_TEST_ERRORLEVEL! NEQ 0 (
    echo.
    echo Some tests failed! Check the output above.
    set "ERR=!ALL_TEST_ERRORLEVEL!"
    goto :cleanup
)

echo.
echo [3/3] Running Gaussian Splatting specific tests...
echo ========================================
echo.
bin\godot.windows.editor.x86_64.exe --test --test-case="*GaussianSplatting*"
set MODULE_TEST_ERRORLEVEL=!ERRORLEVEL!
if !MODULE_TEST_ERRORLEVEL! NEQ 0 (
    echo.
    echo Gaussian Splatting tests failed!
    set "ERR=!MODULE_TEST_ERRORLEVEL!"
    goto :cleanup
)

echo.
echo ========================================
echo All tests passed successfully!
echo ========================================

:cleanup
popd >nul
endlocal & exit /b %ERR%
