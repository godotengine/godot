@echo off
setlocal

cd /d "%~dp0\..\.."

echo ====================================
echo Testing GPU Memory Streaming Build
echo ====================================

echo.
echo Step 1: Clean previous build artifacts
if exist bin\obj\modules\gaussian_splatting\*.obj del /q bin\obj\modules\gaussian_splatting\*.obj
if exist bin\obj\modules\gaussian_splatting\renderer\*.obj del /q bin\obj\modules\gaussian_splatting\renderer\*.obj

echo.
echo Step 2: Verify canonical module path
if not exist modules\gaussian_splatting (
    echo Missing canonical module path: modules\gaussian_splatting
    endlocal
    exit /b 1
)

echo.
echo Step 3: Build module only (quick test)
pushd . >nul
echo Building gaussian_splatting module...
scons platform=windows tools=yes target=editor -j4 --config=force

if %ERRORLEVEL% neq 0 (
    echo.
    echo ====================================
    echo BUILD FAILED!
    echo ====================================
    echo Check errors above for details
    popd >nul
    endlocal
    exit /b 1
)

echo.
echo ====================================
echo BUILD SUCCESSFUL!
echo ====================================
echo GPU Memory Streaming implementation compiled successfully
echo.
echo Next steps:
echo 1. Run full engine build: scons platform=windows tools=yes -j8
echo 2. Test in editor: bin\godot.windows.editor.x86_64.exe
echo 3. Monitor performance with: --gpu-profile
popd >nul

pause
endlocal
