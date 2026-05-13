@echo off
chcp 65001 >nul 2>&1
title Godot Engine Build

set "GODOT_BIN=bin\godot.windows.editor.x86_64.mono.exe"
set "SCONS_BASE=p=windows target=editor module_mono_enabled=yes accesskit=no d3d12=no angle=no"

:menu
echo.
echo ============================================
echo        Godot Engine 4.7 Build
echo ============================================
echo.
echo   1. Debug          (debug_symbols, optimize=debug)
echo   2. Release        (optimize=speed)
echo   3. Full Debug      (engine + C# glue + assemblies)
echo   4. Full Release    (engine + C# glue + assemblies)
echo   5. C# Only         (glue + assemblies)
echo   6. Install Deps    (accesskit, d3d12 SDK)
echo   7. Clean
echo   8. Exit
echo.
set /p choice=Please select [1-8]:

if "%choice%"=="1" goto debug
if "%choice%"=="2" goto release
if "%choice%"=="3" goto full_debug
if "%choice%"=="4" goto full_release
if "%choice%"=="5" goto cs_only
if "%choice%"=="6" goto install_deps
if "%choice%"=="7" goto clean
if "%choice%"=="8" goto end
echo Invalid choice.
goto menu

:: ============================================
::  Engine only
:: ============================================

:debug
echo.
echo [DEBUG] Starting build...
echo.
scons %SCONS_BASE% debug_symbols=true optimize=debug
echo.
echo [DEBUG] Build finished with exit code: %ERRORLEVEL%
pause
goto menu

:release
echo.
echo [RELEASE] Starting build...
echo.
scons %SCONS_BASE% optimize=speed
echo.
echo [RELEASE] Build finished with exit code: %ERRORLEVEL%
pause
goto menu

:: ============================================
::  Full build: engine + C#
:: ============================================

:full_debug
echo.
echo [FULL DEBUG] Step 1/3: Building engine...
echo.
scons %SCONS_BASE% debug_symbols=true optimize=debug
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL DEBUG] Step 2/3: Generating C# glue...
echo.
if not exist "%GODOT_BIN%" (
    echo Error: %GODOT_BIN% not found!
    goto buildfail
)
"%GODOT_BIN%" --generate-mono-glue ./modules/mono/glue
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL DEBUG] Step 3/3: Building C# assemblies...
echo.
python ./modules/mono/build_scripts/build_assemblies.py --godot-output-dir ./bin
echo.
echo [FULL DEBUG] Build finished with exit code: %ERRORLEVEL%
pause
goto menu

:full_release
echo.
echo [FULL RELEASE] Step 1/3: Building engine...
echo.
scons %SCONS_BASE% optimize=speed
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL RELEASE] Step 2/3: Generating C# glue...
echo.
if not exist "%GODOT_BIN%" (
    echo Error: %GODOT_BIN% not found!
    goto buildfail
)
"%GODOT_BIN%" --generate-mono-glue ./modules/mono/glue
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL RELEASE] Step 3/3: Building C# assemblies...
echo.
python ./modules/mono/build_scripts/build_assemblies.py --godot-output-dir ./bin
echo.
echo [FULL RELEASE] Build finished with exit code: %ERRORLEVEL%
pause
goto menu

:: ============================================
::  C# only (glue + assemblies)
:: ============================================

:cs_only
echo.
echo [C# ONLY] Step 1/2: Generating C# glue...
echo.
if not exist "%GODOT_BIN%" (
    echo Error: %GODOT_BIN% not found! Build the engine first.
    pause
    goto menu
)
"%GODOT_BIN%" --generate-mono-glue ./modules/mono/glue
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [C# ONLY] Step 2/2: Building C# assemblies...
echo.
python ./modules/mono/build_scripts/build_assemblies.py --godot-output-dir ./bin
echo.
echo [C# ONLY] Build finished with exit code: %ERRORLEVEL%
pause
goto menu

:: ============================================
::  Install optional dependencies
:: ============================================

:install_deps
echo.
echo [INSTALL] Installing accesskit dependencies...
echo.
python misc\scripts\install_accesskit.py
if %ERRORLEVEL% neq 0 (
    echo accesskit install FAILED.
    pause
    goto menu
)
echo.
echo [INSTALL] Installing d3d12 SDK...
echo.
python misc\scripts\install_d3d12_sdk_windows.py
if %ERRORLEVEL% neq 0 (
    echo d3d12 SDK install FAILED.
    pause
    goto menu
)
echo.
echo [INSTALL] All dependencies installed.
echo   Now you can remove 'accesskit=no d3d12=no' from SCONS_BASE to enable them.
echo.
pause
goto menu

:: ============================================
::  Clean
:: ============================================

:clean
echo.
echo [CLEAN] Removing build artifacts...
echo.
if exist ".scons_cache" (
    echo Removing .scons_cache ...
    rmdir /s /q ".scons_cache"
)
if exist "bin" (
    echo Removing bin/ ...
    rmdir /s /q "bin"
)
echo.
echo [CLEAN] Done.
pause
goto menu

:: ============================================
::  Error & Exit
:: ============================================

:buildfail
echo.
echo Build FAILED with exit code: %ERRORLEVEL%
pause
goto menu

:end
