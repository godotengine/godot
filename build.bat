@echo off
chcp 65001 >nul 2>&1
title Godot Engine Build

set "GODOT_BIN=bin\godot.windows.editor.x86_64.mono.exe"
set "SCONS_BASE=p=windows target=editor module_mono_enabled=yes accesskit=no d3d12=no angle=no"
set "NUPKGS_DIR=bin\GodotSharp\Tools\nupkgs"

:menu
echo.
echo ============================================
echo        Godot Engine 4.7 Build
echo ============================================
echo.
echo   1. Debug          (debug_symbols, optimize=debug)
echo   2. RelWithDebInfo (debug_symbols, optimize=speed)
echo   3. Release        (optimize=speed)
echo   4. Full Debug      (engine + C# glue + assemblies)
echo   5. Full RelWithDeb (engine + C# glue + assemblies)
echo   6. Full Release    (engine + C# glue + assemblies)
echo   7. C# Only         (glue + assemblies)
echo   8. Install Deps    (accesskit, d3d12 SDK)
echo   9. Clean
echo   10. Exit
echo.
set /p choice=Please select [1-10]:

if "%choice%"=="1" goto debug
if "%choice%"=="2" goto relwithdebinfo
if "%choice%"=="3" goto release
if "%choice%"=="4" goto full_debug
if "%choice%"=="5" goto full_relwithdeb
if "%choice%"=="6" goto full_release
if "%choice%"=="7" goto cs_only
if "%choice%"=="8" goto install_deps
if "%choice%"=="9" goto clean
if "%choice%"=="10" goto end
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

:relwithdebinfo
echo.
echo [RELWITHDEBINFO] Starting build...
echo.
scons %SCONS_BASE% debug_symbols=true optimize=speed
echo.
echo [RELWITHDEBINFO] Build finished with exit code: %ERRORLEVEL%
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
echo [FULL DEBUG] Step 1/4: Building engine...
echo.
scons %SCONS_BASE% debug_symbols=true optimize=debug
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL DEBUG] Step 2/4: Generating C# glue...
echo.
if not exist "%GODOT_BIN%" (
    echo Error: %GODOT_BIN% not found!
    goto buildfail
)
"%GODOT_BIN%" --generate-mono-glue ./modules/mono/glue
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL DEBUG] Step 3/4: Building C# assemblies...
echo.
python ./modules/mono/build_scripts/build_assemblies.py --godot-output-dir ./bin --push-nupkgs-local ./bin/GodotSharp/Tools/nupkgs
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL DEBUG] Step 4/4: Registering NuGet source...
echo.
dotnet nuget add source "%CD%\%NUPKGS_DIR%" -n GodotLocal --configfile "%APPDATA%\NuGet\NuGet.Config" 2>nul || echo NuGet source may already exist, skipping.
echo.
echo [FULL DEBUG] Build finished with exit code: %ERRORLEVEL%
pause
goto menu

:full_relwithdeb
echo.
echo [FULL RELWITHDEB] Step 1/4: Building engine...
echo.
scons %SCONS_BASE% debug_symbols=true optimize=speed
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL RELWITHDEB] Step 2/4: Generating C# glue...
echo.
if not exist "%GODOT_BIN%" (
    echo Error: %GODOT_BIN% not found!
    goto buildfail
)
"%GODOT_BIN%" --generate-mono-glue ./modules/mono/glue
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL RELWITHDEB] Step 3/4: Building C# assemblies...
echo.
python ./modules/mono/build_scripts/build_assemblies.py --godot-output-dir ./bin --push-nupkgs-local ./bin/GodotSharp/Tools/nupkgs
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL RELWITHDEB] Step 4/4: Registering NuGet source...
echo.
dotnet nuget add source "%CD%\%NUPKGS_DIR%" -n GodotLocal --configfile "%APPDATA%\NuGet\NuGet.Config" 2>nul || echo NuGet source may already exist, skipping.
echo.
echo [FULL RELWITHDEB] Build finished with exit code: %ERRORLEVEL%
pause
goto menu

:full_release
echo.
echo [FULL RELEASE] Step 1/4: Building engine...
echo.
scons %SCONS_BASE% optimize=speed
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL RELEASE] Step 2/4: Generating C# glue...
echo.
if not exist "%GODOT_BIN%" (
    echo Error: %GODOT_BIN% not found!
    goto buildfail
)
"%GODOT_BIN%" --generate-mono-glue ./modules/mono/glue
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL RELEASE] Step 3/4: Building C# assemblies...
echo.
python ./modules/mono/build_scripts/build_assemblies.py --godot-output-dir ./bin --push-nupkgs-local ./bin/GodotSharp/Tools/nupkgs
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [FULL RELEASE] Step 4/4: Registering NuGet source...
echo.
dotnet nuget add source "%CD%\%NUPKGS_DIR%" -n GodotLocal --configfile "%APPDATA%\NuGet\NuGet.Config" 2>nul || echo NuGet source may already exist, skipping.
echo.
echo [FULL RELEASE] Build finished with exit code: %ERRORLEVEL%
pause
goto menu

:: ============================================
::  C# only (glue + assemblies)
:: ============================================

:cs_only
echo.
echo [C# ONLY] Step 1/3: Generating C# glue...
echo.
if not exist "%GODOT_BIN%" (
    echo Error: %GODOT_BIN% not found! Build the engine first.
    pause
    goto menu
)
"%GODOT_BIN%" --generate-mono-glue ./modules/mono/glue
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [C# ONLY] Step 2/3: Building C# assemblies...
echo.
python ./modules/mono/build_scripts/build_assemblies.py --godot-output-dir ./bin --push-nupkgs-local ./bin/GodotSharp/Tools/nupkgs
if %ERRORLEVEL% neq 0 goto buildfail
echo.
echo [C# ONLY] Step 3/3: Registering NuGet source...
echo.
dotnet nuget add source "%CD%\%NUPKGS_DIR%" -n GodotLocal --configfile "%APPDATA%\NuGet\NuGet.Config" 2>nul || echo NuGet source may already exist, skipping.
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
