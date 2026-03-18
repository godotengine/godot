@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo Smart Build System - Targeted Fix
echo ===========================================
echo.

pushd "%~dp0\..\.." >nul

if not defined SCONS_PATH (
    set "SCONS_PATH=scons"
)

set "ERR=0"

:: Step 1: Clean only main and modules object files (where linker issues occur)
echo [1/3] Cleaning linker-related objects...
if exist bin\obj\main (
    echo   Removing main objects...
    rmdir /s /q bin\obj\main
)
if exist bin\obj\modules (
    echo   Removing modules objects...
    rmdir /s /q bin\obj\modules
)
if exist bin\*.exe (
    echo   Removing old executable...
    del /f bin\*.exe
)
echo.

:: Step 2: Remove any zero-size object files
echo [2/3] Removing corrupted files...
set CORRUPTED=0
for /f "delims=" %%f in ('dir /s /b bin\obj\*.obj 2^>nul') do (
    for %%A in ("%%f") do (
        if %%~zA==0 (
            echo   Removing empty: %%~nxf
            del "%%f"
            set /a CORRUPTED+=1
        )
    )
)
echo   Cleaned !CORRUPTED! corrupted files
echo.

:: Step 3: Build with tests disabled and moderate parallelism
echo [3/3] Building Godot (tests disabled, 4 threads)...
echo Command: %SCONS_PATH% platform=windows tools=yes tests=no -j4
call %SCONS_PATH% platform=windows tools=yes tests=no -j4

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    echo Trying with reduced parallelism...
    call %SCONS_PATH% platform=windows tools=yes tests=no -j2

    if errorlevel 1 (
        echo.
        echo [CRITICAL] Build still failing. Manual intervention required.
        set "ERR=1"
        goto :cleanup
    )
)

echo.
echo ===========================================
echo Build Complete - Verifying...
echo ===========================================
echo.

:: Verify binary
if exist bin\godot.windows.editor.x86_64.exe (
    echo [SUCCESS] Binary found:
    dir bin\*.exe | findstr godot
    echo.
    echo Testing binary version...
    bin\godot.windows.editor.x86_64.exe --version
    echo.
    echo [SUCCESS] Build verified and working!
) else (
    echo [ERROR] Binary not found!
    echo Checking bin directory:
    dir bin\*.* | findstr /i exe
    set "ERR=1"
)

:cleanup
popd >nul
endlocal & exit /b %ERR%
