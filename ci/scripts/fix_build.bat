@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo Godot Build System Fix and Rebuild Script
echo ===========================================
echo.

pushd "%~dp0\..\.." >nul

if not defined SCONS_PATH (
    set "SCONS_PATH=scons"
)

set "ERR=0"

:: Check for and remove corrupted object files
echo [1/5] Checking for corrupted object files...
set CORRUPTED=0
for /f "delims=" %%f in ('dir /s /b bin\obj\*.obj 2^>nul') do (
    for %%A in ("%%f") do (
        if %%~zA==0 (
            echo   Removing corrupted file: %%f
            del "%%f"
            set /a CORRUPTED+=1
        )
    )
)
if !CORRUPTED! GTR 0 (
    echo   Removed !CORRUPTED! corrupted files
) else (
    echo   No corrupted files found
)
echo.

:: Clean specific problematic modules if needed
echo [2/5] Cleaning problematic modules...
if exist bin\obj\modules\gdscript (
    echo   Cleaning gdscript module objects...
    rmdir /s /q bin\obj\modules\gdscript 2>nul
)
if exist bin\obj\modules\gaussian_splatting (
    echo   Cleaning gaussian_splatting module objects...
    rmdir /s /q bin\obj\modules\gaussian_splatting 2>nul
)
echo.

:: First attempt: Incremental build with moderate parallelism
echo [3/5] Attempting incremental build...
echo Running: %SCONS_PATH% platform=windows tools=yes -j4
call %SCONS_PATH% platform=windows tools=yes -j4

:: Check if build succeeded
if errorlevel 1 (
    echo.
    echo [WARNING] Incremental build failed. Attempting clean rebuild...
    echo.

    :: Clean build
    echo [4/5] Cleaning all build artifacts...
    call %SCONS_PATH% platform=windows tools=yes -c

    :: Try again with single-threaded build to avoid race conditions
    echo [5/5] Rebuilding with single thread to avoid corruption...
    call %SCONS_PATH% platform=windows tools=yes -j1

    if errorlevel 1 (
        echo.
        echo [ERROR] Build failed even with clean single-threaded build!
        echo Please check the error messages above.
        set "ERR=1"
        goto :cleanup
    )
)

echo.
echo [SUCCESS] Build completed successfully!
echo.

:: Verify the binary exists
if exist bin\godot.windows.editor.x86_64.exe (
    echo Binary verified: bin\godot.windows.editor.x86_64.exe
    dir bin\*.exe | findstr godot.windows.editor.x86_64.exe
    echo.

    :: Test the binary
    echo Testing binary...
    bin\godot.windows.editor.x86_64.exe --version
    echo.
    echo Build and verification complete!
) else (
    echo [ERROR] Binary not found after successful build!
    set "ERR=1"
)

:cleanup
popd >nul
endlocal & exit /b %ERR%
