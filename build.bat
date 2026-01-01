@echo off
REM Quick build script for Godot AI Studio on Windows
REM Run this from the Godot source root directory (where SConstruct is located)

echo ========================================
echo   Building Godot AI Studio (Windows)
echo ========================================
echo.

REM Check if we're in the correct directory
if not exist "SConstruct" (
    echo ERROR: SConstruct file not found!
    echo.
    echo Please run this script from the Godot source root directory.
    echo The SConstruct file should be in the current directory.
    echo.
    echo Current directory: %CD%
    echo.
    echo Expected location: D:\AIGODOT\godot\ (or wherever your Godot source is)
    pause
    exit /b 1
)

echo Found SConstruct - we're in the correct directory!
echo Building from: %CD%
echo.

REM Check if scons is available
where scons >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: SCons not found!
    echo Please install: pip install scons
    echo.
    pause
    exit /b 1
)

echo Starting build...
echo This may take 30-60 minutes on first build...
echo.

REM Clean build cache if there were previous errors (optional - uncomment if needed)
REM scons --clean

REM Build with production settings
REM Use -j8 for parallel builds (adjust to your CPU cores)
scons platform=windows target=editor production=yes -j8

if %ERRORLEVEL% == 0 (
    echo.
    echo ========================================
    echo   Build Successful!
    echo ========================================
    echo.
    echo Executable location:
    echo   bin\godot.windows.editor.x86_64.exe
    echo.
    
    REM Optionally rename
    if exist bin\godot.windows.editor.x86_64.exe (
        copy /Y bin\godot.windows.editor.x86_64.exe bin\godot_ai_studio.exe >nul
        echo Also copied to:
        echo   bin\godot_ai_studio.exe
        echo.
    )
    
    echo You can now run the executable!
) else (
    echo.
    echo ========================================
    echo   Build Failed!
    echo ========================================
    echo.
    echo Check the error messages above.
    echo Common issues:
    echo   - Python/SCons not in PATH
    echo   - Visual Studio Build Tools not installed
    echo   - Missing dependencies
    echo.
    echo See modules\claude_ai\BUILD_WINDOWS_EXE.md for troubleshooting
)

pause

