@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\..\.."

echo ===============================================
echo GPU Memory Streaming Implementation Verification
echo ===============================================
echo.

echo Checking implementation files...
echo.

set MISSING=0

if exist modules\gaussian_splatting\renderer\gpu_memory_stream.h (
    echo [OK] gpu_memory_stream.h
) else (
    echo [MISSING] gpu_memory_stream.h
    set MISSING=1
)

if exist modules\gaussian_splatting\renderer\gpu_memory_stream.cpp (
    echo [OK] gpu_memory_stream.cpp
) else (
    echo [MISSING] gpu_memory_stream.cpp
    set MISSING=1
)

if exist modules\gaussian_splatting\tests\test_gpu_streaming.cpp (
    echo [OK] test_gpu_streaming.cpp
) else (
    echo [MISSING] test_gpu_streaming.cpp
    set MISSING=1
)

if exist docs\gpu_memory_streaming.md (
    echo [OK] gpu_memory_streaming.md
) else (
    echo [MISSING] gpu_memory_streaming.md
    set MISSING=1
)

echo.
echo Checking modifications...
if exist modules\gaussian_splatting\renderer\gaussian_splat_renderer.h (
    findstr /C:"gpu_memory_stream.h" modules\gaussian_splatting\renderer\gaussian_splat_renderer.h >nul
    if %ERRORLEVEL% equ 0 (
        echo [OK] gaussian_splat_renderer.h includes gpu_memory_stream.h
    ) else (
        echo [ERROR] gaussian_splat_renderer.h missing include
        set MISSING=1
    )
) else (
    echo [MISSING] gaussian_splat_renderer.h
    set MISSING=1
)

if exist modules\gaussian_splatting\renderer\gaussian_splat_renderer.cpp (
    findstr /C:"memory_stream" modules\gaussian_splatting\renderer\gaussian_splat_renderer.cpp >nul
    if %ERRORLEVEL% equ 0 (
        echo [OK] gaussian_splat_renderer.cpp uses memory_stream
    ) else (
        echo [ERROR] gaussian_splat_renderer.cpp not using memory_stream
        set MISSING=1
    )
) else (
    echo [MISSING] gaussian_splat_renderer.cpp
    set MISSING=1
)

echo.
if %MISSING% equ 0 (
    echo ===============================================
    echo VERIFICATION PASSED!
    echo ===============================================
    echo All GPU Memory Streaming components are in place.
    echo.
    echo File Statistics:
    for %%F in (modules\gaussian_splatting\renderer\gpu_memory_stream.cpp) do echo gpu_memory_stream.cpp: %%~zF bytes
    for %%F in (modules\gaussian_splatting\renderer\gpu_memory_stream.h) do echo gpu_memory_stream.h: %%~zF bytes
    for %%F in (docs\gpu_memory_streaming.md) do echo Documentation: %%~zF bytes
    echo.
    echo Ready for compilation!
) else (
    echo ===============================================
    echo VERIFICATION FAILED!
    echo ===============================================
    echo Some components are missing or incorrectly configured.
    echo Please review the errors above.
)

echo.
pause
if %MISSING% equ 0 (
    endlocal & exit /b 0
) else (
    endlocal & exit /b 1
)
