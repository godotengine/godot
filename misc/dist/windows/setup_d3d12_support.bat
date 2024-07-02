@echo off

set godot_version=4.3.dev
set dxc_version=v1.8.2403.2
set dxc_filename=dxc_2024_03_29.zip
set agility_sdk_version=1.610.4

echo.
echo --------------------------------------------------------------------------------
echo This script sets up Direct3D 12 support for Godot [96m%godot_version%[0m.
echo.
echo As Direct3D 12 support requires a proprietary library, it is not enabled by default.
echo Note that you don't need Direct3D 12 to use Godot, as Godot already comes with
echo Vulkan and OpenGL support out of the box.
echo.
echo This script will download and install the required libraries for Direct3D 12 support.
echo The libraries will also be installed in the export templates folder, so they can be
echo copied on export when a project is configured to use Direct3D 12.
echo The total download size is about 65 MB.
echo.
echo [1mBy installing DirectX Shader Compiler, you accept the terms of the Microsoft EULA:[0m
echo.
echo     TODO
echo.
echo [90mTip: Use the -y or /y command line argument to automatically accept the EULA.[0m
echo --------------------------------------------------------------------------------
echo.

echo [94mDo you accept the Microsoft EULA linked above? (y/n)[0m
if "%1" == "-y" set eula_accepted_cli=1
if "%1" == "/y" set eula_accepted_cli=1
if "%eula_accepted_cli%" == "1" (
    echo EULA accepted via command line argument.
    goto yes
)
set /p eula_accepted="> " %=%
if /i "%eula_accepted%" == "y" goto yes
if /i "%eula_accepted%" == "yes" goto yes
goto no

:yes
echo.
echo [1m[1/2] Downloading DirectX Shader Compiler release %dxc_version%...[0m
curl.exe -L https://github.com/microsoft/DirectXShaderCompiler/releases/download/%dxc_version%/%dxc_filename% -o "%TEMP%\dxc.zip"
echo Download completed.
echo Extracting DirectX Shader Compiler...
call :unzipfile "%TEMP%\dxc" "%TEMP%\dxc.zip"
del "%TEMP%\dxc.zip"
mkdir "%APPDATA%\Godot" 2>nul
mkdir "%APPDATA%\Godot\export_templates" 2>nul
mkdir "%APPDATA%\Godot\export_templates\%godot_version%" 2>nul
mkdir x64 2>nul
mkdir arm64 2>nul
mkdir x86 2>nul
copy "%TEMP%\dxc\bin\x64\dxil.dll" %APPDATA%\Godot\export_templates\%godot_version%\dxil.x64.dll >nul
move "%TEMP%\dxc\bin\x64\dxil.dll" x64\dxil.dll >nul
copy "%TEMP%\dxc\bin\arm64\dxil.dll" %APPDATA%\Godot\export_templates\%godot_version%\dxil.arm64.dll >nul
move "%TEMP%\dxc\bin\arm64\dxil.dll" arm64\dxil.dll >nul
copy "%TEMP%\dxc\bin\x86\dxil.dll" %APPDATA%\Godot\export_templates\%godot_version%\dxil.x86.dll >nul
move "%TEMP%\dxc\bin\x86\dxil.dll" x86\dxil.dll >nul
del /s /q "%TEMP%\dxc" >nul 2>&1
echo Done installing `dxil.dll` from DirectX Shader Compiler.

echo.
echo [1m[2/2] Downloading DirectX 12 Agility SDK release %agility_sdk_version%...[0m
curl.exe -L https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/%agility_sdk_version% -o "%TEMP%\agility_sdk.zip"
echo Download completed.
echo Extracting DirectX 12 Agility SDK...
call :unzipfile "%TEMP%\agility_sdk" "%TEMP%\agility_sdk.zip"
del "%TEMP%\agility_sdk.zip"
copy "%TEMP%\agility_sdk\build\native\bin\x64\D3D12Core.dll" %APPDATA%\Godot\export_templates\%godot_version%\D3D12Core.x64.dll >nul
move "%TEMP%\agility_sdk\build\native\bin\x64\D3D12Core.dll" x64\D3D12Core.dll >nul
copy "%TEMP%\agility_sdk\build\native\bin\arm64\D3D12Core.dll" %APPDATA%\Godot\export_templates\%godot_version%\D3D12Core.arm64.dll >nul
move "%TEMP%\agility_sdk\build\native\bin\arm64\D3D12Core.dll" arm64\D3D12Core.dll >nul
copy "%TEMP%\agility_sdk\build\native\bin\win32\D3D12Core.dll" %APPDATA%\Godot\export_templates\%godot_version%\D3D12Core.x86.dll >nul
move "%TEMP%\agility_sdk\build\native\bin\win32\D3D12Core.dll" x86\D3D12Core.dll >nul
copy "%TEMP%\agility_sdk\build\native\bin\x64\d3d12SDKLayers.dll" %APPDATA%\Godot\export_templates\%godot_version%\d3d12SDKLayers.x64.dll >nul
move "%TEMP%\agility_sdk\build\native\bin\x64\d3d12SDKLayers.dll" x64\d3d12SDKLayers.dll >nul
copy "%TEMP%\agility_sdk\build\native\bin\arm64\d3d12SDKLayers.dll" %APPDATA%\Godot\export_templates\%godot_version%\d3d12SDKLayers.arm64.dll >nul
move "%TEMP%\agility_sdk\build\native\bin\arm64\d3d12SDKLayers.dll" arm64\d3d12SDKLayers.dll >nul
copy "%TEMP%\agility_sdk\build\native\bin\win32\d3d12SDKLayers.dll" %APPDATA%\Godot\export_templates\%godot_version%\d3d12SDKLayers.x86.dll >nul
move "%TEMP%\agility_sdk\build\native\bin\win32\d3d12SDKLayers.dll" x86\d3d12SDKLayers.dll >nul
del /s /q "%TEMP%\agility_sdk" >nul 2>&1
echo Done installing Agility SDK libraries.

echo.
echo --------------------------------------------------------------------------------
echo [92mSuccess! Direct3D 12 support is now enabled for this Godot installation (%godot_version%).[0m
echo You can now choose the `d3d12` rendering driver project setting in any project,
echo or run a project with the `--rendering-driver d3d12` command line argument
echo to override the rendering driver for a single session.
echo.
echo Direct3D 12 libraries will automatically be copied along with a project exported
echo for Windows if the project is configured to use Direct3D 12, or if the
echo `application/export_d3d12` option is enabled in the Windows export preset
echo in a given project.
echo.
echo [1mAfter updating Godot to a newer version, remember to run the Direct3D 12[0m
echo [1msetup script that comes with it to install the Direct3D 12 libraries[0m
echo [1mcorresponding to the new Godot version. Each Godot version stores
echo [1mDirect3D 12 libraries in a version-specific folder to be copied when exporting.[0m
echo --------------------------------------------------------------------------------
pause
exit /b 0

:unzipfile <extractTo> <newZipFile>
set vbs="%TEMP%\_.vbs"
if exist %vbs% del /f /q %vbs%
>%vbs%  echo Set fso = CreateObject("Scripting.FileSystemObject")
>>%vbs% echo If NOT fso.FolderExists(%1) Then
>>%vbs% echo fso.CreateFolder(%1)
>>%vbs% echo End If
>>%vbs% echo Set objShell = CreateObject("Shell.Application")
>>%vbs% echo Set filesInZip=objShell.NameSpace(%2).items
>>%vbs% echo objShell.NameSpace(%1).CopyHere(filesInZip)
>>%vbs% echo Set fso = Nothing
>>%vbs% echo Set objShell = Nothing
cscript //nologo %vbs%
if exist %vbs% del /f /q %vbs%
goto :eof

:no
echo.
echo [91mYou must accept the Microsoft EULA to set up Direct3D 12 support. Aborting.[0m
pause
exit /b 1
