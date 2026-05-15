@echo off


:: Parse Args
:: ----------------------------------------------------------------------
set arg_target=editor
set arg_dev_build=yes

echo [Status] Parse Args
:parse
if "%1"=="" goto :after_parse
if "%1"=="-target" set arg_target="%2" & shift
shift
goto :parse

:after_parse

if not %arg_target%==editor set arg_dev_build=no

echo target: %arg_target%
echo dev_build: %arg_dev_build%


:: Setup nuget
:: ----------------------------------------------------------------------
echo [Status] Setup nuget
mkdir godot_nuget
dotnet nuget remove source godot_nuget
dotnet nuget add source %CD%/godot_nuget --name godot_nuget


:: Build
:: ----------------------------------------------------------------------
echo [Status] Build project
CALL scons platform=windows target=%arg_target% module_mono_enabled=yes dev_build=%arg_dev_build%
CALL ./bin/godot.windows.editor.dev.x86_64.mono --headless --generate-mono-glue modules/mono/glue
CALL python ./modules/mono/build_scripts/build_assemblies.py --godot-output-dir ./bin --push-nupkgs-local %CD%/godot_nuget --godot-platform=windows


:: Generate vsproj
:: ----------------------------------------------------------------------
if %arg_dev_build%==yes (
 echo [Status] Generate vsproj
 CALL scons platform=windows target=%arg_target% module_mono_enabled=yes vsproj=yes dev_build=yes
)

:end
pause