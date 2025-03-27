scons -j4 platform=windows target=editor module_mono_enabled=yes
.\bin\godot.windows.editor.x86_64.mono.exe --headless --generate-mono-glue modules/mono/glue
python ./modules/mono/build_scripts/build_assemblies.py --godot-output-dir=./bin --godot-platform=windows
scons platform=windows target=template_release tools=no -j4
scons platform=android target=template_release tools=no -j4
scons platform=linux target=template_release tools=no -j4