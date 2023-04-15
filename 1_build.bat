"scons" p=windows target=editor arch="x86_64" debug_symbols="yes" builtin_embree="no" use_llvm="yes" optimize="speed_trace" vsproj="yes" compiledb="yes" openxr="no" opengl3="no" deprecated="no" module_arkit_enabled="no" module_mobile_vr_enabled="no" module_mono_enabled="no" module_webxr_enabled="no" module_webrtc_enabled="no" module_websocket_enabled="no" -j 64
 
bin/godot.windows.editor.dev.x86_64.mono.exe --generate-mono-glue modules/mono/glue
python "./modules/mono/build_scripts/build_assemblies.py" --godot-output-dir=./bin --godot-platform=windows

scons p=windows target=editor profile=./custom.py vsproj="yes" module_mono_enabled=yes module_arkit_enabled=no module_mobile_vr_enabled=no module_webxr_enabled=no debug_symbols="yes" -j 128

C:\ws\Godot\godot>scons p=windows target=editor profile=./custom.py vsproj="yes" module_mono_enabled=yes module_arkit_enabled=no module_mobile_vr_enabled=no module_webxr_enabled=no debug_symbols="yes" use_simd=avx2 opengl3=no openxr=no -j 128