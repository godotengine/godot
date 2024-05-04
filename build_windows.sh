scons platform=windows target=template_debug arch=x86_32 tools=no build_feature_profile="kaetram.build"
scons platform=windows target=template_release arch=x86_32 tools=no build_feature_profile="kaetram.build"
scons platform=windows target=template_debug arch=x86_64 tools=no build_feature_profile="kaetram.build"
scons platform=windows target=template_release arch=x86_64 tools=no build_feature_profile="kaetram.build"

# Rename the templates and place them in the correct directory.
cp bin/godot.windows.template_debug.x86_32.console.exe bin/windows_debug_x86_32_console.exe
cp bin/godot.windows.template_debug.x86_32.exe bin/windows_debug_x86_32.exe
cp bin/godot.windows.template_release.x86_32.console.exe bin/windows_release_x86_32_console.exe
cp bin/godot.windows.template_release.x86_32.exe bin/windows_release_x86_32.exe

cp bin/godot.windows.template_debug.x86_64.console.exe bin/windows_debug_x86_64_console.exe
cp bin/godot.windows.template_debug.x86_64.exe bin/windows_debug_x86_64.exe
cp bin/godot.windows.template_release.x86_64.console.exe bin/windows_release_x86_64_console.exe
cp bin/godot.windows.template_release.x86_64.exe bin/windows_release_x86_64.exe

# Copy the Windows templates to the templates_windows directory.
mkdir -p bin/templates_windows

mv bin/windows_debug_x86_32_console.exe bin/templates_windows/windows_debug_x86_32_console.exe
mv bin/windows_debug_x86_32.exe bin/templates_windows/windows_debug_x86_32.exe
mv bin/windows_release_x86_32_console.exe bin/templates_windows/windows_release_x86_32_console.exe
mv bin/windows_release_x86_32.exe bin/templates_windows/windows_release_x86_32.exe

mv bin/windows_debug_x86_64_console.exe bin/templates_windows/windows_debug_x86_64_console.exe
mv bin/windows_debug_x86_64.exe bin/templates_windows/windows_debug_x86_64.exe
mv bin/windows_release_x86_64_console.exe bin/templates_windows/windows_release_x86_64_console.exe
mv bin/windows_release_x86_64.exe bin/templates_windows/windows_release_x86_64.exe