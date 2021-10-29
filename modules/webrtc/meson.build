_module_webrtc_srcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))
_module_webrtc_compile_args = []
_module_webrtc_deps = [DEP_GODOT_ALL]

DEP_MODULE_WEBRTC = []

if 'gdnative' in MODULES_ENABLED
    _module_webrtc_compile_args += '-DWEBRTC_GDNATIVE_ENABLED'
    _module_webrtc_deps += DEP_MODULE_GDNATIVE
endif

if PLATFORM == 'javascript'
    DEP_MODULE_WEBRTC += cpp.find_library('library_godot_webrtc.js', dirs: [meson.current_source_dir()])
endif

lib_module_webrtc = library('module_webrtc',
    _module_webrtc_srcs,
    dependencies: _module_webrtc_deps,
    cpp_args: _module_webrtc_compile_args)

DEP_MODULE_WEBRTC += declare_dependency(link_with: lib_module_webrtc)

MODULE_DEPENDENCIES += DEP_MODULE_WEBRTC
