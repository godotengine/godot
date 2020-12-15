_module_webxr_srcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))

DEP_MODULE_WEBXR = []
if PLATFORM == 'javascript'
    DEP_MODULE_WEBXR += cpp.find_library('library_godot_webxr.js', dirs: [meson.current_source_dir() / 'native'])

endif

lib_module_webxr = library('module_webxr', _module_webxr_srcs, dependencies: [DEP_GODOT_ALL])

DEP_MODULE_WEBXR += declare_dependency(link_with: lib_module_webxr)

MODULE_DEPENDENCIES += DEP_MODULE_WEBXR
