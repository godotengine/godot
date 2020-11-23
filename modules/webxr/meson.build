_module_webxr_srcs = files([
    'register_types.cpp',
    'webxr_interface_js.cpp',
    'webxr_interface.cpp'
])

if PLATFORM == 'javascript'
    error('js not setup yet')
    # env.AddJSLibraries(["native/library_godot_webxr.js"])
    # env.AddJSExterns(["native/webxr.externs.js"])
endif

lib_module_webxr = library('module_webxr', _module_webxr_srcs, dependencies: [DEP_GODOT_ALL])

DEP_MODULE_WEBXR = declare_dependency(link_with: lib_module_webxr)

MODULE_DEPENDENCIES += DEP_MODULE_WEBXR