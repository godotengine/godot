_module_arkit_srcs = files([
    'arkit_interface.mm',
    'arkit_module.cpp',
    'arkit_session_delegate.mm'
])

# NOTE: should probably check the compiler here, but only iphone targets anyways...
_module_arkit_cpp_flags = ['-fmodules', '-fcxx-modules']

# force static lib for ios
_lib_module_arkit = static_library('module_arkit', _module_arkit_srcs, dependencies: DEP_GODOT_ALL, cpp_flags: _module_arkit_cpp_flags)

DEP_MODULE_ARKIT = declare_dependency(link_with: _lib_module_arkit)

MODULE_DEPENDENCIES += DEP_MODULE_ARKIT