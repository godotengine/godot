_module_inappstore_srcs = files([
    'in_app_store_module.cpp',
    'in_app_store.mm',
])

_module_inappstore_cpp_args = ['-fmodules', '-fcxx-modules']

# static lib for ios
_lib_module_inappstore = static_library('module_inappstore', _module_inappstore_srcs,
    dependencies: DEP_GODOT_ALL,
    cpp_args: _module_inappstore_cpp_args
)

DEP_MODULE_INAPPSTORE = declare_dependency(link_with: _lib_module_inappstore)

MODULE_DEPENDENCIES += DEP_MODULE_INAPPSTORE