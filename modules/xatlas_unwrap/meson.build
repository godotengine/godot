_module_xatlas_unwrap_srcs = files([
    'register_types.cpp'
])

lib_module_xatlas_unwrap = library('module_xatlas_unwrap', _module_xatlas_unwrap_srcs, dependencies: [DEP_GODOT_ALL, DEP_XATLAS])

DEP_MODULE_XATLAS_UNWARP = declare_dependency(link_with: lib_module_xatlas_unwrap)

MODULE_DEPENDENCIES += DEP_MODULE_XATLAS_UNWARP