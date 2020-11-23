_module_meshoptimizer_srcs = files([
    'register_types.cpp'
])

lib_module_meshoptimizer = library('module_meshoptimizer', _module_meshoptimizer_srcs, dependencies: [DEP_GODOT_ALL, DEP_MESHOPTIMIZER])

DEP_MODULE_MESHOPTIMIZER = declare_dependency(link_with: lib_module_meshoptimizer)

MODULE_DEPENDENCIES += DEP_MODULE_MESHOPTIMIZER