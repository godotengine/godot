_module_vhacd_srcs = files([
    'register_types.cpp'
])

lib_module_vhacd = library('module_vhacd', _module_vhacd_srcs, dependencies: [DEP_GODOT_ALL, DEP_VHACD])

DEP_MODULE_VHACD = declare_dependency(link_with: lib_module_vhacd)

MODULE_DEPENDENCIES += DEP_MODULE_VHACD