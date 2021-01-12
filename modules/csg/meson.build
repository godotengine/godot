_module_csg_srcs = files([
    'csg_gizmos.cpp',
    'csg_shape.cpp',
    'csg.cpp',
    'register_types.cpp'
])

_lib_module_csg = library('module_csg', _module_csg_srcs, dependencies: DEP_GODOT_ALL)

DEP_MODULE_CSG = declare_dependency(link_with: _lib_module_csg)

MODULE_DEPENDENCIES += DEP_MODULE_CSG