_module_gridmap_srcs = files([
    'grid_map_editor_plugin.cpp',
    'grid_map.cpp',
    'register_types.cpp'
])

_lib_module_gridmap = library('module_gridmap', _module_gridmap_srcs, dependencies: [DEP_GODOT_ALL])

DEP_MODULE_GRIDMAP = declare_dependency(link_with: _lib_module_gridmap)

MODULE_DEPENDENCIES += DEP_MODULE_GRIDMAP