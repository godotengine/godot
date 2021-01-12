_module_gdnaviation_srcs = files([
    'gd_navigation_server.cpp',
    'nav_map.cpp',
    'nav_region.cpp',
    'navigation_mesh_editor_plugin.cpp',
    'navigation_mesh_generator.cpp',
    'register_types.cpp',
    'rvo_agent.cpp',
])

_lib_module_gdnavigation = library('module_gdnavigation', _module_gdnaviation_srcs, dependencies: [DEP_GODOT_ALL, DEP_RECAST, DEP_RVO2])

DEP_MODULE_GDNAVIGATION = declare_dependency(link_with: _lib_module_gdnavigation)

MODULE_DEPENDENCIES += [DEP_MODULE_GDNAVIGATION]