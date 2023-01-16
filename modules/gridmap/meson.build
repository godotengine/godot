_module_gridmap_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
_lib_module_gridmap = library('module_gridmap',
    _module_gridmap_srcs,
    'editor/grid_map_editor_plugin.cpp',
    include_directories: '../..',
    dependencies: [godotcore_dep, DEP_ZLIB])

DEP_MODULE_GRIDMAP = declare_dependency(link_with: _lib_module_gridmap)

MODULE_DEPENDENCIES += DEP_MODULE_GRIDMAP
MODULES_ENABLED += 'gridmap'
