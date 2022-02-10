_module_gdnaviation_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

_lib_module_gdnavigation = library('module_gdnavigation',
                                   _module_gdnaviation_srcs,
                                   dependencies: [DEP_GODOT_ALL, DEP_RECAST, DEP_RVO2])

DEP_MODULE_GDNAVIGATION = declare_dependency(link_with: _lib_module_gdnavigation)

MODULE_DEPENDENCIES += [DEP_MODULE_GDNAVIGATION]
MODULES_ENABLED += 'navigation'
