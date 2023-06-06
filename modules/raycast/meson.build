lib_module_raycast = library('module_raycast',
    files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n')),
    dependencies: [godotcore_dep, embree_dep]
)

DEP_MODULE_RAYCAST = declare_dependency(link_with: lib_module_raycast)

MODULE_DEPENDENCIES += DEP_MODULE_RAYCAST
MODULES_ENABLED += 'raycast'
