multiplayer_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
multiplayer_srcs += files(run_command(GLOB, 'editor/*.cpp', check: true).stdout().split('\n'))

lib_module_multiplayer = library('module_multiplayer',
    multiplayer_srcs,
    dependencies: [godotcore_dep, DEP_ASTCENC]
)

DEP_MODULE_MULTIPLAYER = declare_dependency(link_with: lib_module_multiplayer)

MODULE_DEPENDENCIES += DEP_MODULE_MULTIPLAYER
MODULES_ENABLED += 'multiplayer'
