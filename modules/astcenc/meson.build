astcmod_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

lib_module_astcenc = library('module_astcenc',
    astcmod_srcs,
    dependencies: [godotcore_dep, DEP_ASTCENC]
)

DEP_MODULE_ASTC = declare_dependency(link_with: lib_module_astcenc)

MODULE_DEPENDENCIES += DEP_MODULE_ASTC
MODULES_ENABLED += 'astcenc'
