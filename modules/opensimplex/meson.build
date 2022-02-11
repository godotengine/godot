_module_opensimplex_srcs = files('../../thirdparty/misc/open-simplex-noise.c')
_module_opensimplex_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

lib_module_opensimplex = library('module_opensimplex', _module_opensimplex_srcs,
    dependencies: [godotcore_dep]
)

DEP_MODULE_OPENSIMPLEX = declare_dependency(link_with: lib_module_opensimplex)

MODULE_DEPENDENCIES += DEP_MODULE_OPENSIMPLEX
MODULES_ENABLED += 'opensimplex'
