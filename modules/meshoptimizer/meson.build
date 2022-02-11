_module_meshoptimizer_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

lib_module_meshoptimizer = library('module_meshoptimizer',
                                   _module_meshoptimizer_srcs,
                                   dependencies: [godotcore_dep, DEP_MESHOPTIMIZER])

DEP_MODULE_MESHOPTIMIZER = declare_dependency(link_with: lib_module_meshoptimizer)

MODULE_DEPENDENCIES += DEP_MODULE_MESHOPTIMIZER
MODULES_ENABLED += 'meshoptimizer'
