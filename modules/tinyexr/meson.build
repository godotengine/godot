_module_tinyexr_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
lib_module_tinyexr = library('module_tinyexr', _module_tinyexr_srcs, dependencies: [godotcore_dep, DEP_TINYEXR, DEP_ZLIB])

DEP_MODULE_TINYEXR = declare_dependency(link_with: lib_module_tinyexr)

MODULE_DEPENDENCIES += DEP_MODULE_TINYEXR
MODULES_ENABLED += 'tinyexr'
