_module_bmp_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

_lib_module_bmp = library('module_bmp', _module_bmp_srcs,
    dependencies: godotcore_dep,
)

DEP_MODULE_BMP = declare_dependency(link_with: _lib_module_bmp)

MODULE_DEPENDENCIES += DEP_MODULE_BMP
MODULES_ENABLED += 'bmp'
