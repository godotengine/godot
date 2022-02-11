_module_cvtt_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
_lib_module_cvtt = library('module_cvtt', _module_cvtt_srcs, dependencies: [godotcore_dep, DEP_CVTT])

DEP_MODULE_CVTT = declare_dependency(link_with: _lib_module_cvtt)

MODULE_DEPENDENCIES += DEP_MODULE_CVTT
MODULES_ENABLED += 'cvtt'
