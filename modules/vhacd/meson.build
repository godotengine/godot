_module_vhacd_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

lib_module_vhacd = library('module_vhacd', _module_vhacd_srcs, dependencies: [godotcore_dep, DEP_VHACD])

DEP_MODULE_VHACD = declare_dependency(link_with: lib_module_vhacd)

MODULE_DEPENDENCIES += DEP_MODULE_VHACD
MODULES_ENABLED += 'vhacd'
