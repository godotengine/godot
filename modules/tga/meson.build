_module_tga_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
lib_module_tga = library('module_tga', _module_tga_srcs, dependencies: [godotcore_dep])

DEP_MODULE_TGA = declare_dependency(link_with: lib_module_tga)

MODULE_DEPENDENCIES += DEP_MODULE_TGA
MODULES_ENABLED += 'tga'
