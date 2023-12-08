_module_webp_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
lib_module_webp = library('module_webp', _module_webp_srcs, dependencies: [godotcore_dep, DEP_LIBWEBP])

DEP_MODULE_WEBP = declare_dependency(link_with: lib_module_webp)

MODULE_DEPENDENCIES += DEP_MODULE_WEBP
MODULES_ENABLED += 'webp'
