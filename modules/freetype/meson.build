ftmod_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

lib_module_freetype = library('module_freetype',
    ftmod_srcs,
    dependencies: [godotcore_dep, DEP_FREETYPE]
)

DEP_MODULE_FREETYPE = declare_dependency(link_with: lib_module_freetype)

MODULE_DEPENDENCIES += DEP_MODULE_FREETYPE
MODULES_ENABLED += 'freetype'
