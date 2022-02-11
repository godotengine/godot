_module_msdfgen_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
_lib_module_msdfgen = library('module_msdfgen', _module_msdfgen_srcs,
    dependencies: [godotcore_dep, DEP_MSDF]
)

DEP_MODULE_MSDFGEN = declare_dependency(link_with: _lib_module_msdfgen)

MODULE_DEPENDENCIES += DEP_MODULE_MSDFGEN
MODULES_ENABLED += 'msdfgen'
