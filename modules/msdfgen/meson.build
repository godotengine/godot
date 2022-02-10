_module_msdfgen_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
_lib_module_msdfgen = library('module_msdfgen', _module_msdfgen_srcs,
    dependencies: [DEP_GODOT_ALL, DEP_MSDF]
)

DEP_MODULE_MSDFGEN = declare_dependency(link_with: _lib_module_msdfgen)

MODULE_DEPENDENCIES += DEP_MODULE_MSDFGEN
MODULES_ENABLED += 'msdfgen'
