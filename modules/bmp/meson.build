_module_bmp_srcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))

_lib_module_bmp = library('module_bmp', _module_bmp_srcs,
    dependencies: DEP_GODOT_ALL,
)

DEP_MODULE_BMP = declare_dependency(link_with: _lib_module_bmp)

MODULE_DEPENDENCIES += DEP_MODULE_BMP
