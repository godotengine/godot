_module_bmp_srcs = files([
    'image_loader_bmp.cpp',
    'register_types.cpp',
])

_lib_module_bmp = library('module_bmp', _module_bmp_srcs,
    dependencies: DEP_GODOT_ALL,
)

DEP_MODULE_BMP = declare_dependency(link_with: _lib_module_bmp)

MODULE_DEPENDENCIES += DEP_MODULE_BMP