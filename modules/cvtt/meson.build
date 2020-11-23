_module_cvtt_srcs = files([
    'image_compress_cvtt.cpp',
    'register_types.cpp'
])

_lib_module_cvtt = library('module_cvtt', _module_cvtt_srcs, dependencies: [DEP_GODOT_ALL, DEP_CVTT])

DEP_MODULE_CVTT = declare_dependency(link_with: _lib_module_cvtt)

MODULE_DEPENDENCIES += DEP_MODULE_CVTT