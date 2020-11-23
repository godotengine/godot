_module_hdr_srcs = files([
    'image_loader_hdr.cpp',
    'register_types.cpp',
])

_lib_module_hdr = library('module_hdr', _module_hdr_srcs,
    dependencies: [DEP_GODOT_ALL]
)

DEP_MODULE_HDR = declare_dependency(link_with: _lib_module_hdr)

MODULE_DEPENDENCIES += DEP_MODULE_HDR