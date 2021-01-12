_module_webp_srcs = files([
    'image_loader_webp.cpp',
    'register_types.cpp'
])

lib_module_webp = library('module_webp', _module_webp_srcs, dependencies: [DEP_GODOT_ALL, DEP_LIBWEBP])

DEP_MODULE_WEBP = declare_dependency(link_with: lib_module_webp)

MODULE_DEPENDENCIES += DEP_MODULE_WEBP