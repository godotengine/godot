_module_squish_srcs = files([
    'image_compress_squish.cpp',
    'register_types.cpp'
])

lib_module_squish = library('module_squish', _module_squish_srcs, dependencies: [DEP_GODOT_ALL, DEP_SQUISH])

DEP_MODULE_SQUISH = declare_dependency(link_with: lib_module_squish)

MODULE_DEPENDENCIES += DEP_MODULE_SQUISH