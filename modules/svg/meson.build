_module_svg_sourcs = files([
    'image_loader_svg.cpp',
    'register_types.cpp'
])

_lib_module_svg = library('module_svg', _module_svg_sourcs, dependencies: [DEP_GODOT_ALL, DEP_NANOSVG])

DEP_MODULE_SVG = declare_dependency(link_with: _lib_module_svg)

MODULE_DEPENDENCIES += DEP_MODULE_SVG