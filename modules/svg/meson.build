_module_svg_sourcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))
_lib_module_svg = library('module_svg', _module_svg_sourcs, gdivirtuals_gen, dependencies: [DEP_GODOT_ALL, DEP_NANOSVG])

DEP_MODULE_SVG = declare_dependency(link_with: _lib_module_svg)

MODULE_DEPENDENCIES += DEP_MODULE_SVG
