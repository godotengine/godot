_module_svg_sourcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
_lib_module_svg = library('module_svg', _module_svg_sourcs, dependencies: [DEP_GODOT_ALL, thorvg_dep])

DEP_MODULE_SVG = declare_dependency(link_with: _lib_module_svg)

MODULE_DEPENDENCIES += DEP_MODULE_SVG
MODULES_ENABLED += 'svg'
