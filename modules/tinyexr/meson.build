_module_tinyexr_srcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))
lib_module_tinyexr = library('module_tinyexr', _module_tinyexr_srcs, dependencies: [DEP_GODOT_ALL, DEP_TINYEXR])

DEP_MODULE_TINYEXR = declare_dependency(link_with: lib_module_tinyexr)

MODULE_DEPENDENCIES += DEP_MODULE_TINYEXR
