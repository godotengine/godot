_module_squish_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
lib_module_squish = library('module_squish', _module_squish_srcs, dependencies: [DEP_GODOT_ALL, DEP_SQUISH])

DEP_MODULE_SQUISH = declare_dependency(link_with: lib_module_squish)

MODULE_DEPENDENCIES += DEP_MODULE_SQUISH
