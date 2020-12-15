_module_cvtt_srcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))
_lib_module_cvtt = library('module_cvtt', _module_cvtt_srcs, dependencies: [DEP_GODOT_ALL, DEP_CVTT])

DEP_MODULE_CVTT = declare_dependency(link_with: _lib_module_cvtt)

MODULE_DEPENDENCIES += DEP_MODULE_CVTT
