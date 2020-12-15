_module_dds_srcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))
_lib_module_dds = library('module_dds', _module_dds_srcs,
    dependencies: [DEP_GODOT_ALL]
)

DEP_MODULE_DDS = declare_dependency(link_with: _lib_module_dds)

MODULE_DEPENDENCIES += DEP_MODULE_DDS
