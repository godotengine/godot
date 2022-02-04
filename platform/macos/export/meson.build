_module_pvr_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

lib_module_pvr = library('module_pvr', _module_pvr_srcs,
    dependencies: [DEP_GODOT_ALL, DEP_PVRTCCOMPRESSOR]
)

DEP_MODULE_PVR = declare_dependency(link_with: lib_module_pvr)

MODULE_DEPENDENCIES += DEP_MODULE_PVR
