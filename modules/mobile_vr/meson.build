_module_mobile_vr_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
lib_module_mobile_vr = library('module_mobile_vr', _module_mobile_vr_srcs,
    dependencies: [DEP_GODOT_ALL]
)

DEP_MODULE_MOBILE_VR = declare_dependency(link_with: lib_module_mobile_vr)

MODULE_DEPENDENCIES += DEP_MODULE_MOBILE_VR
MODULES_ENABLED += 'mobile_vr'
