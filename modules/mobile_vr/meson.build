_module_mobile_vr_srcs = files([
    'mobile_vr_interface.cpp',
    'register_types.cpp'
])

lib_module_mobile_vr = library('module_mobile_vr', _module_mobile_vr_srcs,
    dependencies: [DEP_GODOT_ALL]
)

DEP_MODULE_MOBILE_VR = declare_dependency(link_with: lib_module_mobile_vr)

MODULE_DEPENDENCIES += DEP_MODULE_MOBILE_VR