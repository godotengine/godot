_module_camera_iphone_srcs = files([
    'camera_ios.mm',
    'camera_module.cpp',
])

_module_camera_iphone_cpp_args = ['-fmodules', '-fcxx-modules']

# static lib for ios
_lib_module_camera_iphone = static_library('module_camera', _module_camera_iphone_srcs,
    dependencies: DEP_GODOT_ALL,
    cpp_args: _module_camera_iphone_cpp_args
)

DEP_MODULE_CAMERA_IPHONE = declare_dependency(link_with: _lib_module_camera_iphone)

MODULE_DEPENDENCIES += DEP_MODULE_CAMERA_IPHONE