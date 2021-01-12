_modules_camera_srcs = files([
    'register_types.cpp'
])

if PLATFORM == 'windows'
    _modules_camera_srcs += files(['camera_win.cpp'])
elif PLATFORM == 'osx'
    _modules_camera_srcs += files(['camera_osx.mm'])
endif

_lib_module_camera = library('module_camera', _modules_camera_srcs,
    dependencies: DEP_GODOT_ALL
)

DEP_MODULE_CAMERA = declare_dependency(link_with: _lib_module_camera)

MODULE_DEPENDENCIES += DEP_MODULE_CAMERA