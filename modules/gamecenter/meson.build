_module_gamecenter_srcs = files([
    'game_center_delegate.mm',
    'game_center_module.cpp',
    'game_center.mm',
])

_module_gamecenter_cpp_args = ['-fmodules', '-fcxx-modules']

# static lib for ios
_lib_module_gamecenter = static_library('module_gamecenter', _module_gamecenter_srcs,
    dependencies: DEP_GODOT_ALL,
    cpp_args: _module_gamecenter_cpp_args
)

DEP_MODULE_GAMECENTER = declare_dependency(link_with: _lib_module_gamecenter)

MODULE_DEPENDENCIES += DEP_MODULE_GAMECENTER