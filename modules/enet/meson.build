_module_enet_srcs = files([
    'networked_multiplayer_enet.cpp',
    'register_types.cpp',
])

_lib_module_enet = library('module_enet', _module_enet_srcs,
    dependencies: [DEP_GODOT_ALL, DEP_ENET]
)

DEP_MODULE_ENET = declare_dependency(link_with: _lib_module_enet)

MODULE_DEPENDENCIES += DEP_MODULE_ENET