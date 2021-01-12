_module_opensimplex_srcs = files([
    # thirdparty misc
    '../../thirdparty/misc/open-simplex-noise.c',

    # our srcs
    'noise_texture.cpp',
    'open_simplex_noise.cpp',
    'register_types.cpp'
])

lib_module_opensimplex = library('module_opensimplex', _module_opensimplex_srcs,
    dependencies: [DEP_GODOT_ALL]
)

DEP_MODULE_OPENSIMPLEX = declare_dependency(link_with: lib_module_opensimplex)

MODULE_DEPENDENCIES += DEP_MODULE_OPENSIMPLEX