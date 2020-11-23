# NOTE: do not use files() here
_module_lmrd_src = files([
    'lightmapper_rd.cpp',
    'register_types.cpp'
])

_module_lmrd_raw_shaders = [
    'lm_raster.glsl',
    'lm_compute.glsl',
    'lm_blendseams.glsl'
]

foreach glsl_file : _module_lmrd_raw_shaders
    _raw_target = custom_target(
        'raw_target_' + glsl_file,
        output: '@PLAINNAME@.gen.h',
        input: glsl_file,
        command: [SCRIPT_GLSL, 'raw', '@INPUT@', '@OUTPUT@'],
        build_by_default: true
    )

    _module_lmrd_src += [_raw_target]
endforeach

lib_module_lmrd = library('module_lightmapper_rd', _module_lmrd_src, dependencies: [DEP_GODOT_ALL])

DEP_MODULE_LIGHTMAPPER_RD = declare_dependency(link_with: lib_module_lmrd)

MODULE_DEPENDENCIES += DEP_MODULE_LIGHTMAPPER_RD