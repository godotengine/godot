_module_gles3_shaders_src = files()
_module_gles3_shaders_raw_shaders = [
    'canvas.glsl',
    'copy.glsl',
    'scene.glsl',
    'sky.glsl',
    'cubemap_filter.glsl',
    'canvas_occlusion.glsl',
    'canvas_sdf.glsl',
    'particles.glsl',
    'particles_copy.glsl',
    'skeleton.glsl',
]

foreach glsl_file : _module_gles3_shaders_raw_shaders
    _raw_target = custom_target(
        'raw_target_' + glsl_file,
        input: files(glsl_file),
        output: '@PLAINNAME@.gen.h',
        env: SCRIPTS_ENV,
        command: [SCRIPT_COMPAT, 'build_gles3_headers', SCRIPT_GLES3_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@'],
        build_by_default: true,
    )

    _module_gles3_shaders_src += [_raw_target]
endforeach

lib_module_gles3_shaders = library('module_gles3_shaders', _module_gles3_shaders_src, dependencies: [godotcore_dep])

DEP_MODULE_GLES3_SHADERS = declare_dependency(link_with: lib_module_gles3_shaders)

MODULE_DEPENDENCIES += DEP_MODULE_GLES3_SHADERS
MODULES_ENABLED += 'gles3_shaders'
