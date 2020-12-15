_module_lmrd_src = files(run_command(GLOB, '*.cpp').stdout().split('\n'))
# NOTE: do not use files() here
_module_lmrd_raw_shaders = run_command(GLOB, '*.glsl').stdout().split('\n')

#build_raw_headers

foreach glsl_file : _module_lmrd_raw_shaders
    _raw_target = custom_target(
        'raw_target_' + glsl_file,
        input: files(glsl_file),
        output: '@PLAINNAME@.gen.h',
	env: SCRIPTS_ENV,
        command: [SCRIPT_COMPAT, 'build_raw_headers', SCRIPT_GLSL_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@'],
        build_by_default: true
    )

    _module_lmrd_src += [_raw_target]
endforeach

lib_module_lmrd = library('module_lightmapper_rd', _module_lmrd_src, dependencies: [DEP_GODOT_ALL])

DEP_MODULE_LIGHTMAPPER_RD = declare_dependency(link_with: lib_module_lmrd)

MODULE_DEPENDENCIES += DEP_MODULE_LIGHTMAPPER_RD
