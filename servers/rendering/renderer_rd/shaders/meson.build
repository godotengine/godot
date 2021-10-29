# NOTE: do not use files() here
_rd_shaders = run_command(GLOB, '*.glsl').stdout().split('\n')

foreach glsl_file : _rd_shaders
    _rd_target = custom_target(
        'rd_target_' + glsl_file,
        output: '@PLAINNAME@.gen.h',
        input: files(glsl_file),
        env: SCRIPTS_ENV,
        command: [SCRIPT_COMPAT, 'build_rd_headers', SCRIPT_GLSL_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@'],
        build_by_default: true
    )

    SOURCES += [_rd_target]
endforeach
