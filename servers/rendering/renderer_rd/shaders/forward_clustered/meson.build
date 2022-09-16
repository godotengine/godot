environment_shaders = run_command(GLOB, '*.glsl', check: true).stdout().split('\n')

foreach glsl_file : environment_shaders
   shad_target = custom_target(
        'forward_clustered_target_' + glsl_file,
        output: '@PLAINNAME@.gen.h',
        input: files(glsl_file),
        env: SCRIPTS_ENV,
        command: [SCRIPT_COMPAT, 'build_rd_headers', SCRIPT_GLSL_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@'],
        build_by_default: true
    )

    SOURCES += [shad_target]
endforeach
