# generated certs

_certs_target = custom_target(
    'certs',
    output: ['certs_compressed.gen.h'],
    input: [meson.project_source_root() + '/thirdparty/certs/ca-certificates.crt'],
    env: SCRIPTS_ENV,
    command: [SCRIPT_COMPAT, 'make_certs_header', SCRIPT_CORE_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@', '-e',
      'builtin_certs=' + (get_option('builtin_certs') ? 'true' : 'false'),
      'system_certs_path=' + get_option('system_certs_path')],
    build_by_default: true
)

SOURCES += [_certs_target]

SOURCES += files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
