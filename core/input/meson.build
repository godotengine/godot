# generated input mappings
_default_controller_mappings_target = custom_target(
    'default_controller_mappings',
    output: ['default_controller_mappings.gen.cpp'],
    input: ['gamecontrollerdb.txt', 'godotcontrollerdb.txt'],
    env: SCRIPTS_ENV,
    command: [SCRIPT_COMPAT, 'make_default_controller_mappings', SCRIPT_INPUT_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@'],
    build_by_default: true
)

SOURCES += [_default_controller_mappings_target]

SOURCES += files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
