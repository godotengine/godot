# generated input mappings
_default_controller_mappings_target = custom_target(
    'default_controller_mappings',
    output: ['default_controller_mappings.gen.cpp'],
    input: ['gamecontrollerdb.txt', 'godotcontrollerdb.txt'],
    command: [SCRIPT_DEFAULT_CONTROLLER_MAPPINGS, '@INPUT@', '@OUTPUT@'],
    build_by_default: true
)

SOURCES += [_default_controller_mappings_target]

SOURCES += files([
    'input.cpp',
    'input_event.cpp',
    'input_map.cpp',
])