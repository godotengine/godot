
 _modules_enabled_target = custom_target(
    'modules_enabled',
    output: ['modules_enabled.gen.h'],
    input: [MODULES_DB_FILE],
    command: [SCRIPT_MODULES_GEN, 'modules_enabled', '@INPUT@', '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_modules_enabled_target]

_register_module_types_target = custom_target(
    'register_module_types',
    output: ['register_module_types.gen.cpp'],
    input: [MODULES_DB_FILE],
    command: [SCRIPT_MODULES_GEN, 'register_module_types', '@INPUT@', meson.project_source_root(), '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_register_module_types_target]
