_register_module_types_target = custom_target(
    'register_module_types',
    output: ['register_module_types.gen.cpp'],
    input: [MODULES_DB_FILE],
    command: [SCRIPT_MODULES_GEN, 'register_module_types', '@INPUT@', MESON_SOURCE_ROOT, '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_register_module_types_target]
