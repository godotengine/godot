# The theme data will only be triggered when it does not exist.

# generated version file
_default_theme_target = custom_target(
    'default_theme',
    output: ['theme_data.gen.h'],
    command: [SCRIPT_THEME_DATA, 'scene/resources/default_theme', meson.project_source_root(), '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_default_theme_target]

SOURCES += files([
    'default_theme.cpp'
])