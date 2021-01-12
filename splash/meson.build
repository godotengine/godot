
_splash_target = custom_target(
    'splash',
    output: ['splash.gen.h'],
    input: ['splash.png'],
    command: [SCRIPT_SPLASH_GENERATOR, 'splash', '@INPUT@', '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_splash_target]

_splash_editor_target = custom_target(
    'splash_editor',
    output: ['splash_editor.gen.h'],
    input: ['splash_editor.png'],
    command: [SCRIPT_SPLASH_GENERATOR, 'splash_editor', '@INPUT@', '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_splash_editor_target]

_app_icon_target = custom_target(
    'app_icon',
    output: ['app_icon.gen.h'],
    input: ['app_icon.png'],
    command: [SCRIPT_SPLASH_GENERATOR, 'app_icon', '@INPUT@', '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_app_icon_target]