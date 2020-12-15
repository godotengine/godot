# The theme data will only be triggered when it does not exist.

_generated_font = custom_target('default_font',
    input: '../../../thirdparty/fonts/OpenSans_SemiBold.ttf',
    output: 'default_font.gen.h',
    env: SCRIPTS_ENV,
    command: [SCRIPT_COMPAT, 'make_fonts_header', SCRIPT_DEFAULT_THEME_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@'])

SOURCES += [_generated_font]

SOURCES += files(run_command(GLOB, '*.cpp').stdout().split('\n'))
