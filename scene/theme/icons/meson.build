# The theme data will only be triggered when it does not exist.

generated_font = custom_target('default_font',
    input: '../../../thirdparty/fonts/OpenSans_SemiBold.ttf',
    output: 'default_font.gen.h',
    env: SCRIPTS_ENV,
    command: [SCRIPT_COMPAT, 'make_fonts_header', SCRIPT_DEFAULT_THEME_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@'])

theme_icons = custom_target('default_theme_icosn',
  output: 'default_theme_icons.gen.h',
  command: [find_program('default_theme_icons_builders.py'), '@OUTPUT@', meson.current_source_dir()],
)

SOURCES += [generated_font, theme_icons]

SOURCES += files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
