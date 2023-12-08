MAIN_SOURCES += files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

configure_file(input: 'app_icon.png',
    output: 'app_icon.gen.h',
    command: [find_program('main_builders.py'), 'make_app_icon', '@INPUT@', '@OUTPUT@'])

configure_file(
    output: 'splash.gen.h',
    input: 'splash.png',
    command: [find_program('main_builders.py'),
              'make_splash',
              '@OUTPUT@',
              '@INPUT@',
       ]
)
