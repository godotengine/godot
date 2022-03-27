ext_wrappers_gen = configure_file(output: 'ext_wrappers.gen.inc',
    command: [find_program('make_wrappers.py'), '@OUTPUT@'])

SOURCES += files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
