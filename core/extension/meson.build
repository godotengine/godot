ext_wrappers_gen = configure_file(output: 'ext_wrappers.gen.inc',
    command: [find_program('make_wrappers.py'), '@OUTPUT@'])

ext_iface_gen = configure_file(output: 'gdextension_interface_dump.gen.h',
    input: 'gdextension_interface.h',
    command: [find_program('make_interface_dumper.py'), '@INPUT@', '@OUTPUT@'])

SOURCES += files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
