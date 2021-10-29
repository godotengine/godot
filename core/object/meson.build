SOURCES += files(run_command(GLOB, '*.cpp').stdout().split('\n'))

configure_file(output: 'gdvirtual.gen.inc',
               command: [find_program('make_virtuals.py'), '@OUTPUT@'])
