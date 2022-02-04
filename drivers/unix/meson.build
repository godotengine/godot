if meson.get_compiler('cpp').has_header('mntent.h')
    CPP_ARGS += ['-DHAVE_MNTENT']
endif

SOURCES += files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
