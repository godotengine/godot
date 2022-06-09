# thirdparty misc sources
DEPENDENCIES += [ DEP_FREETYPE, DEP_BROTLI ]

SOURCES += files([
    # C sources
    '../thirdparty/misc/mikktspace.c',
])

SOURCES += files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

subdir('2d')
subdir('3d')
subdir('animation')
subdir('audio')
subdir('debugger')
subdir('gui')
subdir('main')
subdir('multiplayer')
subdir('resources')
subdir('theme')
