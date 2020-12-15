# thirdparty misc sources
DEPENDENCIES += [ DEP_FREETYPE ]

SOURCES += files([
    # C sources
    '../thirdparty/misc/mikktspace.c',
    # C++ sources
    '../thirdparty/misc/easing_equations.cpp',
])

SOURCES += files(run_command(GLOB, '*.cpp').stdout().split('\n'))

subdir('2d')
subdir('3d')
subdir('animation')
subdir('audio')
subdir('debugger')
subdir('gui')
subdir('main')
subdir('resources')
