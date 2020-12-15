# This entire driver confuses me...

CPP_FLAGS += ['-DXAUDIO2_ENABLED']

_lib_xaudio2 = meson.get_compiler('cpp').find_library('xaudio2')

DEPENDENCIES += [_lib_xaudio2]

SOURCES += files(run_command(GLOB, '*.cpp').stdout().split('\n'))
