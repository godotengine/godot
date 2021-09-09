_misc_incdirs = include_directories('.')

_misc_srcs = files(
'clipper.cpp',
'pcg.cpp',
'polypartition.cpp',
'smolv.cpp',
'fastlz.c',
'mikktspace.c',
'open-simplex-noise.c',
'r128.c',
'smaz.c',
)

if PLATFORM == 'android'
    _misc_srcs += files('ifaddrs-android.cc')
endif

_lib_misc = static_library('misc', _misc_srcs,
    include_directories: [_misc_incdirs, '../../', '../../platform/linuxbsd'],
    dependencies: [DEP_ZLIB],
    )

DEP_MISC = declare_dependency(link_with: _lib_misc, include_directories: _misc_incdirs)
