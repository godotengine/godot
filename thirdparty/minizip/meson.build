_minizip_incdirs = include_directories('.')

_minizip_srcs = files([
    'ioapi.c',
    'unzip.c',
    'zip.c'
])

_lib_minizip = static_library('minizip', _minizip_srcs, include_directories: _minizip_incdirs, dependencies: [DEP_ZLIB], build_by_default: false)

DEP_MINIZIP = declare_dependency(link_with: _lib_minizip, include_directories: _minizip_incdirs)