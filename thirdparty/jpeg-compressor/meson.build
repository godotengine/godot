_jpeg_compressor_srcs = files([
    'jpgd.cpp'
])

_jpeg_compressor_incdirs = include_directories('.')

_lib_jpeg_compressor = static_library('builtin_jpeg_compressor', _jpeg_compressor_srcs, include_directories: _jpeg_compressor_incdirs, build_by_default: false)

DEP_JPEG_COMPRESSOR = declare_dependency(link_with: _lib_jpeg_compressor, include_directories: _jpeg_compressor_incdirs)