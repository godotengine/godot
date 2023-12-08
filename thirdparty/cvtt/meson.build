_cvtt_incdirs = include_directories('.')

_cvtt_srcs = files(
    'ConvectionKernels_SingleFile.cpp'
)

_lib_cvtt = static_library('builtin_cvtt',
                          _cvtt_srcs,
                          cpp_args: ['-DCVTT_SINGLE_FILE'],
                          include_directories: _cvtt_incdirs,
                          build_by_default: false)

DEP_CVTT = declare_dependency(link_with: _lib_cvtt, include_directories: _cvtt_incdirs)
