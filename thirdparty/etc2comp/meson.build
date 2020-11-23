_etc2comp_srcs = files([
    'EtcBlock4x4.cpp',
    'EtcBlock4x4Encoding.cpp',
    'EtcBlock4x4Encoding_ETC1.cpp',
    'EtcBlock4x4Encoding_R11.cpp',
    'EtcBlock4x4Encoding_RG11.cpp',
    'EtcBlock4x4Encoding_RGB8A1.cpp',
    'EtcBlock4x4Encoding_RGB8.cpp',
    'EtcBlock4x4Encoding_RGBA8.cpp',
    'Etc.cpp',
    'EtcDifferentialTrys.cpp',
    'EtcFilter.cpp',
    'EtcImage.cpp',
    'EtcIndividualTrys.cpp',
    'EtcMath.cpp',
    'EtcSortedBlockList.cpp',
])

_etc2comp_incdirs = include_directories('.')

_lib_etc2comp = static_library('builtin_etc2comp', _etc2comp_srcs, include_directories: _etc2comp_incdirs, build_by_default: false)

DEP_ETC2COMP = declare_dependency(link_with: _lib_etc2comp, include_directories: _etc2comp_incdirs)