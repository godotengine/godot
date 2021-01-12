_builtin_pvrtc_srcs = files([
    'BitScale.cpp',
    'MortonTable.cpp',
    'PvrTcDecoder.cpp',
    'PvrTcEncoder.cpp',
    'PvrTcPacket.cpp',
])

# we use core/typedefs.h from godot, so inject that core folder in
_builtin_pvrtc_build_incdirs = include_directories('../../', '../../' / DIR_PLATFORM)

_builtin_pvrtc_dep_incdirs = include_directories('.')

lib_builtin_pvrtc = static_library('builtin_pvrtccompressor', _builtin_pvrtc_srcs, 
    include_directories: _builtin_pvrtc_build_incdirs,
    build_by_default: false
)

DEP_PVRTCCOMPRESSOR = declare_dependency(link_with: lib_builtin_pvrtc,
    include_directories: _builtin_pvrtc_dep_incdirs)