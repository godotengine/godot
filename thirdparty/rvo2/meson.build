_rvo2_incdirs = include_directories('.')

_rvo2_srcs = files([
    'Agent.cpp',
    'KdTree.cpp'
])

_lib_rvo2 = static_library('builtin_rvo2', _rvo2_srcs, include_directories: _rvo2_incdirs, build_by_default: false)

DEP_RVO2 = declare_dependency(link_with: _lib_rvo2, include_directories: _rvo2_incdirs)