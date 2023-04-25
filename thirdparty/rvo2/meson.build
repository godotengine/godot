_rvo2_incdirs = include_directories('rvo2_2d', 'rvo2_3d')

_rvo2_srcs = files(
  'rvo2_2d/Agent2d.cpp',
  'rvo2_2d/KdTree2d.cpp',
  'rvo2_2d/Obstacle2d.cpp',
  'rvo2_2d/RVOSimulator2d.cpp',
  'rvo2_3d/Agent3d.cpp',
  'rvo2_3d/KdTree3d.cpp',
  'rvo2_3d/RVOSimulator3d.cpp',
)

_lib_rvo2 = static_library('builtin_rvo2',
    _rvo2_srcs,
    include_directories: _rvo2_incdirs, build_by_default: false)

DEP_RVO2 = declare_dependency(link_with: _lib_rvo2,
    include_directories: _rvo2_incdirs)
