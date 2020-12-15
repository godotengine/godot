_builtin_vhacd_srcs = files([
    'src/vhacdManifoldMesh.cpp',
    'src/FloatMath.cpp',
    'src/vhacdMesh.cpp',
    'src/vhacdICHull.cpp',
    'src/vhacdVolume.cpp',
    'src/VHACD-ASYNC.cpp',
    'src/btAlignedAllocator.cpp',
    'src/vhacdRaycastMesh.cpp',
    'src/VHACD.cpp',
    'src/btConvexHullComputer.cpp',
])

_builtin_vhacd_incdirs = include_directories('inc')

lib_builtin_vhacd = static_library('builtin_vhacd', _builtin_vhacd_srcs, include_directories: _builtin_vhacd_incdirs, build_by_default: false)

DEP_VHACD = declare_dependency(link_with: lib_builtin_vhacd, include_directories: _builtin_vhacd_incdirs)