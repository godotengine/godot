_recast_incdirs = include_directories('Recast/Include')

_recast_srcs = files([
    'Recast/Source/Recast.cpp',
    'Recast/Source/RecastAlloc.cpp',
    'Recast/Source/RecastArea.cpp',
    'Recast/Source/RecastAssert.cpp',
    'Recast/Source/RecastContour.cpp',
    'Recast/Source/RecastFilter.cpp',
    'Recast/Source/RecastLayers.cpp',
    'Recast/Source/RecastMesh.cpp',
    'Recast/Source/RecastMeshDetail.cpp',
    'Recast/Source/RecastRasterization.cpp',
    'Recast/Source/RecastRegion.cpp',
])

_lib_recast = static_library('bultin_recast', _recast_srcs, include_directories: _recast_incdirs, build_by_default: false)

DEP_RECAST = declare_dependency(link_with: _lib_recast, include_directories: _recast_incdirs)