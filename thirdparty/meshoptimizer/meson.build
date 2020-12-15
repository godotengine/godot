_builtin_meshoptimizer_srcs = files([
    'allocator.cpp',
    'clusterizer.cpp',
    'indexcodec.cpp',
    'indexgenerator.cpp',
    'overdrawanalyzer.cpp',
    'overdrawoptimizer.cpp',
    'simplifier.cpp',
    'spatialorder.cpp',
    'stripifier.cpp',
    'vcacheanalyzer.cpp',
    'vcacheoptimizer.cpp',
    'vertexcodec.cpp',
    'vertexfilter.cpp',
    'vfetchanalyzer.cpp',
    'vfetchoptimizer.cpp',
])

_builtin_meshoptimizer_incdirs = include_directories('.')

lib_builtin_meshoptimizer = static_library('builtin_meshoptimizer', _builtin_meshoptimizer_srcs,
    include_directories: _builtin_meshoptimizer_incdirs,
    build_by_default: false
)

DEP_MESHOPTIMIZER = declare_dependency(link_with: lib_builtin_meshoptimizer, include_directories: _builtin_meshoptimizer_incdirs)