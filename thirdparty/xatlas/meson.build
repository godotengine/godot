_builtin_xatlas_srcs = files([
    'xatlas.cpp'
])

_builtin_xatlas_incdirs = include_directories('.')

lib_bultin_xatlas = static_library('bultin_xatlas', _builtin_xatlas_srcs,
    include_directories: _builtin_xatlas_incdirs,
    build_by_default: false
)

DEP_XATLAS = declare_dependency(link_with: lib_bultin_xatlas, include_directories: _builtin_xatlas_incdirs)