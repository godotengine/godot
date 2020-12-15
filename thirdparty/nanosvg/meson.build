_nanosvg_incdirs = include_directories('.')

_nanosvg_srcs = files([
    'nanosvg.cc'
])

_lib_nanosvg = static_library('nanosvg', _nanosvg_srcs,
    include_directories: _nanosvg_incdirs,
    build_by_default: false
)

DEP_NANOSVG = declare_dependency(link_with: _lib_nanosvg, include_directories: _nanosvg_incdirs)

