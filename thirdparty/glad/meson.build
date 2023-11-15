_builtin_glad_dep_incdirs = include_directories('.')
_builtin_glad_incdirs = _builtin_glad_dep_incdirs

_builtin_glad_srcs = files('egl.c', 'gl.c')

if PLATFORM == 'linuxbsd'
    _builtin_glad_srcs += files('glx.c')
endif

lib_builtin_glad = static_library('builtin_glad', _builtin_glad_srcs,
    include_directories: _builtin_glad_incdirs,
    build_by_default: false
)

DEP_GLAD = declare_dependency(
    link_with: lib_builtin_glad,
    include_directories: _builtin_glad_dep_incdirs,
)
