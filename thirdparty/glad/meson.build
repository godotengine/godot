builtin_glad_dep_incdirs = include_directories('.')

builtin_glad_srcs = files('egl.c', 'gl.c')

if PLATFORM == 'linuxbsd'
    builtin_glad_srcs += files('glx.c')
endif

lib_builtin_glad = static_library('builtin_glad', builtin_glad_srcs,
    build_by_default: false
)

DEP_GLAD = declare_dependency(
    link_with: lib_builtin_glad,
    include_directories: builtin_glad_dep_incdirs,
)
