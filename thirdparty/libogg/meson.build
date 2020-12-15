if get_option('builtin_libogg')

    _builtin_libogg_srcs = files([
        'bitwise.c',
        'framing.c'
    ])

    _builtin_libogg_incdirs = include_directories('.')

    lib_builtin_libogg = static_library('builtin_libogg', _builtin_libogg_srcs, include_directories: _builtin_libogg_incdirs,
        build_by_default: false)

    DEP_LIBOGG = declare_dependency(link_with: lib_builtin_libogg, include_directories: _builtin_libogg_incdirs)
else
    DEP_LIBOGG = dependency('ogg')
endif