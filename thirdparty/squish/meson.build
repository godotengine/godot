if get_option('builtin_squish')
    _builtin_squish_srcs = files([
        'alpha.cpp',
        'clusterfit.cpp',
        'colourblock.cpp',
        'colourfit.cpp',
        'colourset.cpp',
        'maths.cpp',
        'rangefit.cpp',
        'singlecolourfit.cpp',
        'squish.cpp',
    ])

    _builtin_squish_incdirs = include_directories('.')

    lib_builtin_squish = static_library('builtin_squish', _builtin_squish_srcs,
        include_directories: _builtin_squish_incdirs,
        build_by_default: false
    )

    DEP_SQUISH = declare_dependency(link_with: lib_builtin_squish, include_directories: _builtin_squish_incdirs)
else
    DEP_SQUISH = dependency('libsquish')
endif