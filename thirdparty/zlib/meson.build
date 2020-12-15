

if get_option('builtin_zlib')
    _zlib_incdirs = include_directories('.')
    _zlib_srcs = files([
        'adler32.c',
        'compress.c',
        'crc32.c',
        'deflate.c',
        'infback.c',
        'inffast.c',
        'inflate.c',
        'inftrees.c',
        'trees.c',
        'uncompr.c',
        'zutil.c'
    ])

    _zlib_link_args = []
    _zlib_c_args = []

    if get_option('debug')
        _zlib_c_args += '-DZLIB_DEBUG'
    endif

    _c_compiler = meson.get_compiler('c').get_id()

    if _c_compiler == 'clang'
        _zlib_link_args += ['-Wdeprecated-declarations']
    endif

    _lib_zlib = static_library('zlib', _zlib_srcs, include_directories: _zlib_incdirs, link_args: _zlib_link_args, c_args: _zlib_c_args, build_by_default: false)

    _zlib_dep_flags = []
    if get_option('debug')
        _zlib_dep_flags += ['-DZLIB_DEBUG']
    endif

    DEP_ZLIB = declare_dependency(link_with: _lib_zlib, include_directories: _zlib_incdirs, compile_args: _zlib_dep_flags)
else
    DEP_ZLIB = dependency('zlib', required=false)
endif