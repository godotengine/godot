if get_option('builtin_libpng')
    _libpng_incdirs = include_directories('.')

    _libpng_sources = files([
        'png.c',
        'pngerror.c',
        'pngget.c',
        'pngmem.c',
        'pngpread.c',
        'pngread.c',
        'pngrio.c',
        'pngrtran.c',
        'pngrutil.c',
        'pngset.c',
        'pngtrans.c',
        'pngwio.c',
        'pngwrite.c',
        'pngwtran.c',
        'pngwutil.c',
    ])

    if host_machine.cpu_family() == 'aarch64' or host_machine.cpu_family() == 'arm'
        _libpng_sources += files(
            'arm/arm_init.c',
            'arm/filter_neon_intrinsics.c',
            'arm/palette_neon_intrinsics.c',
        )
    endif

    _lib_libpng = static_library('libpng', _libpng_sources, include_directories: _libpng_incdirs, dependencies: [DEP_ZLIB], build_by_default: false)

    DEP_LIBPNG = declare_dependency(link_with: _lib_libpng, include_directories: _libpng_incdirs)
else
    DEP_LIBPNG = dependency('libpng')
endif
