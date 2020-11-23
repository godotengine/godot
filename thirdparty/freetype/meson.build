if get_option('builtin_freetype')
    _freetype_incdirs = include_directories('include')

    _freetype_srcs = files([
        'src/autofit/autofit.c',
        'src/base/ftbase.c',
        'src/base/ftbbox.c',
        'src/base/ftbdf.c',
        'src/base/ftbitmap.c',
        'src/base/ftcid.c',
        'src/base/ftdebug.c',
        'src/base/ftfstype.c',
        'src/base/ftgasp.c',
        'src/base/ftglyph.c',
        'src/base/ftgxval.c',
        'src/base/ftinit.c',
        'src/base/ftmm.c',
        'src/base/ftotval.c',
        'src/base/ftpatent.c',
        'src/base/ftpfr.c',
        'src/base/ftstroke.c',
        'src/base/ftsynth.c',
        'src/base/ftsystem.c',
        'src/base/fttype1.c',
        'src/base/ftwinfnt.c',
        'src/bdf/bdf.c',
        'src/bzip2/ftbzip2.c',
        'src/cache/ftcache.c',
        'src/cff/cff.c',
        'src/cid/type1cid.c',
        'src/gxvalid/gxvalid.c',
        'src/gzip/ftgzip.c',
        'src/lzw/ftlzw.c',
        'src/otvalid/otvalid.c',
        'src/pcf/pcf.c',
        'src/pfr/pfr.c',
        'src/psaux/psaux.c',
        'src/pshinter/pshinter.c',
        'src/psnames/psnames.c',
        'src/raster/raster.c',
        'src/smooth/smooth.c',
        'src/truetype/truetype.c',
        'src/type1/type1.c',
        'src/type42/type42.c',
        'src/winfonts/winfnt.c',
        'src/sfnt/sfnt.c'
    ])

    _freetype_dep_defines = ['-DFT2_BUILD_LIBRARY', '-DFT_CONFIG_OPTION_USE_PNG']
    
    _freetype_c_defines = []
    if PLATFORM == 'javascript'
        _freetype_c_defines += ['-U__OPTIMIZE__']
    endif

    
    if meson.get_compiler('c').get_id() == 'msvc'
        _freetype_c_defines += ['/wd4312']
    endif

    _lib_freetype2 = static_library('freetype2', _freetype_srcs, include_directories: _freetype_incdirs, c_args: _freetype_dep_defines + _freetype_c_defines, dependencies: [DEP_ZLIB, DEP_LIBPNG], build_by_default: false)

    DEP_FREETYPE = declare_dependency(link_with: _lib_freetype2, include_directories: _freetype_incdirs, compile_args: _freetype_dep_defines)
else
    DEP_FREETYPE = dependency('freetype2')
endif