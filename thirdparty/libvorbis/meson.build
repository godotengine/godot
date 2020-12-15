if get_option('builtin_libvorbis')
    _builtin_libvorbis_srcs = files([
        # 'analysis.c',
        # 'barkmel.c',
        'bitrate.c',
        'block.c',
        'codebook.c',
        'envelope.c',
        'floor0.c',
        'floor1.c',
        'info.c',
        'lookup.c',
        'lpc.c',
        'lsp.c',
        'mapping0.c',
        'mdct.c',
        'psy.c',
        # 'psytune.c',
        'registry.c',
        'res0.c',
        'sharedbook.c',
        'smallft.c',
        'synthesis.c',
        # 'tone.c',
        # 'vorbisenc.c',
        'vorbisfile.c',
        'window.c',
    ])

    _builtin_libvorbis_incdirs = include_directories('.')

    c_dep_args = []
    if PLATFORM != 'windows'
        c_dep_args += '-DHAVE_ALLOCA_H'
    endif

    lib_builtin_libvorbis = static_library('builtin_libvorbis', _builtin_libvorbis_srcs,
        include_directories: _builtin_libvorbis_incdirs,
        dependencies: [DEP_LIBOGG],
        c_args: c_dep_args,
        build_by_default: false
    )


    DEP_LIBVORBIS = declare_dependency(link_with: lib_builtin_libvorbis, include_directories: _builtin_libvorbis_incdirs, compile_args: c_dep_args)

else
    DEP_LIBVORBIS = [
        dependency('vorbis'),
        dependency('vorbisfile')
    ]
endif
