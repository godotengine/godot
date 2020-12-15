if get_option('builtin_libtheora')

    _builtin_libtheora_srcs = files([
        # 'analyze.c',
        # 'apiwrapper.c',
        'bitpack.c',
        'cpu.c',
        # 'decapiwrapper.c',
        'decinfo.c',
        'decode.c',
        'dequant.c',
        # 'encapiwrapper.c',
        # 'encfrag.c',
        # 'encinfo.c',
        # 'encode.c',
        # 'encoder_disabled.c',
        # 'enquant.c',
        # 'fdct.c',
        'fragment.c',
        'huffdec.c',
        # 'huffenc.c',
        'idct.c',
        'info.c',
        'internal.c',
        # 'mathops.c',
        # 'mcenc.c',
        'quant.c',
        # 'rate.c',
        'state.c',
        # 'tokenize.c',
    ])

    _builtin_libtheora_compile_args = []

    _builtin_libtheora_incdirs = include_directories('.')

    if host_machine.cpu_family() in ['x86', 'x86_64']
        _builtin_libtheora_compile_args += '-DOC_X86_ASM'

        if PLATFORM == 'windows'
            _builtin_libtheora_srcs += files([
                # 'x86_vc/mmxencfrag.c',
                # 'x86_vc/mmxfdct.c',
                'x86_vc/mmxfrag.c',
                'x86_vc/mmxidct.c',
                'x86_vc/mmxstate.c',
                # 'x86_vc/sse2fdct.c',
                # 'x86_vc/x86enc.c',
                'x86_vc/x86state.c',
            ])
        else
            _builtin_libtheora_srcs += files([
                # 'x86/mmxencfrag.c',
                # 'x86/mmxfdct.c',
                'x86/mmxfrag.c',
                'x86/mmxidct.c',
                'x86/mmxstate.c',
                # 'x86/sse2fdct.c',
                # 'x86/x86enc.c',
                'x86/x86state.c',
            ])
        endif
    endif

    lib_builtin_libtheora = static_library('builtin_libtheora', _builtin_libtheora_srcs,
        include_directories: _builtin_libtheora_incdirs,
        dependencies: [DEP_LIBOGG, DEP_LIBVORBIS],
        build_by_default: false
    )

    DEP_LIBTHEORA = declare_dependency(link_with: lib_builtin_libtheora, include_directories: _builtin_libtheora_incdirs)
else
    DEP_LIBTHEORA = [
        dependency('theora'),
        dependency('theoradec')
    ]
endif