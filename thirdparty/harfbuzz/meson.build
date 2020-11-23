if get_option('builtin_harfbuzz')
    _builtin_harfbuzz_srcs = files([
        'src/hb-aat-layout.cc',
        'src/hb-aat-map.cc',
        'src/hb-blob.cc',
        'src/hb-buffer-serialize.cc',
        'src/hb-buffer.cc',
        'src/hb-common.cc',
        #'src/hb-coretext.cc',
        #'src/hb-directwrite.cc',
        'src/hb-draw.cc',
        'src/hb-face.cc',
        'src/hb-fallback-shape.cc',
        'src/hb-font.cc',
        'src/hb-ft.cc',
        #'src/hb-gdi.cc',
        #'src/hb-glib.cc',
        #'src/hb-gobject-structs.cc',
        'src/hb-graphite2.cc',
        'src/hb-icu.cc',
        'src/hb-map.cc',
        'src/hb-number.cc',
        'src/hb-ot-cff1-table.cc',
        'src/hb-ot-cff2-table.cc',
        'src/hb-ot-color.cc',
        'src/hb-ot-face.cc',
        'src/hb-ot-font.cc',
        'src/hb-ot-layout.cc',
        'src/hb-ot-map.cc',
        'src/hb-ot-math.cc',
        'src/hb-ot-meta.cc',
        'src/hb-ot-metrics.cc',
        'src/hb-ot-name.cc',
        'src/hb-ot-shape-complex-arabic.cc',
        'src/hb-ot-shape-complex-default.cc',
        'src/hb-ot-shape-complex-hangul.cc',
        'src/hb-ot-shape-complex-hebrew.cc',
        'src/hb-ot-shape-complex-indic-table.cc',
        'src/hb-ot-shape-complex-indic.cc',
        'src/hb-ot-shape-complex-khmer.cc',
        'src/hb-ot-shape-complex-myanmar.cc',
        'src/hb-ot-shape-complex-thai.cc',
        'src/hb-ot-shape-complex-use-table.cc',
        'src/hb-ot-shape-complex-use.cc',
        'src/hb-ot-shape-complex-vowel-constraints.cc',
        'src/hb-ot-shape-fallback.cc',
        'src/hb-ot-shape-normalize.cc',
        'src/hb-ot-shape.cc',
        'src/hb-ot-tag.cc',
        'src/hb-ot-var.cc',
        'src/hb-set.cc',
        'src/hb-shape-plan.cc',
        'src/hb-shape.cc',
        'src/hb-shaper.cc',
        'src/hb-static.cc',
        'src/hb-style.cc',
        'src/hb-subset-cff-common.cc',
        'src/hb-subset-cff1.cc',
        'src/hb-subset-cff2.cc',
        'src/hb-subset-input.cc',
        'src/hb-subset-plan.cc',
        'src/hb-subset.cc',
        'src/hb-ucd.cc',
        'src/hb-unicode.cc',
        #'src/hb-uniscribe.cc'
    ])

    _builtin_harfbuzz_incdirs = include_directories('src')

    _builtin_harfbuzz_dep_compile_flags = []
    _builtin_harfbuzz_compile_flags = [
        '-DHAVE_ICU_BUILTIN',
        '-DHAVE_ICU',
        '-DHAVE_FREETYPE',
        '-DHAVE_GRAPHITE2',
        '-DHB_DISABLE_DEPRECATED'
    ]

    if PLATFORM in ['android', 'linuxbsd', 'server']
        _builtin_harfbuzz_compile_flags += ['-DHAVE_PTHREAD']
    endif

    if PLATFORM == 'javascript'
        if get_option('threads_enabled')
            _builtin_harfbuzz_compile_flags += ['-DHAVE_PTHREAD']
        else
            _builtin_harfbuzz_dep_compile_flags += ['-DHB_NO_MT']
        endif
    endif

    # compiler warnings
    if meson.get_compiler('cpp').get_id() == 'msvc'
        _builtin_harfbuzz_compile_flags += '/wd4172'
    elif meson.get_compiler('cpp').get_id() == 'gcc'
        _builtin_harfbuzz_compile_flags += ['-DHB_NO_PRAGMA_GCC_DIAGNOSTIC_WARNING', '-Wno-deprecated-declarations']
    endif

    lib_builtin_harfbuzz = static_library('builtin_harfbuzz', _builtin_harfbuzz_srcs,
        include_directories: _builtin_harfbuzz_incdirs,
        cpp_args: [_builtin_harfbuzz_dep_compile_flags, _builtin_harfbuzz_compile_flags],
        dependencies: [DEP_GRAPHITE, DEP_FREETYPE, DEP_ICU],
        build_by_default: false
    )

    DEP_HARFBUZZ = declare_dependency(link_with: lib_builtin_harfbuzz,
        include_directories: _builtin_harfbuzz_incdirs,
        compile_args: _builtin_harfbuzz_dep_compile_flags
    )

else
    DEP_HARFBUZZ = [
        dependency('harfbuzz'),
        dependency('harfbuzz-icu')
    ]
endif