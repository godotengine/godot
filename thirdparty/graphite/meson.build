if get_option('builtin_graphite')

    _builtin_graphite_dep_incdirs = include_directories('include')
    _builtin_graphite_incdirs = include_directories('include', 'src')

    _builtin_graphite_srcs = files([
        'src/gr_char_info.cpp',
        'src/gr_face.cpp',
        'src/gr_features.cpp',
        'src/gr_font.cpp',
        'src/gr_logging.cpp',
        'src/gr_segment.cpp',
        'src/gr_slot.cpp',
        'src/CmapCache.cpp',
        'src/Code.cpp',
        'src/Collider.cpp',
        'src/Decompressor.cpp',
        'src/Face.cpp',
        #'src/FileFace.cpp',
        'src/FeatureMap.cpp',
        'src/Font.cpp',
        'src/GlyphCache.cpp',
        'src/GlyphFace.cpp',
        'src/Intervals.cpp',
        'src/Justifier.cpp',
        'src/NameTable.cpp',
        'src/Pass.cpp',
        'src/Position.cpp',
        'src/Segment.cpp',
        'src/Silf.cpp',
        'src/Slot.cpp',
        'src/Sparse.cpp',
        'src/TtfUtil.cpp',
        'src/UtfCodec.cpp',
        'src/FileFace.cpp',
        'src/json.cpp',
    ])

    _builtin_graphite_compile_args = [
        '-DGRAPHITE2_STATIC',
        '-DGRAPHITE2_NTRACING',
        '-DGRAPHITE2_NFILEFACE',
    ]

    if meson.get_compiler('cpp').get_id() == 'msvc'
        _builtin_graphite_srcs += files(['src/call_machine.cpp'])
    else
        _builtin_graphite_srcs += files(['src/direct_machine.cpp'])
    endif

    if meson.get_compiler('cpp').get_id() == 'msvc'
        # disable std::iterator implementation warning
        _builtin_graphite_compile_args += '/wd4996'
    endif


    lib_builtin_graphite = static_library('builtin_graphite', _builtin_graphite_srcs,
        include_directories: _builtin_graphite_incdirs,
        cpp_args: _builtin_graphite_compile_args,
        build_by_default: false
    )

    DEP_GRAPHITE = declare_dependency(
        link_with: lib_builtin_graphite,
        include_directories: _builtin_graphite_dep_incdirs,
        compile_args: _builtin_graphite_compile_args
    )
else
    DEP_GRAPHITE = dependency('graphite2')
endif