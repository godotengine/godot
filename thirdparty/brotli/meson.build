if get_option('builtin_brotli')
    lib_brotli = static_library('builtin_brotli',
        'common/constants.c',
        'common/context.c',
        'common/dictionary.c',
        'common/platform.c',
        'common/shared_dictionary.c',
        'common/transform.c',
        'dec/bit_reader.c',
        'dec/decode.c',
        'dec/huffman.c',
        'dec/state.c',
        include_directories: 'include',
    )

    DEP_BROTLI = declare_dependency(link_with: lib_brotli,
                                    include_directories: 'include')

else
    DEP_BROTLI = [
        dependency('libbrotlidec'),
    ]
endif
