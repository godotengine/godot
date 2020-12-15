if get_option('builtin_zstd')
    _zstd_srcs = files([
        'common/debug.c',
        'common/entropy_common.c',
        'common/error_private.c',
        'common/fse_decompress.c',
        'common/pool.c',
        'common/threading.c',
        'common/xxhash.c',
        'common/zstd_common.c',
        'compress/fse_compress.c',
        'compress/hist.c',
        'compress/huf_compress.c',
        'compress/zstd_compress.c',
        'compress/zstd_double_fast.c',
        'compress/zstd_fast.c',
        'compress/zstd_lazy.c',
        'compress/zstd_ldm.c',
        'compress/zstd_opt.c',
        'compress/zstdmt_compress.c',
        'compress/zstd_compress_literals.c',
        'compress/zstd_compress_sequences.c',
        'compress/zstd_compress_superblock.c',
        'decompress/huf_decompress.c',
        'decompress/zstd_ddict.c',
        'decompress/zstd_decompress_block.c',
        'decompress/zstd_decompress.c',
    ])

    _zstd_incdirs = include_directories('.')

    #_zstd_c_args = ['-DZSTD_STATIC_LINKING_ONLY']
    
    _lib_zstd = static_library('zstd', _zstd_srcs, include_directories: _zstd_incdirs, build_by_default: false)

    DEP_ZSTD = declare_dependency(link_with: _lib_zstd, include_directories: _zstd_incdirs)
else
    DEP_ZSTD = dependency('libzstd')
endif