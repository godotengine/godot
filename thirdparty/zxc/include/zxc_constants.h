/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_constants.h
 * @brief Public constants: library version and compression levels.
 *
 * Include this header to query the library version at compile time or to
 * reference the predefined compression-level constants used throughout the API.
 */

#ifndef ZXC_CONSTANTS_H
#define ZXC_CONSTANTS_H

/**
 * @defgroup version Library Version
 * @brief Compile-time version information.
 * @{
 */

/** @brief Major version number. */
#define ZXC_VERSION_MAJOR 0
/** @brief Minor version number. */
#define ZXC_VERSION_MINOR 12
/** @brief Patch version number. */
#define ZXC_VERSION_PATCH 0

/** @cond INTERNAL */
#define ZXC_STR_HELPER(x) #x
#define ZXC_STR(x) ZXC_STR_HELPER(x)
/** @endcond */

/**
 * @brief Human-readable version string in "MAJOR.MINOR.PATCH" form (e.g. "0.12.0").
 */
#define ZXC_LIB_VERSION_STR    \
    ZXC_STR(ZXC_VERSION_MAJOR) \
    "." ZXC_STR(ZXC_VERSION_MINOR) "." ZXC_STR(ZXC_VERSION_PATCH)

/** @} */ /* end of version */

/**
 * @defgroup block_size Block Size
 * @brief Block size constraints for compression.
 *
 * Block size must be a power of two in range
 * [@ref ZXC_BLOCK_SIZE_MIN, @ref ZXC_BLOCK_SIZE_MAX].
 * Pass 0 to any API to use @ref ZXC_BLOCK_SIZE_DEFAULT.
 * @{
 */
/** @brief log2(ZXC_BLOCK_SIZE_MIN) - exponent code for minimum block size. */
#define ZXC_BLOCK_SIZE_MIN_LOG2 12
/** @brief log2(ZXC_BLOCK_SIZE_MAX) - exponent code for maximum block size. */
#define ZXC_BLOCK_SIZE_MAX_LOG2 21
/** @brief Default block size (512 KB). */
#define ZXC_BLOCK_SIZE_DEFAULT (512 * 1024)
/** @brief Minimum allowed block size (4 KB = 2^12). */
#define ZXC_BLOCK_SIZE_MIN (1U << ZXC_BLOCK_SIZE_MIN_LOG2)
/** @brief Maximum allowed block size (2 MB = 2^21). */
#define ZXC_BLOCK_SIZE_MAX (1U << ZXC_BLOCK_SIZE_MAX_LOG2)
/** @} */ /* end of block_size */

/**
 * @defgroup dictionary Dictionary
 * @brief Constants for pre-trained dictionary support.
 * @{
 */
/** @brief Maximum dictionary content size in bytes (64 KB - 1).
 *
 * Bounded to a 16-bit value (65535) by two constraints that both cap at the
 * same number: the `.zxd` header stores the content size in a 16-bit field, and
 * LZ77 match offsets are 16-bit (max distance 65535). */
#define ZXC_DICT_SIZE_MAX ((1U << 16) - 1U)
/** @brief Size of the .zxd dictionary file header in bytes. */
#define ZXC_DICT_HEADER_SIZE 16
/** @brief Size in bytes of a packed literal Huffman code-lengths table
 *         (256 symbols, 4 bits each): the shared table carried by a .zxd
 *         file and, internally, the per-block lengths header. See
 *         zxc_train_dict_huf() / zxc_dict_huf(). */
#define ZXC_HUF_TABLE_SIZE 128
/** @} */ /* end of dictionary */

/**
 * @defgroup threading Threading Limits
 * @brief Bounds on thread-count parameters accepted by the streaming APIs.
 * @{
 */
/** @brief Maximum value accepted for `n_threads` in `zxc_stream_compress`
 *  / `zxc_stream_decompress`. Higher values are clamped to `ZXC_MAX_THREADS`. */
#define ZXC_MAX_THREADS 512
/** @} */ /* end of threading */

/**
 * @defgroup file_format File Format Constants
 * @brief On-disk byte sizes of the archive header and footer.
 *
 * @{
 */
/** @brief File header size: Magic(4) + Version(1) + Chunk(1) + Flags(1) + Reserved(7) + CRC(2). */
#define ZXC_FILE_HEADER_SIZE 16
/** @brief File footer size: original_size(8) + global_checksum(4). */
#define ZXC_FILE_FOOTER_SIZE 12
/** @} */ /* end of file_format */

/**
 * @defgroup levels Compression Levels
 * @brief Predefined compression levels for the ZXC library.
 *
 * Higher levels trade encoding speed for better compression ratio.
 * All levels produce data that can be decompressed at the same speed.
 * @{
 */

/**
 * @brief Enumeration of ZXC compression levels.
 *
 * Use one of these constants as the @p level parameter of
 * zxc_compress() or zxc_stream_compress().
 */
typedef enum {
    ZXC_LEVEL_FASTEST = 1,  /**< Fastest compression, best for real-time applications. */
    ZXC_LEVEL_FAST = 2,     /**< Fast compression, good for real-time applications. */
    ZXC_LEVEL_DEFAULT = 3,  /**< Recommended: ratio > LZ4, decode speed > LZ4. */
    ZXC_LEVEL_BALANCED = 4, /**< Good ratio, good decode speed. */
    ZXC_LEVEL_COMPACT = 5,  /**< High density. Best for storage/firmware/assets. */
    ZXC_LEVEL_DENSITY = 6   /**< Maximum density: Huffman-coded literals on top of COMPACT. */
} zxc_compression_level_t;

/** @} */ /* end of levels */

#endif  // ZXC_CONSTANTS_H
