/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_opts.h
 * @brief Shared option structures for the ZXC compression APIs.
 *
 * Defines @ref zxc_compress_opts_t, @ref zxc_decompress_opts_t and
 * @ref zxc_progress_callback_t. These types are consumed by every public
 * ZXC API (one-shot buffer, multi-threaded @c FILE* streaming, push
 * streaming, seekable).
 *
 * This header is never used in isolation: include the API header you
 * actually use (@c zxc_buffer.h, @c zxc_stream.h, @c zxc_pstream.h, ...)
 * which pulls this one in transitively.
 */

#ifndef ZXC_OPTS_H
#define ZXC_OPTS_H

#include <stddef.h>
#include <stdint.h>

#include "zxc_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Progress callback function type.
 *
 * This callback is invoked periodically during compression/decompression to report
 * progress. It is called from the writer thread after each block is processed.
 *
 * @param[in] bytes_processed Total input bytes processed so far.
 * @param[in] bytes_total     Total input bytes to process (0 if unknown, e.g., stdin).
 * @param[in] user_data       User-provided context pointer (passed through from API call).
 *
 * @note The callback should be fast and non-blocking. Avoid heavy I/O or mutex locks.
 */
typedef void (*zxc_progress_callback_t)(uint64_t bytes_processed, uint64_t bytes_total,
                                        const void* user_data);

/**
 * @brief Options for streaming compression.
 *
 * Zero-initialise for safe defaults: level 0 maps to ZXC_LEVEL_DEFAULT (3),
 * block_size 0 maps to ZXC_BLOCK_SIZE_DEFAULT, n_threads 0 means
 * auto-detect, and all other fields are disabled.
 *
 * @code
 * zxc_compress_opts_t opts = { .level = ZXC_LEVEL_COMPACT };
 * zxc_stream_compress(f_in, f_out, &opts);
 * @endcode
 */
typedef struct {
    int n_threads;     /**< Worker thread count (0 = auto-detect CPU cores). */
    int level;         /**< Compression level 1-6 (0 = default, ZXC_LEVEL_DEFAULT). */
    size_t block_size; /**< Block size in bytes (0 = default ZXC_BLOCK_SIZE_DEFAULT). Must be power
                          of 2, [4KB - 2MB]. */
    int checksum_enabled; /**< 1 to enable per-block and global checksums, 0 to disable. */
    int seekable;         /**< 1 to append a seek table for random-access decompression. */
    const void* dict;     /**< Pre-trained dictionary content (NULL = none). */
    size_t dict_size;     /**< Dictionary size in bytes (0 = none, max ZXC_DICT_SIZE_MAX). */
    const void* dict_huf; /**< Optional shared literal Huffman table: 128-byte packed
                               code-lengths header from zxc_train_dict_huf() /
                               zxc_dict_huf() (NULL = none; ignored without dict).
                               Becomes part of the archive's dict_id binding. */
    zxc_progress_callback_t progress_cb; /**< Optional progress callback (NULL to disable). */
    void* user_data;                     /**< User context pointer passed to progress_cb. */
} zxc_compress_opts_t;

/**
 * @brief Options for streaming decompression.
 *
 * Zero-initialise for safe defaults.
 *
 * @code
 * zxc_decompress_opts_t opts = { .checksum_enabled = 1 };
 * zxc_stream_decompress(f_in, f_out, &opts);
 * @endcode
 */
typedef struct {
    int n_threads;        /**< Worker thread count (0 = auto-detect CPU cores). */
    int checksum_enabled; /**< 1 to verify per-block and global checksums, 0 to skip. */
    const void* dict;     /**< Pre-trained dictionary content (NULL = none). */
    size_t dict_size;     /**< Dictionary size in bytes (0 = none). */
    const void* dict_huf; /**< Optional shared literal Huffman table: 128-byte packed
                               code-lengths header matching the one used at
                               compression time (NULL = none; ignored without dict). */
    zxc_progress_callback_t progress_cb; /**< Optional progress callback (NULL to disable). */
    void* user_data;                     /**< User context pointer passed to progress_cb. */
} zxc_decompress_opts_t;

/**
 * @brief Returns `sizeof(zxc_compress_opts_t)` as compiled into the library.
 *
 * Layout guard for bindings that mirror the options structs by hand (raw FFI
 * declarations, byte-offset serialization) instead of compiling against this
 * header: comparing the mirrored size against this value at load time turns a
 * silent layout drift (undefined behaviour) into an immediate, explicit error.
 */
ZXC_EXPORT size_t zxc_compress_opts_size(void);

/**
 * @brief Returns `sizeof(zxc_decompress_opts_t)` as compiled into the library.
 * @see zxc_compress_opts_size
 */
ZXC_EXPORT size_t zxc_decompress_opts_size(void);

#ifdef __cplusplus
}
#endif

#endif  // ZXC_OPTS_H
