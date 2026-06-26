/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_dict.h
 * @brief Pre-trained dictionary API for ZXC compression.
 *
 * Provides functions to train, save, load, and identify dictionaries that
 * improve compression ratio on small, similar payloads. Dictionaries are
 * stored as external `.zxd` files and referenced by a 32-bit ID in the
 * ZXC file header.
 *
 * A dictionary contains raw byte content that prefills the LZ77 sliding
 * window at the start of each block, giving the compressor immediate
 * access to representative patterns without waiting for them to appear
 * in the input stream.
 *
 * @code
 * // Train a dictionary from a corpus of JSON samples
 * void* dict_buf = malloc(32768);
 * int64_t dict_sz = zxc_train_dict(samples, sizes, n, dict_buf, 32768);
 *
 * // Train the shared literal Huffman table on the same corpus
 * uint8_t huf[ZXC_HUF_TABLE_SIZE];
 * zxc_train_dict_huf(samples, sizes, n, dict_buf, dict_sz, huf);
 *
 * // Save to .zxd file (content + table)
 * void* zxd = malloc(zxc_dict_save_bound(dict_sz));
 * int64_t zxd_sz = zxc_dict_save(dict_buf, dict_sz, huf, zxd, ...);
 *
 * // Use for compression
 * zxc_compress_opts_t opts = {
 *     .level = 6, .dict = dict_buf, .dict_size = dict_sz, .dict_huf = huf };
 * zxc_compress(src, src_size, dst, dst_capacity, &opts);
 * @endcode
 */

#ifndef ZXC_DICT_H
#define ZXC_DICT_H

#include <stddef.h>
#include <stdint.h>

#include "zxc_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup dict Dictionary
 * @brief Pre-trained dictionary training, serialization, and identification.
 * @{
 */

/**
 * @brief Compute the dictionary ID for the given content and optional table.
 *
 * The ID is a deterministic 32-bit hash stored in the ZXC file header so the
 * decoder can verify that the correct dictionary is provided at decompression
 * time. With @p huf_lengths NULL it hashes the raw content only (the in-memory
 * content-only dictionary of the buffer API). With a table it binds the
 * (content, table) pair: `hash(table, seed = hash(content))` -- the value
 * stored in `.zxd` files and in archives compressed with a shared table.
 *
 * @param[in] dict        Pointer to dictionary content.
 * @param[in] dict_size   Size in bytes.
 * @param[in] huf_lengths Shared literal Huffman table (@ref ZXC_HUF_TABLE_SIZE
 *                        bytes), or NULL for a content-only ID.
 * @return 32-bit dictionary ID. Returns 0 if @p dict is NULL or @p dict_size is 0.
 */
ZXC_EXPORT uint32_t zxc_dict_id(const void* dict, size_t dict_size, const void* huf_lengths);

/**
 * @brief Load and validate a `.zxd` dictionary file from a memory buffer.
 *
 * On success, @p content_out and @p huf_out (when non-NULL) point into the
 * input buffer (zero-copy); the caller must keep @p buf alive while they are in
 * use. A single call yields everything needed to (de)compress with the
 * dictionary, pass @p content_out / @p huf_out straight to the @c dict /
 * @c dict_huf option fields.
 *
 * @param[in]  buf              Buffer containing the .zxd file.
 * @param[in]  buf_size         Size of @p buf in bytes.
 * @param[out] content_out      Receives a pointer to the dictionary content.
 * @param[out] content_size_out Receives the content size in bytes.
 * @param[out] huf_out          Receives a pointer to the 128-byte shared Huffman
 *                              table (may be NULL if not needed).
 * @param[out] dict_id_out      Receives the dictionary ID (may be NULL).
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
ZXC_EXPORT int zxc_dict_load(const void* buf, size_t buf_size, const void** content_out,
                             size_t* content_size_out, const void** huf_out, uint32_t* dict_id_out);

/**
 * @brief Serialize dictionary content and its shared Huffman table to the
 *        `.zxd` file format.
 *
 * The 128-byte packed code-lengths table (from zxc_train_dict_huf()) is
 * mandatory and is appended after the content. The stored dict_id covers both
 * content and table, so archives compressed with this dictionary are bound to
 * the exact pair.
 *
 * @param[in]  content       Raw dictionary content.
 * @param[in]  content_size  Size of @p content in bytes (max ZXC_DICT_SIZE_MAX).
 * @param[in]  huf_lengths   128-byte packed Huffman code lengths (required).
 * @param[out] buf           Output buffer for the .zxd file.
 * @param[in]  buf_capacity  Capacity of @p buf (see zxc_dict_save_bound()).
 * @return Number of bytes written on success, or a negative @ref zxc_error_t code.
 */
ZXC_EXPORT int64_t zxc_dict_save(const void* content, size_t content_size, const void* huf_lengths,
                                 void* buf, size_t buf_capacity);

/**
 * @brief Returns the maximum .zxd file size for a given content size.
 *
 * @param[in] content_size Size of the dictionary content.
 * @return Total .zxd file size (header + content).
 */
ZXC_EXPORT size_t zxc_dict_save_bound(size_t content_size);

/**
 * @brief Returns the dictionary ID stored in a `.zxd` file buffer.
 *
 * Reads the dict_id field from the .zxd header without validating the full
 * file. Returns 0 if the buffer is too small or the magic word doesn't match.
 *
 * @param[in] buf       Buffer containing the .zxd file.
 * @param[in] buf_size  Size of @p buf in bytes.
 * @return Dictionary ID, or 0 if the buffer is not a valid .zxd file.
 */
ZXC_EXPORT uint32_t zxc_dict_get_id(const void* buf, size_t buf_size);

/**
 * @brief Train a dictionary from a corpus of samples.
 *
 * Analyzes the samples to select byte sequences that maximize LZ77 match
 * coverage. The resulting dictionary content can be passed directly to
 * zxc_compress_opts_t::dict or serialized with zxc_dict_save().
 *
 * @param[in]  samples        Array of pointers to sample buffers.
 * @param[in]  sample_sizes   Array of sample sizes in bytes.
 * @param[in]  n_samples      Number of samples.
 * @param[out] dict_buf       Output buffer for trained dictionary content.
 * @param[in]  dict_capacity  Capacity of @p dict_buf (max ZXC_DICT_SIZE_MAX).
 * @return Size of the trained dictionary on success, or a negative
 *         @ref zxc_error_t code.
 */
ZXC_EXPORT int64_t zxc_train_dict(const void* const* samples, const size_t* sample_sizes,
                                  size_t n_samples, void* dict_buf, size_t dict_capacity);

/**
 * @brief Train the shared literal Huffman table for an already-trained dictionary.
 *
 * Compresses the samples with @p dict and builds canonical Huffman code
 * lengths from the real post-LZ literal distribution. The resulting 128-byte
 * packed table can be embedded in a `.zxd` file via zxc_dict_save() and
 * passed to the compressor/decompressor via the `dict_huf` option field.
 * Blocks whose literals compress better with the shared table skip their
 * per-block 128-byte table header, which is decisive at small block sizes.
 *
 * @param[in]  samples         Array of pointers to sample buffers (typically
 *                             the same corpus used for zxc_train_dict()).
 * @param[in]  sample_sizes    Array of sample sizes in bytes.
 * @param[in]  n_samples       Number of samples.
 * @param[in]  dict            Trained dictionary content.
 * @param[in]  dict_size       Dictionary content size in bytes.
 * @param[out] huf_lengths_out Receives the 128-byte packed code-lengths table.
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
ZXC_EXPORT int zxc_train_dict_huf(const void* const* samples, const size_t* sample_sizes,
                                  size_t n_samples, const void* dict, size_t dict_size,
                                  uint8_t* huf_lengths_out);

/**
 * @brief One-call dictionary creation: train content + shared table, serialize
 *        to ready-to-write `.zxd` bytes.
 *
 * Convenience over the train/train-table/save sequence: it runs
 * zxc_train_dict() then zxc_train_dict_huf() (which depends on the trained
 * content) then zxc_dict_save(), writing a complete `.zxd` into @p zxd_buf.
 * Use zxc_dict_save_bound(ZXC_DICT_SIZE_MAX) for a safe @p zxd_capacity, or
 * size to the dictionary you expect. The lower-level primitives remain
 * available for advanced use (raw content-only dictionaries, retraining only
 * the table, or supplying externally-sourced content).
 *
 * @param[in]  samples       Array of pointers to sample buffers.
 * @param[in]  sample_sizes  Array of sample sizes in bytes.
 * @param[in]  n_samples     Number of samples.
 * @param[out] zxd_buf       Output buffer for the `.zxd` file.
 * @param[in]  zxd_capacity  Capacity of @p zxd_buf.
 * @return Number of `.zxd` bytes written on success, or a negative
 *         @ref zxc_error_t code.
 */
ZXC_EXPORT int64_t zxc_dict_train(const void* const* samples, const size_t* sample_sizes,
                                  size_t n_samples, void* zxd_buf, size_t zxd_capacity);

/**
 * @brief Returns a pointer to the shared Huffman table inside a `.zxd` buffer.
 *
 * Zero-copy accessor: the returned pointer aims into @p buf and is valid as
 * long as @p buf is. Returns NULL if the buffer is not a valid `.zxd` file or
 * carries no table.
 *
 * @param[in] buf       Buffer containing the .zxd file.
 * @param[in] buf_size  Size of @p buf in bytes.
 * @return Pointer to the 128-byte packed code-lengths table, or NULL.
 */
ZXC_EXPORT const void* zxc_dict_huf(const void* buf, size_t buf_size);

/** @} */ /* end of dict */

#ifdef __cplusplus
}
#endif

#endif /* ZXC_DICT_H */
