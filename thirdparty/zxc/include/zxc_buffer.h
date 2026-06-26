/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_buffer.h
 * @brief Buffer-based (single-shot) compression and decompression API.
 *
 * This header exposes the simplest way to use ZXC: pass an entire input buffer
 * and receive the result in a single output buffer.  All functions in this
 * header are single-threaded and blocking.
 *
 * @par Typical usage
 * @code
 * // Compress
 * size_t bound = zxc_compress_bound(src_size);
 * void *dst    = malloc(bound);
 * zxc_compress_opts_t opts = { .level = ZXC_LEVEL_DEFAULT, .checksum_enabled = 1 };
 * int64_t csize = zxc_compress(src, src_size, dst, bound, &opts);
 *
 * // Decompress
 * uint64_t orig = zxc_get_decompressed_size(dst, csize);
 * void *out     = malloc(orig);
 * zxc_decompress_opts_t dopts = { .checksum_enabled = 1 };
 * int64_t dsize = zxc_decompress(dst, csize, out, orig, &dopts);
 * @endcode
 *
 * @see zxc_stream.h  for the streaming (multi-threaded) API.
 * @see zxc_pstream.h for single-threaded push-based streaming.
 */

#ifndef ZXC_BUFFER_H
#define ZXC_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#include "zxc_export.h"
#include "zxc_opts.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup library_info Library Information
 * @brief Runtime-queryable library metadata.
 *
 * These functions allow callers (including filesystem integrations)
 * to discover the supported compression level range and library version at
 * runtime, without relying on compile-time constants alone.
 * @{
 */

/**
 * @brief Returns the minimum supported compression level.
 *
 * Currently returns @ref ZXC_LEVEL_FASTEST (1).
 *
 * @return Minimum compression level value.
 */
ZXC_EXPORT int zxc_min_level(void);

/**
 * @brief Returns the maximum supported compression level.
 *
 * Currently returns @ref ZXC_LEVEL_DENSITY (6).
 *
 * @return Maximum compression level value.
 */
ZXC_EXPORT int zxc_max_level(void);

/**
 * @brief Returns the default compression level.
 *
 * Currently returns @ref ZXC_LEVEL_DEFAULT (3).
 *
 * @return Default compression level value.
 */
ZXC_EXPORT int zxc_default_level(void);

/**
 * @brief Returns the human-readable library version string.
 *
 * The returned pointer is a compile-time constant and must not be freed.
 * Format: "MAJOR.MINOR.PATCH" (e.g. "0.12.0").
 *
 * @return Null-terminated version string.
 */
ZXC_EXPORT const char* zxc_version_string(void);

/** @} */ /* end of library_info */

/**
 * @defgroup buffer_api Buffer API
 * @brief Single-shot, buffer-based compression and decompression.
 * @{
 */

/**
 * @brief Calculates the maximum theoretical compressed size for a given input.
 *
 * Useful for allocating output buffers before compression.
 * Accounts for file headers, block headers, and potential expansion
 * of incompressible data.
 *
 * @param[in] input_size Size of the input data in bytes.
 *
 * @return           Maximum required buffer size in bytes.
 */
ZXC_EXPORT uint64_t zxc_compress_bound(const size_t input_size);

/**
 * @brief Compresses a data buffer using the ZXC algorithm.
 *
 * This version uses standard size_t types and void pointers.
 * It executes in a single thread (blocking operation).
 * It writes the ZXC file header followed by compressed blocks.
 *
 * @param[in] src          Pointer to the source buffer.
 * @param[in] src_size     Size of the source data in bytes.
 * @param[out] dst          Pointer to the destination buffer.
 * @param[in] dst_capacity Maximum capacity of the destination buffer.
 * @param[in] opts         Compression options (NULL uses all defaults).
 *                         @c n_threads and the progress callback are ignored
 *                         (this call is single-threaded and blocking).
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return The number of bytes written to dst (>0 on success),
 *         or a negative zxc_error_t code (e.g., ZXC_ERROR_DST_TOO_SMALL) on failure.
 */
ZXC_EXPORT int64_t zxc_compress(const void* src, const size_t src_size, void* dst,
                                const size_t dst_capacity, const zxc_compress_opts_t* opts);

/**
 * @brief Decompresses a ZXC compressed buffer.
 *
 * This version uses standard size_t types and void pointers.
 * It executes in a single thread (blocking operation).
 * It expects a valid ZXC file header followed by compressed blocks.
 *
 * @param[in] src          Pointer to the source buffer containing compressed data.
 * @param[in] src_size      Size of the compressed data in bytes.
 * @param[out] dst          Pointer to the destination buffer.
 * @param[in] dst_capacity  Capacity of the destination buffer.
 * @param[in] opts          Decompression options (NULL uses all defaults).
 *                          @c n_threads and the progress callback are ignored
 *                          (this call is single-threaded and blocking).
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return The number of bytes written to dst (>0 on success),
 *         or a negative zxc_error_t code (e.g., ZXC_ERROR_CORRUPT_DATA) on failure.
 */
ZXC_EXPORT int64_t zxc_decompress(const void* src, const size_t src_size, void* dst,
                                  const size_t dst_capacity, const zxc_decompress_opts_t* opts);

/**
 * @brief Returns the decompressed size stored in a ZXC compressed buffer.
 *
 * This function reads the file footer to extract the original uncompressed size
 * without performing any decompression. Useful for allocating output buffers.
 *
 * @param[in] src       Pointer to the compressed data buffer.
 * @param[in] src_size  Size of the compressed data in bytes.
 *
 * @return The original uncompressed size in bytes, or 0 if the buffer is invalid
 *         or too small to contain a valid ZXC archive.
 */
ZXC_EXPORT uint64_t zxc_get_decompressed_size(const void* src, const size_t src_size);

/**
 * @brief Returns the dictionary ID stored in a ZXC compressed buffer.
 *
 * Reads the file header flag and dict_id field without decompressing.
 * Returns 0 if the file does not require a dictionary or the buffer is invalid.
 *
 * @param[in] src       Pointer to the compressed data buffer.
 * @param[in] src_size  Size of the compressed data in bytes.
 * @return Dictionary ID, or 0 if no dictionary is required.
 */
ZXC_EXPORT uint32_t zxc_get_dict_id(const void* src, size_t src_size);

/* ========================================================================= */
/*  Block-Level API (no file framing)                                        */
/* ========================================================================= */

/**
 * @defgroup block_api Block API
 * @brief Single-block compression/decompression without file framing.
 *
 * These functions compress or decompress a single independent block, producing
 * only the block header (8 bytes) + compressed payload + optional checksum (4 bytes).
 * No file header, EOF block, or footer is written.
 *
 * This API is designed for filesystem integrations where the filesystem manages its own block
 * indexing and each block is compressed independently.
 *
 * @par Typical usage
 * @code
 * // Compress a single filesystem block
 * zxc_cctx* cctx = zxc_create_cctx(NULL);
 * zxc_compress_opts_t opts = { .level = 3 };
 * size_t bound = zxc_compress_block_bound(block_size);
 * void *dst = malloc(bound);
 * int64_t csize = zxc_compress_block(cctx, block, block_size, dst, bound, &opts);
 *
 * // Decompress
 * zxc_dctx* dctx = zxc_create_dctx();
 * int64_t dsize = zxc_decompress_block(dctx, dst, csize, out, block_size, NULL);
 *
 * zxc_free_cctx(cctx);
 * zxc_free_dctx(dctx);
 * @endcode
 * @{
 */

/* Forward declarations for context types (defined below). */
/** @brief Opaque reusable compression context (see @ref zxc_create_cctx). */
typedef struct zxc_cctx_s zxc_cctx;
/** @brief Opaque reusable decompression context (see @ref zxc_create_dctx). */
typedef struct zxc_dctx_s zxc_dctx;

/**
 * @brief Returns the maximum compressed size for a single block.
 *
 * Unlike zxc_compress_bound(), this does NOT include file header,
 * EOF block, or footer overhead.  Use this to size the destination
 * buffer for zxc_compress_block().
 *
 * @param[in] input_size Size of the uncompressed block in bytes
 *                       (must be <= @ref ZXC_BLOCK_SIZE_MAX).
 * @return Upper bound on compressed block size, or 0 if @p input_size is
 *         out of range for the Block API
 *         (@p input_size > @ref ZXC_BLOCK_SIZE_MAX) or would overflow.
 */
ZXC_EXPORT uint64_t zxc_compress_block_bound(size_t input_size);

/**
 * @brief Returns the minimum destination capacity required by
 *        zxc_decompress_block() for a block of @p uncompressed_size bytes.
 *
 * The decoder uses speculative (wild-copy) writes on its fast path and
 * therefore needs a tail pad beyond the declared uncompressed size.
 * Passing exactly @c uncompressed_size as @c dst_capacity forces the slow
 * tail path and may trigger @ref ZXC_ERROR_OVERFLOW on some inputs.
 *
 * Use this helper to size the destination buffer. The returned value is
 * guaranteed to enable the fastest decode path without aliasing or
 * overrun checks tripping.
 *
 * @param[in] uncompressed_size Original uncompressed block size in bytes
 *                              (must be <= @ref ZXC_BLOCK_SIZE_MAX).
 * @return Minimum @c dst_capacity to pass to zxc_decompress_block(), or 0 if
 *         @p uncompressed_size is out of range for the Block API
 *         (@p uncompressed_size > @ref ZXC_BLOCK_SIZE_MAX) or would overflow.
 */
ZXC_EXPORT uint64_t zxc_decompress_block_bound(const size_t uncompressed_size);

/**
 * @brief Compresses a single block without file framing.
 *
 * Output format: @c block_header(8B) + payload + optional @c checksum(4B).
 * The output can be decompressed with zxc_decompress_block().
 *
 * The Block API processes a single format-conformant block at a time:
 * @p src_size must not exceed @ref ZXC_BLOCK_SIZE_MAX (2 MiB). For larger
 * payloads, use the frame API (zxc_compress) or the streaming API
 * (zxc_cstream_*), both of which chunk transparently into compliant blocks.
 *
 * @param[in,out] cctx         Reusable compression context.
 * @param[in]     src          Source data.
 * @param[in]     src_size     Source data size in bytes
 *                             (must be in [1, @ref ZXC_BLOCK_SIZE_MAX]).
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of the destination buffer
 *                             (use zxc_compress_block_bound() to size).
 * @param[in]     opts         Compression options, or NULL for defaults.
 *                             Only @c level, @c block_size, and
 *                             @c checksum_enabled are used.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Compressed block size in bytes (> 0) on success,
 *         or a negative @ref zxc_error_t code on failure.
 *         Returns @ref ZXC_ERROR_BAD_BLOCK_SIZE if
 *         @p src_size > @ref ZXC_BLOCK_SIZE_MAX.
 */
ZXC_EXPORT int64_t zxc_compress_block(zxc_cctx* cctx, const void* src, size_t src_size, void* dst,
                                      size_t dst_capacity, const zxc_compress_opts_t* opts);

/**
 * @brief Decompresses a single block produced by zxc_compress_block().
 *
 * The Block API decompresses a single format-conformant block at a time:
 * @p dst_capacity must not exceed @ref ZXC_BLOCK_SIZE_MAX +
 * @ref ZXC_DECOMPRESS_TAIL_PAD (the size returned by
 * zxc_decompress_block_bound() for the maximum block size). For payloads
 * produced by the frame or streaming APIs, use zxc_decompress instead.
 *
 * @param[in,out] dctx         Reusable decompression context.
 * @param[in]     src          Compressed block data.
 * @param[in]     src_size     Compressed data size in bytes.
 * @param[out]    dst          Destination buffer for decompressed data.
 * @param[in]     dst_capacity Capacity of the destination buffer (must be
 *                             at least the original uncompressed size,
 *                             and at most @ref ZXC_BLOCK_SIZE_MAX +
 *                             @ref ZXC_DECOMPRESS_TAIL_PAD).
 * @param[in]     opts         Decompression options (NULL for defaults).
 *                             Only @c checksum_enabled is used.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Decompressed size in bytes (> 0) on success,
 *         or a negative @ref zxc_error_t code on failure.
 *         Returns @ref ZXC_ERROR_BAD_BLOCK_SIZE if @p dst_capacity exceeds
 *         the per-block limit.
 */
ZXC_EXPORT int64_t zxc_decompress_block(zxc_dctx* dctx, const void* src, size_t src_size, void* dst,
                                        size_t dst_capacity, const zxc_decompress_opts_t* opts);

/**
 * @brief Decompresses a single block with a strict-sized destination buffer.
 *
 * Identical semantics to zxc_decompress_block() but accepts
 * @p dst_capacity == @c uncompressed_size (no trailing @c ZXC_DECOMPRESS_TAIL_PAD
 * required). Intended for integrations whose destination buffer cannot be
 * oversized (for example, decoding into an exactly-sized, page-aligned
 * region). Here "in-place" means a tightly-sized destination, not an
 * overlapping @p src / @p dst (see @note below).
 *
 * This path is slightly slower than zxc_decompress_block() on the same input
 * because it avoids the wild-copy overshoot that the fast decoder relies on.
 * Output is bit-identical to zxc_decompress_block().
 *
 * RAW blocks transparently forward to zxc_decompress_block(); only
 * GLO/GHI use the strict-tail decoder path.
 *
 * Strict-tail variant: @p dst_capacity is the exact uncompressed size with
 * no tail-pad margin, so the upper limit is @ref ZXC_BLOCK_SIZE_MAX (not
 * @c MAX+TAIL_PAD as for zxc_decompress_block).
 *
 * @param[in,out] dctx         Reusable decompression context.
 * @param[in]     src          Compressed block data.
 * @param[in]     src_size     Compressed data size in bytes.
 * @param[out]    dst          Destination buffer for decompressed data.
 * @param[in]     dst_capacity Capacity of the destination buffer (must be
 *                             at least the original uncompressed size,
 *                             and at most @ref ZXC_BLOCK_SIZE_MAX; unlike
 *                             zxc_decompress_block, no trailing tail-pad
 *                             margin is required).
 * @param[in]     opts         Decompression options (NULL for defaults).
 *                             Only @c checksum_enabled is used.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Decompressed size in bytes (> 0) on success,
 *         or a negative @ref zxc_error_t code on failure.
 *         Returns @ref ZXC_ERROR_BAD_BLOCK_SIZE if
 *         @p dst_capacity > @ref ZXC_BLOCK_SIZE_MAX.
 */
ZXC_EXPORT int64_t zxc_decompress_block_safe(zxc_dctx* dctx, const void* src, const size_t src_size,
                                             void* dst, const size_t dst_capacity,
                                             const zxc_decompress_opts_t* opts);

/**
 * @brief Estimates the peak memory used by compression for a given block & level.
 *
 * Returns the total bytes reserved by @ref zxc_compress_block for a block of
 * @p src_size bytes: all per-chunk working buffers (chain table, literals,
 * sequence/token/offset/extras buffers) plus the fixed hash tables and
 * cache-line alignment padding. At @p level >= 6 the value also includes the
 * `opt_scratch` region (~8.125 x @p src_size bytes) used by the price-based
 * optimal parser. That region is lazy-allocated on the first level-6 call
 * and reused across blocks for the lifetime of the cctx. Scales roughly
 * linearly with @p src_size.
 *
 * Intended for integrators that need an accurate memory-budget figure.
 *
 * @param[in] src_size Uncompressed block size in bytes.
 * @param[in] level    Compression level (1..6). Levels <= 5 share the same
 *                     persistent cctx footprint; level 6 adds the optimal-
 *                     parser scratch.
 * @return Estimated peak cctx memory usage in bytes, or 0 if @p src_size is 0.
 */
ZXC_EXPORT uint64_t zxc_estimate_cctx_size(size_t src_size, int level);

/** @} */ /* end of block_api */

/* ========================================================================= */
/*  Reusable Context API (opaque, heap-allocated)                            */
/* ========================================================================= */

/**
 * @defgroup context_api Reusable Context API
 * @brief Opaque, reusable compression / decompression contexts.
 *
 * This API eliminates per-call allocation overhead by letting callers retain
 * a context across multiple operations.  The internal layout is hidden behind
 * an opaque pointer.
 *
 * @{
 */

/* --- Compression context ------------------------------------------------- */

/**
 * @brief Creates a new reusable compression context.
 *
 * When @p opts is non-NULL the context pre-allocates all internal buffers
 * using the supplied level, block_size, and checksum_enabled settings.
 * When @p opts is NULL, allocation is deferred to the first call to
 * zxc_compress_cctx().
 *
 * The returned context must be freed with zxc_free_cctx().
 *
 * @param[in] opts  Compression options for eager init, or NULL for lazy init.
 * @return Pointer to the new context, or @c NULL on allocation failure.
 */
ZXC_EXPORT zxc_cctx* zxc_create_cctx(const zxc_compress_opts_t* opts);

/**
 * @brief Frees a compression context and all associated resources.
 *
 * It is safe to pass @c NULL; the call is a no-op in that case.
 *
 * @param[in] cctx Context to free.
 */
ZXC_EXPORT void zxc_free_cctx(zxc_cctx* cctx);

/**
 * @brief Compresses data using a reusable context.
 *
 * Identical to zxc_compress() but reuses the internal buffers from @p cctx,
 * avoiding per-call malloc/free overhead.  The context automatically
 * re-initializes when block_size or level changes between calls.
 *
 * Options are **sticky**: settings passed via @p opts are remembered and
 * reused on subsequent calls where @p opts is NULL.  The initial sticky
 * values come from the @p opts passed to zxc_create_cctx().
 *
 * @param[in,out] cctx         Reusable compression context.
 * @param[in]     src          Source data.
 * @param[in]     src_size     Source data size in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of the destination buffer.
 * @param[in]     opts         Compression options, or NULL to reuse
 *                             settings from create / last call.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Compressed size in bytes (> 0) on success,
 *         or a negative @ref zxc_error_t code on failure.
 */
ZXC_EXPORT int64_t zxc_compress_cctx(zxc_cctx* cctx, const void* src, size_t src_size, void* dst,
                                     size_t dst_capacity, const zxc_compress_opts_t* opts);

/* --- Decompression context ----------------------------------------------- */

/**
 * @brief Creates a new reusable decompression context.
 *
 * @return Pointer to the new context, or @c NULL on allocation failure.
 */
ZXC_EXPORT zxc_dctx* zxc_create_dctx(void);

/**
 * @brief Frees a decompression context and all associated resources.
 *
 * It is safe to pass @c NULL.
 *
 * @param[in] dctx Context to free.
 */
ZXC_EXPORT void zxc_free_dctx(zxc_dctx* dctx);

/**
 * @brief Decompresses data using a reusable context.
 *
 * Identical to zxc_decompress() but reuses buffers from @p dctx.
 *
 * @param[in,out] dctx         Reusable decompression context.
 * @param[in]     src          Compressed data.
 * @param[in]     src_size     Compressed data size in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of the destination buffer.
 * @param[in]     opts         Decompression options (NULL for defaults).
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Decompressed size in bytes (> 0) on success,
 *         or a negative @ref zxc_error_t code on failure.
 */
ZXC_EXPORT int64_t zxc_decompress_dctx(zxc_dctx* dctx, const void* src, size_t src_size, void* dst,
                                       size_t dst_capacity, const zxc_decompress_opts_t* opts);

/* ========================================================================= */
/*  Static Context API (caller-allocated workspace)                          */
/* ========================================================================= */

/**
 * @defgroup static_context_api Static Context API
 * @brief Caller-allocated, fixed-footprint compression / decompression
 *        contexts.
 *
 * Mirrors the dynamic Reusable Context API but places the entire context
 * (handle + persistent buffers) inside a single buffer allocated and owned
 * by the caller.  This pattern is mandatory for environments where the
 * library cannot call into the host allocator on the hot path: Linux
 * kernel filesystems (one workspace per mount, served via @c vmalloc /
 * @c kmalloc up front), embedded targets without a heap (`.bss` or
 * stack-allocated workspace), sandboxed runtimes with a fixed memory
 * budget, etc.
 *
 * The trade-off vs the dynamic API: the workspace is pinned to a single @c block_size and @c level
 * at init time; subsequent compress/decompress calls cannot enlarge the footprint, so a workload
 * that needs to mix block sizes must size the workspace for the maximum block_size up front.
 *
 * @par Typical usage
 * @code
 * size_t ws_sz = zxc_static_cctx_workspace_size(64 * 1024, ZXC_LEVEL_DEFAULT);
 * void *ws = aligned_alloc(64, ws_sz);                   // or kmalloc, vmalloc, .bss
 * zxc_compress_opts_t opts = { .level = ZXC_LEVEL_DEFAULT, .block_size = 64 * 1024 };
 * zxc_cctx *cctx = zxc_init_static_cctx(ws, ws_sz, &opts);
 *
 * for (each block) zxc_compress_cctx(cctx, src, n, dst, cap, NULL);
 *
 * // zxc_free_cctx is a no-op on a static cctx; the caller owns @c ws.
 * free(ws);
 * @endcode
 * @{
 */

/**
 * @brief Returns the exact byte count required by a static compression
 *        workspace for the given @p block_size and @p level.
 *
 * The value is the sum of the opaque @ref zxc_cctx wrapper plus every
 * persistent sub-buffer the library would partition (hash tables, chain
 * table, sequence buffers, literal scratch, plus the optimal-parser
 * scratch at @ref ZXC_LEVEL_DENSITY). Round up to your allocator's
 * alignment before calling @c posix_memalign / @c aligned_alloc, the
 * workspace must be at least cache-line aligned.
 *
 * @param[in] block_size  Block size in bytes (must satisfy the regular
 *                        block-size constraints: power of two in
 *                        [@ref ZXC_BLOCK_SIZE_MIN, @ref ZXC_BLOCK_SIZE_MAX]).
 * @param[in] level       Compression level (1..6); higher levels at
 *                        @ref ZXC_LEVEL_DENSITY add the optimal-parser
 *                        scratch (~8.125 x block_size).
 * @return Workspace size in bytes, or 0 if either argument is invalid.
 */
ZXC_EXPORT size_t zxc_static_cctx_workspace_size(const size_t block_size, const int level);

/**
 * @brief Initialises a compression context inside a caller-supplied
 *        workspace.
 *
 * @p workspace_size must be at least @ref zxc_static_cctx_workspace_size
 * for the same @c block_size and @c level.  The workspace must remain
 * valid for the lifetime of the returned handle and must be cache-line
 * (64-byte) aligned.  The caller owns the workspace; @ref zxc_free_cctx
 * is a no-op on the returned handle.
 *
 * @par Locked parameters
 * The @c block_size, @c level, and @c checksum_enabled fields of @p opts
 * are pinned at init time.  Subsequent @ref zxc_compress_cctx calls that
 * pass options requesting a different @c block_size return
 * @ref ZXC_ERROR_BAD_BLOCK_SIZE without re-initialising.  A different
 * @c level / @c checksum_enabled is honoured per-call without
 * re-partitioning.
 *
 * @param[in,out] workspace       Caller-allocated buffer, cache-line aligned.
 * @param[in]     workspace_size  Capacity of @p workspace in bytes.
 * @param[in]     opts            Must be non-NULL: @c block_size and
 *                                @c level must be set explicitly to size
 *                                the workspace correctly.
 * @return Handle pointing inside @p workspace on success, or @c NULL if
 *         the workspace is too small or the options are invalid.
 */
ZXC_EXPORT zxc_cctx* zxc_init_static_cctx(void* workspace, const size_t workspace_size,
                                          const zxc_compress_opts_t* opts);

/**
 * @brief Returns the exact byte count required by a static decompression
 *        workspace for the given @p block_size.
 *
 * Unlike the compression variant, this size is independent of the source
 * archive's level: @c lit_buffer is always provisioned worst-case because
 * the decoder cannot predict the per-block literal encoding until it sees
 * each block header.
 *
 * @param[in] block_size  Maximum block size the decoder will encounter
 *                        (must satisfy the regular block-size constraints).
 * @return Workspace size in bytes, or 0 if @p block_size is invalid.
 */
ZXC_EXPORT size_t zxc_static_dctx_workspace_size(const size_t block_size);

/**
 * @brief Initialises a decompression context inside a caller-supplied
 *        workspace.
 *
 * @p workspace_size must be at least @ref zxc_static_dctx_workspace_size
 * for the same @p block_size.  The workspace must remain valid for the
 * lifetime of the returned handle and must be cache-line aligned.  The
 * caller owns the workspace; @ref zxc_free_dctx is a no-op on the
 * returned handle.
 *
 * @par Locked block size
 * @p block_size is pinned at init time: feeding the returned handle an
 * archive whose header declares a different @c block_size returns
 * @ref ZXC_ERROR_BAD_BLOCK_SIZE.
 *
 * @param[in,out] workspace       Caller-allocated buffer, cache-line aligned.
 * @param[in]     workspace_size  Capacity of @p workspace in bytes.
 * @param[in]     block_size      Block size the decoder will accept.
 * @return Handle pointing inside @p workspace on success, or @c NULL if
 *         the workspace is too small or @p block_size is invalid.
 */
ZXC_EXPORT zxc_dctx* zxc_init_static_dctx(void* workspace, const size_t workspace_size,
                                          const size_t block_size);

/** @} */ /* end of static_context_api */
/** @} */ /* end of context_api */
/** @} */ /* end of buffer_api */

#ifdef __cplusplus
}
#endif

#endif  // ZXC_BUFFER_H