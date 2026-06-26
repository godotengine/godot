/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_common.c
 * @brief Shared library utilities: context management, header I/O,
 *        compress-bound calculation, and error-code name lookup.
 *
 * This translation unit contains the functions shared by both the buffer and
 * streaming APIs.  It is linked into every build of libzxc.
 */

#include "../../include/zxc_buffer.h"
#include "../../include/zxc_error.h"
#include "zxc_internal.h"

/*
 * ============================================================================
 * CONTEXT MANAGEMENT
 * ============================================================================
 */

/**
 * @brief Allocates memory aligned to the specified boundary.
 *
 * Uses `_aligned_malloc` on Windows and `posix_memalign` elsewhere.
 *
 * @param[in] size      Number of bytes to allocate.
 * @param[in] alignment Required alignment (must be a power of two).
 * @return Pointer to the allocated block, or @c NULL on failure.
 */
void* zxc_aligned_malloc(const size_t size, const size_t alignment) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    return ptr;
#endif
}

/**
 * @brief Frees memory previously allocated by zxc_aligned_malloc().
 *
 * @param[in] ptr Pointer returned by zxc_aligned_malloc() (may be @c NULL).
 */
void zxc_aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * @brief Returns @c sizeof(zxc_compress_opts_t) for ABI-safe allocation.
 *
 * Public API; see @c zxc_buffer.h. Lets callers (other languages, or a
 * different library version) size the options struct without knowing its layout.
 *
 * @return Size of @ref zxc_compress_opts_t in bytes.
 */
size_t zxc_compress_opts_size(void) { return sizeof(zxc_compress_opts_t); }

/**
 * @brief Returns @c sizeof(zxc_decompress_opts_t) for ABI-safe allocation.
 *
 * Public API; see @c zxc_buffer.h. Lets callers (other languages, or a
 * different library version) size the options struct without knowing its layout.
 *
 * @return Size of @ref zxc_decompress_opts_t in bytes.
 */
size_t zxc_decompress_opts_size(void) { return sizeof(zxc_decompress_opts_t); }

/*
 * Layout of the persistent buffer carved by every cctx/dctx init: both modes
 * (compress, decompress) compute the same offset table, used by the workspace
 * sizer and the in-place init.
 */
typedef struct {
    size_t total;
    /* mode == 0 (decompress) */
    size_t off_work;
    size_t off_lit_dctx;
    /* mode == 0 with dict only: prebuilt shared-dictionary Huffman decode table. */
    size_t off_huf_dict;
    /* mode == 1 (compress) */
    size_t off_hash_pos;
    size_t off_hash_tags;
    size_t off_chain;
    size_t off_seq_union;
    size_t off_extras;
    size_t off_lit_cctx;
    /* meaningful only when sz_opt > 0 (level >= ZXC_LEVEL_DENSITY). */
    size_t off_opt;
    /* both modes: [dict | data] concat scratch, present only when dict_size > 0. */
    size_t off_dict;
    /* Sub-buffer sizes (re-used by the partitioning step + zero-init). */
    size_t sz_hash_pos;
    size_t sz_hash_tags;
    size_t sz_opt;
    size_t sz_dict; /* 0 = no dictionary buffer. */
    size_t max_seq;
} zxc_cctx_layout_t;

/**
 * @brief Computes the single-allocation memory layout for a compression /
 *        decompression context.
 *
 * Walks the same partition order used by @ref zxc_cctx_init_in_workspace and
 * records each sub-buffer's offset plus the running @c total, so the sizing
 * query and the partitioning step share one source of truth and can never
 * disagree.
 *
 * Decompress (@p mode == 0) reserves @c work_buf and @c lit_buffer (both padded
 * for wild-copy overshoot) and, when @p dict_size > 0, the shared-dictionary
 * Huffman decode table. Compress (@p mode == 1) reserves the LZ match-finder
 * tables (hash positions, tags, chain), the sequence / extras / literal buffers
 * and - only at @c level >= ZXC_LEVEL_DENSITY - the optimal-parser scratch. A
 * @p dict_size > 0 appends the [dict | data] concat scratch in both modes.
 *
 * Every offset is cache-line aligned via @c ZXC_ALIGN_CL.
 *
 * @param[in] chunk_size  Block size in bytes.
 * @param[in] mode        1 = compression, 0 = decompression.
 * @param[in] level       Compression level (only consulted when @p mode == 1).
 * @param[in] dict_size   Dictionary prefill size; when > 0 the layout includes
 *                        the [dict | data] concat buffer (and, on decompress,
 *                        the dictionary Huffman decode table).
 * @return Fully populated layout; @c .total is the required workspace size.
 */
static zxc_cctx_layout_t compute_cctx_layout(const size_t chunk_size, const int mode,
                                             const int level, const size_t dict_size) {
    zxc_cctx_layout_t layout = {0};

    if (mode == 0) {
        /* Decompress: work_buf + lit_buffer, both padded for wild-copy
         * overshoot and sized worst-case (chunk_size + ZXC_DECOMPRESS_TAIL_PAD).
         * lit_buffer is provisioned regardless of level because the decoder cannot
         * predict the per-block literal encoding (RAW / RLE / HUFFMAN). */
        const size_t sz_work = chunk_size + ZXC_DECOMPRESS_TAIL_PAD;
        const size_t sz_lit = chunk_size + ZXC_PAD_SIZE;

        layout.off_work = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_work);
        layout.off_lit_dctx = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_lit);
        /* Shared-dictionary Huffman decode table: built once per context by
         * zxc_cctx_attach_dict_huf, read by HUFFMAN_DICT literal sections. */
        if (dict_size > 0) {
            layout.off_huf_dict = layout.total;
            layout.total += ZXC_ALIGN_CL(ZXC_HUF_DEC_TABLE_SIZE * sizeof(zxc_huf_dec_entry_t));
        }
    } else {
        /* Compress: 6 partitions + optional opt_scratch at level >= ZXC_LEVEL_DENSITY. */
        const uint32_t offset_bits = zxc_log2_u32((uint32_t)chunk_size);
        layout.max_seq = chunk_size / ZXC_LZ_MIN_MATCH_LEN + 16;
        layout.sz_hash_pos = ZXC_LZ_HASH_SIZE * sizeof(uint32_t);
        layout.sz_hash_tags = ZXC_LZ_HASH_SIZE * sizeof(uint8_t);
        const size_t sz_chain = ZXC_LZ_WINDOW_SIZE * sizeof(uint16_t);
        /* buf_sequences (GHI, level <= ZXC_LEVEL_FAST) aliases buf_offsets + buf_tokens (GLO,
         * level >= ZXC_LEVEL_DEFAULT). Mutually exclusive per block; sized for the larger. */
        const size_t sz_seq_union = layout.max_seq * sizeof(uint32_t);
        const size_t vbyte_len = (offset_bits + 6) / 7;
        const size_t sz_extras = layout.max_seq * 2 * vbyte_len;
        const size_t sz_lit = chunk_size + ZXC_PAD_SIZE;

        /* opt_scratch (level >= ZXC_LEVEL_DENSITY only): DP arrays for the optimal parser, also
         * reused transiently as the package-merge scratch for the length-limited
         * Huffman code-length builder. Sized to the larger of the two demands.
         * The formula must stay in sync with zxc_estimate_cctx_size() and the
         * consumer in zxc_compress.c. */
        if (level >= ZXC_LEVEL_DENSITY) {
            const size_t sz_dp = ZXC_ALIGN_CL((chunk_size + 1) * sizeof(uint32_t));
            const size_t sz_pl = ZXC_ALIGN_CL((chunk_size + 1) * sizeof(uint16_t));
            const size_t sz_po = ZXC_ALIGN_CL((chunk_size + 1) * sizeof(uint16_t));
            const size_t n_bm_words = ZXC_BITMAP_WORDS(chunk_size + 1);
            const size_t sz_bm = ZXC_ALIGN_CL(n_bm_words * sizeof(uint64_t));
            const size_t dp_needed = sz_dp + sz_pl + sz_po + sz_bm;
            layout.sz_opt =
                (dp_needed > ZXC_HUF_BUILD_SCRATCH_SIZE) ? dp_needed : ZXC_HUF_BUILD_SCRATCH_SIZE;
        }

        layout.off_hash_pos = layout.total;
        layout.total += ZXC_ALIGN_CL(layout.sz_hash_pos);
        layout.off_hash_tags = layout.total;
        layout.total += ZXC_ALIGN_CL(layout.sz_hash_tags);
        layout.off_chain = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_chain);
        layout.off_seq_union = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_seq_union);
        layout.off_extras = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_extras);
        layout.off_lit_cctx = layout.total;
        layout.total += ZXC_ALIGN_CL(sz_lit);
        /* opt_scratch is appended last so it is absent for levels 1..5 (zero
         * waste on the common path) and only inflates the workspace at level 6. */
        if (layout.sz_opt) {
            layout.off_opt = layout.total;
            layout.total += ZXC_ALIGN_CL(layout.sz_opt);
        }
    }

    /* [dict | data] concat scratch (dict only). Compress chunk_size already
     * spans [dict | block]; decompress prepends dict to a (chunk + PAD) region. */
    if (dict_size > 0) {
        layout.sz_dict = (mode == 1) ? (chunk_size + ZXC_DECOMPRESS_TAIL_PAD)
                                     : (dict_size + chunk_size + ZXC_DECOMPRESS_TAIL_PAD);
        layout.off_dict = layout.total;
        layout.total += ZXC_ALIGN_CL(layout.sz_dict);
    }
    return layout;
}

/**
 * @brief Returns the workspace byte count required for the given parameters.
 *
 * Public contract documented at the declaration in @c zxc_internal.h. Thin
 * wrapper that returns @c compute_cctx_layout(...).total, or 0 when
 * @p chunk_size is 0.
 *
 * @param[in] chunk_size  Block size in bytes.
 * @param[in] mode        1 = compression, 0 = decompression.
 * @param[in] level       Compression level (only consulted when @p mode == 1).
 * @param[in] dict_size   Dictionary prefill size; > 0 includes the concat buffer.
 * @return Workspace size in bytes, or 0 if @p chunk_size is 0.
 */
size_t zxc_cctx_compute_workspace_size(const size_t chunk_size, const int mode, const int level,
                                       const size_t dict_size) {
    if (UNLIKELY(chunk_size == 0)) return 0;
    return compute_cctx_layout(chunk_size, mode, level, dict_size).total;
}

/**
 * @brief Partitions a caller-supplied workspace into a ready-to-use context.
 *
 * Public contract (alignment, lifetime, return codes) documented at the
 * declaration in @c zxc_internal.h. Computes the layout via
 * @ref compute_cctx_layout, rejects an undersized @p workspace, then carves the
 * sub-buffers out of it. @c ctx->memory_block stays NULL so @ref zxc_cctx_free
 * leaves the caller-owned workspace untouched.
 *
 * @param[out] ctx               Context to initialise.
 * @param[in]  workspace         Caller-allocated, cache-line-aligned buffer.
 * @param[in]  workspace_size    Capacity of @p workspace in bytes.
 * @param[in]  chunk_size        Block size in bytes.
 * @param[in]  mode              1 = compression, 0 = decompression.
 * @param[in]  level             Compression level (ignored when @p mode == 0).
 * @param[in]  checksum_enabled  Non-zero to enable checksum computation.
 * @param[in]  dict_size         Dictionary prefill size; > 0 carves the concat buffer.
 * @return @ref ZXC_OK, @ref ZXC_ERROR_NULL_INPUT, or @ref ZXC_ERROR_DST_TOO_SMALL.
 */
int zxc_cctx_init_in_workspace(zxc_cctx_t* RESTRICT ctx, void* RESTRICT workspace,
                               const size_t workspace_size, const size_t chunk_size, const int mode,
                               const int level, const int checksum_enabled,
                               const size_t dict_size) {
    if (UNLIKELY(!ctx || !workspace || chunk_size == 0)) return ZXC_ERROR_NULL_INPUT;

    const zxc_cctx_layout_t layout = compute_cctx_layout(chunk_size, mode, level, dict_size);
    if (UNLIKELY(workspace_size < layout.total)) return ZXC_ERROR_DST_TOO_SMALL;

    ZXC_MEMSET(ctx, 0, sizeof(zxc_cctx_t));
    ctx->checksum_enabled = checksum_enabled;
    ctx->chunk_size = chunk_size;
    const uint32_t offset_bits = zxc_log2_u32((uint32_t)chunk_size);
    ctx->offset_bits = offset_bits;
    ctx->offset_mask = (uint32_t)((1ULL << offset_bits) - 1);
    ctx->max_epoch = (uint32_t)(1ULL << (32 - offset_bits));

    /* memory_block stays NULL on the static-init path so zxc_cctx_free does
     * not try to free the caller's workspace.  Sub-buffer pointers carry the
     * partition; ownership is implicit (the caller owns @p workspace). */
    uint8_t* const mem = (uint8_t*)workspace;

    /* Dictionary concat scratch (both modes); init owns dict_size now so callers
     * no longer assign ctx->dict_size after init. */
    ctx->dict_size = dict_size;
    if (dict_size > 0) {
        ctx->dict_buffer = mem + layout.off_dict;
        ctx->dict_buffer_cap = layout.sz_dict;
    }

    if (mode == 0) {
        ctx->work_buf = mem + layout.off_work;
        ctx->work_buf_cap = chunk_size + ZXC_DECOMPRESS_TAIL_PAD;
        ctx->lit_buffer = mem + layout.off_lit_dctx;
        ctx->lit_buffer_cap = chunk_size + ZXC_PAD_SIZE;
        if (dict_size > 0) ctx->dict_huf_table = (zxc_huf_dec_entry_t*)(mem + layout.off_huf_dict);
        return ZXC_OK;
    }

    ctx->hash_table = (uint32_t*)(mem + layout.off_hash_pos);
    ctx->hash_tags = mem + layout.off_hash_tags;
    ctx->chain_table = (uint16_t*)(mem + layout.off_chain);
    ctx->buf_sequences = (uint32_t*)(mem + layout.off_seq_union);
    ctx->buf_offsets = (uint16_t*)(mem + layout.off_seq_union);
    ctx->buf_tokens = mem + layout.off_seq_union + layout.max_seq * sizeof(uint16_t);
    ctx->buf_extras = mem + layout.off_extras;
    ctx->literals = mem + layout.off_lit_cctx;
    if (layout.sz_opt) {
        ctx->opt_scratch = mem + layout.off_opt;
        ctx->opt_scratch_cap = layout.sz_opt;
    }

    ctx->compression_level = level;
    ctx->epoch = 1;

    ZXC_MEMSET(ctx->hash_table, 0, layout.sz_hash_pos);
    ZXC_MEMSET(ctx->hash_tags, 0, layout.sz_hash_tags);
    return ZXC_OK;
}

/**
 * @brief Initialises a compression / decompression context, allocating the
 *        persistent buffer with @c ZXC_ALIGNED_MALLOC.
 *
 * Thin wrapper around @ref zxc_cctx_init_in_workspace: sizes the buffer via
 * @ref zxc_cctx_compute_workspace_size, allocates it, then partitions it.
 * The pointer is stored in @c ctx->memory_block so @ref zxc_cctx_free can
 * release it.  The static-cctx public API (see @c zxc_buffer.h) bypasses
 * this wrapper and partitions a caller-supplied workspace directly.
 *
 * @param[out] ctx               Context to initialise.
 * @param[in]  chunk_size        Block size in bytes.
 * @param[in]  mode              1 = compression, 0 = decompression.
 * @param[in]  level             Compression level (ignored when @p mode == 0).
 * @param[in]  checksum_enabled  Non-zero to enable checksum computation.
 * @param[in]  dict_size         Dictionary prefill size.
 * @return @ref ZXC_OK on success, @ref ZXC_ERROR_MEMORY on allocation failure.
 */
int zxc_cctx_init(zxc_cctx_t* RESTRICT ctx, const size_t chunk_size, const int mode,
                  const int level, const int checksum_enabled, const size_t dict_size) {
    const size_t total = zxc_cctx_compute_workspace_size(chunk_size, mode, level, dict_size);
    if (UNLIKELY(total == 0)) return ZXC_ERROR_NULL_INPUT;

    uint8_t* const mem = (uint8_t*)ZXC_ALIGNED_MALLOC(total, ZXC_CACHE_LINE_SIZE);
    if (UNLIKELY(!mem)) return ZXC_ERROR_MEMORY;

    const int rc = zxc_cctx_init_in_workspace(ctx, mem, total, chunk_size, mode, level,
                                              checksum_enabled, dict_size);
    if (UNLIKELY(rc != ZXC_OK)) {
        // LCOV_EXCL_START
        ZXC_ALIGNED_FREE(mem);
        return rc;
        // LCOV_EXCL_STOP
    }
    /* Library-owned buffer: record the allocation so zxc_cctx_free frees it. */
    ctx->memory_block = mem;
    return ZXC_OK;
}

/**
 * @brief Releases all resources owned by a compression context.
 *
 * After this call every pointer inside @p ctx is @c NULL and the context
 * may be safely re-initialised with zxc_cctx_init().
 *
 * @param[in,out] ctx Context to tear down.
 */
void zxc_cctx_free(zxc_cctx_t* ctx) {
    if (ctx->memory_block) {
        ZXC_ALIGNED_FREE(ctx->memory_block);
        ctx->memory_block = NULL;
    }

    ctx->lit_buffer = NULL;
    ctx->hash_table = NULL;
    ctx->hash_tags = NULL;
    ctx->chain_table = NULL;
    ctx->buf_sequences = NULL;
    ctx->buf_tokens = NULL;
    ctx->buf_offsets = NULL;
    ctx->buf_extras = NULL;
    ctx->literals = NULL;
    ctx->work_buf = NULL;
    ctx->opt_scratch = NULL;
    ctx->dict_buffer = NULL;

    ctx->epoch = 0;
    ctx->lit_buffer_cap = 0;
    ctx->work_buf_cap = 0;
    ctx->opt_scratch_cap = 0;
    ctx->dict_buffer_cap = 0;
    ctx->dict_size = 0;
    ctx->dict_huf_lengths = NULL;
    ctx->dict_huf_table = NULL;
    ctx->lit_freq_acc = NULL;
}

/**
 * @brief Attach the shared dictionary literal Huffman table to a context.
 *
 * Stores @p lengths (128-byte packed code-lengths header, caller-owned, must
 * outlive the context) and, on decompression contexts created with
 * @c dict_size > 0, builds the decode table once into the workspace-carved
 * @c dict_huf_table. A NULL @p lengths is a no-op.
 *
 * @param[in,out] ctx      Initialised context to attach the table to.
 * @param[in]     lengths  128-byte packed code lengths, or NULL for a no-op.
 * @return @ref ZXC_OK on success, @ref ZXC_ERROR_CORRUPT_DATA if @p lengths is
 *         structurally invalid (bad nibble, Kraft inequality).
 */
int zxc_cctx_attach_dict_huf(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT lengths) {
    if (UNLIKELY(!ctx)) return ZXC_ERROR_NULL_INPUT;
    ctx->dict_huf_lengths = lengths;
    if (lengths == NULL || ctx->dict_huf_table == NULL) return ZXC_OK;

    /* Empty (all-zero) table from a low-entropy corpus: treat it as "no shared table". */
    int empty = 1;
    for (size_t i = 0; i < ZXC_HUF_TABLE_SIZE; i++) {
        if (lengths[i]) {
            empty = 0;
            break;
        }
    }
    if (empty) {
        ctx->dict_huf_lengths = NULL;
        ctx->dict_huf_table = NULL;
        return ZXC_OK;
    }

    uint8_t code_len[ZXC_HUF_NUM_SYMBOLS];
    int rc = zxc_huf_unpack_lengths(lengths, code_len);
    if (LIKELY(rc == ZXC_OK)) rc = zxc_huf_build_dec_table(code_len, ctx->dict_huf_table);
    if (UNLIKELY(rc != ZXC_OK)) ctx->dict_huf_table = NULL; /* invalid table: refuse decode */
    return rc;
}

/*
 * ============================================================================
 * HEADER I/O
 * ============================================================================
 */

/**
 * @brief Serialises a ZXC file header into @p dst.
 *
 * Layout (16 bytes): Magic (4) | Version (1) | Chunk (1) | Flags (1) |
 * Reserved (7) | CRC-16 (2).
 *
 * @param[out] dst          Destination buffer (>= @ref ZXC_FILE_HEADER_SIZE bytes).
 * @param[in]  dst_capacity Capacity of @p dst.
 * @param[in]  chunk_size   Block size (stored as its log2 exponent).
 * @param[in]  has_checksum Non-zero to set the checksum flag.
 * @param[in]  dict_id      Dictionary id; when non-zero, sets the dictionary flag
 *                          and is stored in the header.
 * @return Number of bytes written (@ref ZXC_FILE_HEADER_SIZE) on success,
 *         or a negative @ref zxc_error_t code.
 */
int zxc_write_file_header(uint8_t* RESTRICT dst, const size_t dst_capacity, const size_t chunk_size,
                          const int has_checksum, const uint32_t dict_id) {
    if (UNLIKELY(dst_capacity < ZXC_FILE_HEADER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_store_le32(dst, ZXC_MAGIC_WORD);
    dst[4] = ZXC_FILE_FORMAT_VERSION;

    // Block size stored as log2 exponent (e.g. 18 = 256 KB)
    dst[5] = (uint8_t)zxc_log2_u32((uint32_t)chunk_size);

    // Flags are at offset 6
    uint8_t flags = has_checksum ? (ZXC_FILE_FLAG_HAS_CHECKSUM | ZXC_CHECKSUM_RAPIDHASH) : 0;
    if (dict_id != 0) flags |= ZXC_FILE_FLAG_HAS_DICTIONARY;
    dst[6] = flags;

    // Bytes 7-13: Reserved / dict_id
    ZXC_MEMSET(dst + 7, 0, 7);
    if (dict_id != 0) zxc_store_le32(dst + 7, dict_id);

    // Bytes 14-15: CRC (16-bit)
    zxc_store_le16(dst + 14, 0);  // Zero out before hashing
    const uint16_t crc = zxc_hash16(dst);
    zxc_store_le16(dst + 14, crc);

    return ZXC_FILE_HEADER_SIZE;
}

/**
 * @brief Parses and validates a ZXC file header from @p src.
 *
 * Checks the magic word, format version, and CRC-16.
 *
 * @param[in]  src              Source buffer (>= @ref ZXC_FILE_HEADER_SIZE bytes).
 * @param[in]  src_size         Size of @p src.
 * @param[out] out_block_size   Receives the decoded block size (may be @c NULL).
 * @param[out] out_has_checksum Receives 1 if checksums are present, 0 otherwise
 *                              (may be @c NULL).
 * @param[out] out_dict_id      Receives the dictionary id, or 0 if none
 *                              (may be @c NULL).
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
int zxc_read_file_header(const uint8_t* RESTRICT src, const size_t src_size,
                         size_t* RESTRICT out_block_size, int* RESTRICT out_has_checksum,
                         uint32_t* RESTRICT out_dict_id) {
    if (UNLIKELY(src_size < ZXC_FILE_HEADER_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;
    if (UNLIKELY(zxc_le32(src) != ZXC_MAGIC_WORD)) return ZXC_ERROR_BAD_MAGIC;
    if (UNLIKELY(src[4] != ZXC_FILE_FORMAT_VERSION)) return ZXC_ERROR_BAD_VERSION;

    uint8_t temp[ZXC_FILE_HEADER_SIZE];
    ZXC_MEMCPY(temp, src, ZXC_FILE_HEADER_SIZE);
    // Zero out CRC bytes (14-15) before hash check
    temp[14] = 0;
    temp[15] = 0;
    // Header CRC16 (integrity), then the checksum-algorithm id in flags bits 0-3
    // (only 0 = RapidHash is defined). CRC is checked first via short-circuit.
    if (UNLIKELY(zxc_le16(src + 14) != zxc_hash16(temp) ||
                 (src[6] & 0x0FU) != ZXC_CHECKSUM_RAPIDHASH))
        return ZXC_ERROR_BAD_HEADER;

    if (out_block_size) {
        const uint8_t code = src[5];
        if (UNLIKELY(code < ZXC_BLOCK_SIZE_MIN_LOG2 || code > ZXC_BLOCK_SIZE_MAX_LOG2))
            return ZXC_ERROR_BAD_BLOCK_SIZE;
        // Exponent encoding: block_size = 2^code  (4 KB - 2 MB)
        *out_block_size = (size_t)1U << code;
    }
    // Flags are at offset 6
    if (out_has_checksum) *out_has_checksum = (src[6] & ZXC_FILE_FLAG_HAS_CHECKSUM) ? 1 : 0;
    if (out_dict_id) *out_dict_id = (src[6] & ZXC_FILE_FLAG_HAS_DICTIONARY) ? zxc_le32(src + 7) : 0;

    return ZXC_OK;
}

/**
 * @brief Serialises a block header (8 bytes) into @p dst.
 *
 * @param[out] dst          Destination buffer (>= @ref ZXC_BLOCK_HEADER_SIZE bytes).
 * @param[in]  dst_capacity Capacity of @p dst.
 * @param[in]  bh           Populated block header descriptor.
 * @return Number of bytes written (@ref ZXC_BLOCK_HEADER_SIZE) on success,
 *         or a negative @ref zxc_error_t code.
 */
int zxc_write_block_header(uint8_t* RESTRICT dst, const size_t dst_capacity,
                           const zxc_block_header_t* RESTRICT bh) {
    if (UNLIKELY(dst_capacity < ZXC_BLOCK_HEADER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    dst[0] = bh->block_type;
    dst[1] = 0;  // Flags not used currently
    dst[2] = 0;  // Reserved
    zxc_store_le32(dst + 3, bh->comp_size);
    dst[7] = 0;               // Zero before hashing
    dst[7] = zxc_hash8(dst);  // Checksum at the end

    return ZXC_BLOCK_HEADER_SIZE;
}

/**
 * @brief Parses and validates a block header from @p src.
 *
 * Validates the 8-bit CRC embedded in the header.
 *
 * @param[in]  src      Source buffer (>= @ref ZXC_BLOCK_HEADER_SIZE bytes).
 * @param[in]  src_size Size of @p src.
 * @param[out] bh       Receives the decoded block header fields.
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
int zxc_read_block_header(const uint8_t* RESTRICT src, const size_t src_size,
                          zxc_block_header_t* RESTRICT bh) {
    if (UNLIKELY(src_size < ZXC_BLOCK_HEADER_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;

    uint8_t temp[ZXC_BLOCK_HEADER_SIZE];
    ZXC_MEMCPY(temp, src, ZXC_BLOCK_HEADER_SIZE);
    temp[7] = 0;  // Zero out checksum byte before hashing
    if (UNLIKELY(src[7] != zxc_hash8(temp))) return ZXC_ERROR_BAD_HEADER;

    bh->block_type = src[0];
    bh->block_flags = 0;  // Flags not used currently
    bh->reserved = src[2];
    bh->comp_size = zxc_le32(src + 3);
    bh->header_crc = src[7];

    return ZXC_OK;
}

/**
 * @brief Writes the 12-byte file footer (source size + global checksum).
 *
 * @param[out] dst              Destination buffer (>= @ref ZXC_FILE_FOOTER_SIZE bytes).
 * @param[in]  dst_capacity     Capacity of @p dst.
 * @param[in]  src_size         Original uncompressed size in bytes.
 * @param[in]  global_hash      Accumulated global checksum value.
 * @param[in]  checksum_enabled Non-zero to write the checksum; zero to zero-fill.
 * @return Number of bytes written (@ref ZXC_FILE_FOOTER_SIZE) on success,
 *         or a negative @ref zxc_error_t code.
 */
int zxc_write_file_footer(uint8_t* RESTRICT dst, const size_t dst_capacity, const uint64_t src_size,
                          const uint32_t global_hash, const int checksum_enabled) {
    if (UNLIKELY(dst_capacity < ZXC_FILE_FOOTER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_store_le64(dst, src_size);

    if (checksum_enabled) {
        zxc_store_le32(dst + sizeof(uint64_t), global_hash);
    } else {
        ZXC_MEMSET(dst + sizeof(uint64_t), 0, sizeof(uint32_t));
    }

    return ZXC_FILE_FOOTER_SIZE;
}

/**
 * @brief Serialises a GLO block header followed by its section descriptors.
 *
 * @param[out] dst  Destination buffer.
 * @param[in]  rem  Remaining capacity of @p dst.
 * @param[in]  gh   Populated GLO header descriptor.
 * @param[in]  desc Array of @ref ZXC_GLO_SECTIONS section descriptors.
 * @return Total bytes written on success, or a negative @ref zxc_error_t code.
 */
int zxc_write_glo_header_and_desc(uint8_t* RESTRICT dst, const size_t rem,
                                  const zxc_gnr_header_t* RESTRICT gh,
                                  const zxc_section_desc_t desc[ZXC_GLO_SECTIONS]) {
    const size_t needed =
        ZXC_GLO_HEADER_BINARY_SIZE + ZXC_GLO_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(rem < needed)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_store_le32(dst, gh->n_sequences);
    zxc_store_le32(dst + 4, gh->n_literals);

    dst[8] = gh->enc_lit;
    dst[9] = gh->enc_litlen;
    dst[10] = gh->enc_mlen;
    dst[11] = gh->enc_off;

    zxc_store_le32(dst + 12, 0);
    uint8_t* p = dst + ZXC_GLO_HEADER_BINARY_SIZE;

    for (int i = 0; i < ZXC_GLO_SECTIONS; i++) {
        zxc_store_le64(p, desc[i].sizes);
        p += ZXC_SECTION_DESC_BINARY_SIZE;
    }

    return (int)needed;
}

/**
 * @brief Parses a GLO block header and its section descriptors from @p src.
 *
 * @param[in]  src  Source buffer.
 * @param[in]  len  Size of @p src.
 * @param[out] gh   Receives the decoded GLO header.
 * @param[out] desc Receives @ref ZXC_GLO_SECTIONS decoded section descriptors.
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
int zxc_read_glo_header_and_desc(const uint8_t* RESTRICT src, const size_t len,
                                 zxc_gnr_header_t* RESTRICT gh,
                                 zxc_section_desc_t desc[ZXC_GLO_SECTIONS]) {
    const size_t needed =
        ZXC_GLO_HEADER_BINARY_SIZE + ZXC_GLO_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(len < needed)) return ZXC_ERROR_SRC_TOO_SMALL;

    gh->n_sequences = zxc_le32(src);
    gh->n_literals = zxc_le32(src + 4);
    gh->enc_lit = src[8];
    gh->enc_litlen = src[9];
    gh->enc_mlen = src[10];
    gh->enc_off = src[11];

    const uint8_t* p = src + ZXC_GLO_HEADER_BINARY_SIZE;

    for (int i = 0; i < ZXC_GLO_SECTIONS; i++) {
        desc[i].sizes = zxc_le64(p);
        p += ZXC_SECTION_DESC_BINARY_SIZE;
    }
    return ZXC_OK;
}

/**
 * @brief Serialises a GHI block header followed by its section descriptors.
 *
 * @param[out] dst  Destination buffer.
 * @param[in]  rem  Remaining capacity of @p dst.
 * @param[in]  gh   Populated GHI header descriptor.
 * @param[in]  desc Array of @ref ZXC_GHI_SECTIONS section descriptors.
 * @return Total bytes written on success, or a negative @ref zxc_error_t code.
 */
int zxc_write_ghi_header_and_desc(uint8_t* RESTRICT dst, const size_t rem,
                                  const zxc_gnr_header_t* RESTRICT gh,
                                  const zxc_section_desc_t desc[ZXC_GHI_SECTIONS]) {
    const size_t needed =
        ZXC_GHI_HEADER_BINARY_SIZE + ZXC_GHI_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(rem < needed)) return ZXC_ERROR_DST_TOO_SMALL;

    zxc_store_le32(dst, gh->n_sequences);
    zxc_store_le32(dst + 4, gh->n_literals);

    dst[8] = gh->enc_lit;
    dst[9] = gh->enc_litlen;
    dst[10] = gh->enc_mlen;
    dst[11] = gh->enc_off;

    zxc_store_le32(dst + 12, 0);
    uint8_t* p = dst + ZXC_GHI_HEADER_BINARY_SIZE;

    for (int i = 0; i < ZXC_GHI_SECTIONS; i++) {
        zxc_store_le64(p, desc[i].sizes);
        p += ZXC_SECTION_DESC_BINARY_SIZE;
    }

    return (int)needed;
}

/**
 * @brief Parses a GHI block header and its section descriptors from @p src.
 *
 * @param[in]  src  Source buffer.
 * @param[in]  len  Size of @p src.
 * @param[out] gh   Receives the decoded GHI header.
 * @param[out] desc Receives @ref ZXC_GHI_SECTIONS decoded section descriptors.
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
int zxc_read_ghi_header_and_desc(const uint8_t* RESTRICT src, const size_t len,
                                 zxc_gnr_header_t* RESTRICT gh,
                                 zxc_section_desc_t desc[ZXC_GHI_SECTIONS]) {
    const size_t needed =
        ZXC_GHI_HEADER_BINARY_SIZE + ZXC_GHI_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    if (UNLIKELY(len < needed)) return ZXC_ERROR_SRC_TOO_SMALL;

    gh->n_sequences = zxc_le32(src);
    gh->n_literals = zxc_le32(src + 4);
    gh->enc_lit = src[8];
    gh->enc_litlen = src[9];
    gh->enc_mlen = src[10];
    gh->enc_off = src[11];

    const uint8_t* p = src + ZXC_GHI_HEADER_BINARY_SIZE;

    for (int i = 0; i < ZXC_GHI_SECTIONS; i++) {
        desc[i].sizes = zxc_le64(p);
        p += ZXC_SECTION_DESC_BINARY_SIZE;
    }
    return ZXC_OK;
}

/*
 * ============================================================================
 * COMPRESS BOUND CALCULATION
 * ============================================================================
 */
/**
 * @brief Returns the maximum compressed size for a given input size.
 *
 * The result accounts for the file header, per-block headers, block
 * checksums, worst-case expansion, EOF block, seekable overhead (SEK
 * block), and the file footer.
 *
 * The block count is derived from @ref ZXC_BLOCK_SIZE_MIN (4 KB) to
 * guarantee the bound holds for all valid block sizes and seekable mode.
 *
 * @param[in] input_size Uncompressed input size in bytes.
 * @return Upper bound on compressed size, or 0 if @p input_size would overflow.
 */
uint64_t zxc_compress_bound(const size_t input_size) {
    // Guard against uint64 overflow when summing per-block overhead
    // across very large inputs (input_size approaching SIZE_MAX).
    if (UNLIKELY(input_size > (SIZE_MAX - (SIZE_MAX >> 8)))) return 0;
    uint64_t n = ((uint64_t)input_size + ZXC_BLOCK_SIZE_MIN - 1) / ZXC_BLOCK_SIZE_MIN;
    if (n == 0) n = 1;
    return ZXC_FILE_HEADER_SIZE +
           (n * (ZXC_BLOCK_HEADER_SIZE + ZXC_BLOCK_CHECKSUM_SIZE + ZXC_BLOCK_FORMAT_OVERHEAD)) +
           (uint64_t)input_size + ZXC_BLOCK_HEADER_SIZE + /* EOF block */
           ZXC_BLOCK_HEADER_SIZE +                        /* SEK block header (seekable) */
           (n * ZXC_SEEK_ENTRY_SIZE) +                    /* SEK entries: 4 bytes per block */
           ZXC_FILE_FOOTER_SIZE;
}

/**
 * @brief Returns the maximum compressed size for a single block (no file framing).
 *
 * @param[in] input_size Uncompressed block size in bytes
 *                       (must be <= @ref ZXC_BLOCK_SIZE_MAX).
 * @return Upper bound on compressed block size, or 0 if @p input_size is out
 *         of range for the Block API (i.e. exceeds ZXC_BLOCK_SIZE_MAX) or if
 *         the arithmetic would overflow.
 */
uint64_t zxc_compress_block_bound(const size_t input_size) {
    // Mirror the Block API contract: src_size must be in [1, ZXC_BLOCK_SIZE_MAX].
    // Inputs outside this range cause zxc_compress_block to fail
    // (NULL_INPUT for 0, BAD_BLOCK_SIZE above MAX), so the bound is undefined.
    // Returning 0 signals "unusable" upfront. The cap also makes the addition
    // below trivially overflow-free.
    if (UNLIKELY(input_size == 0 || input_size > ZXC_BLOCK_SIZE_MAX)) return 0;
    // Outer block header + payload (worst case: incompressible, raw bytes)
    // + inner format overhead + optional checksum.
    return (uint64_t)ZXC_BLOCK_HEADER_SIZE + (uint64_t)input_size + ZXC_BLOCK_FORMAT_OVERHEAD +
           ZXC_BLOCK_CHECKSUM_SIZE;
}

/**
 * @brief Returns the minimum dst_capacity required by zxc_decompress_block().
 *
 * The decoder uses speculative wild-copy writes on its fast path.
 * Sizing the destination to uncompressed_size + ZXC_PAD_SIZE*66 guarantees
 * the fast path is always reachable and that tail bounds checks never
 * spuriously reject the last literals of a valid block.
 *
 * Returns 0 if @p uncompressed_size exceeds ZXC_BLOCK_SIZE_MAX (the Block API
 * limit), or if the arithmetic would overflow.
 *
 * @param[in] uncompressed_size  Exact decompressed size of the block.
 * @return Minimum @c dst_capacity in bytes, or 0 if @p uncompressed_size exceeds
 *         @c ZXC_BLOCK_SIZE_MAX.
 */
uint64_t zxc_decompress_block_bound(const size_t uncompressed_size) {
    if (UNLIKELY(uncompressed_size > ZXC_BLOCK_SIZE_MAX)) return 0;
    return (uint64_t)uncompressed_size + ZXC_DECOMPRESS_TAIL_PAD;
}

/**
 * @brief Estimates the total buffer bytes allocated inside a cctx for a block.
 *
 * Thin wrapper around @ref zxc_cctx_compute_workspace_size for @c mode == 1
 * (compress), with @c src_size clamped up to a valid block size via
 * @ref zxc_block_size_ceil.  The opaque wrapper struct allocated by
 * @ref zxc_create_cctx adds a fixed overhead (< 128 B) that is negligible
 * next to the per-chunk buffers and is intentionally omitted.
 *
 * For @p level >= 6 the figure includes the optimal-parser scratch
 * (@c opt_scratch, ~8.125 bytes per chunk_size byte) used by the optimal
 * parser and reused as transient package-merge scratch for the Huffman
 * code-length builder.
 *
 * @param[in] src_size  Input size; rounded up to a valid block size.
 * @param[in] level     Compression level (>= 6 includes the optimal-parser scratch).
 * @return Estimated context buffer size in bytes, or 0 if @p src_size is 0.
 */
uint64_t zxc_estimate_cctx_size(const size_t src_size, const int level) {
    if (UNLIKELY(src_size == 0)) return 0;
    const size_t chunk_size = zxc_block_size_ceil(src_size);
    return (uint64_t)zxc_cctx_compute_workspace_size(chunk_size, 1, level, 0);
}

/*
 * ============================================================================
 * ERROR CODE UTILITIES
 * ============================================================================
 */

/**
 * @brief Returns a human-readable string for the given error code.
 *
 * @param[in] code An error code from @ref zxc_error_t (or @ref ZXC_OK).
 * @return A static string such as @c "ZXC_OK" or @c "ZXC_ERROR_MEMORY".
 *         Returns @c "ZXC_UNKNOWN_ERROR" for unrecognised codes.
 */
const char* zxc_error_name(const int code) {
    switch ((zxc_error_t)code) {
        case ZXC_OK:
            return "ZXC_OK";
        case ZXC_ERROR_MEMORY:
            return "ZXC_ERROR_MEMORY";
        case ZXC_ERROR_DST_TOO_SMALL:
            return "ZXC_ERROR_DST_TOO_SMALL";
        case ZXC_ERROR_SRC_TOO_SMALL:
            return "ZXC_ERROR_SRC_TOO_SMALL";
        case ZXC_ERROR_BAD_MAGIC:
            return "ZXC_ERROR_BAD_MAGIC";
        case ZXC_ERROR_BAD_VERSION:
            return "ZXC_ERROR_BAD_VERSION";
        case ZXC_ERROR_BAD_HEADER:
            return "ZXC_ERROR_BAD_HEADER";
        case ZXC_ERROR_BAD_CHECKSUM:
            return "ZXC_ERROR_BAD_CHECKSUM";
        case ZXC_ERROR_CORRUPT_DATA:
            return "ZXC_ERROR_CORRUPT_DATA";
        case ZXC_ERROR_BAD_OFFSET:
            return "ZXC_ERROR_BAD_OFFSET";
        case ZXC_ERROR_OVERFLOW:
            return "ZXC_ERROR_OVERFLOW";
        case ZXC_ERROR_IO:
            return "ZXC_ERROR_IO";
        case ZXC_ERROR_NULL_INPUT:
            return "ZXC_ERROR_NULL_INPUT";
        case ZXC_ERROR_BAD_BLOCK_TYPE:
            return "ZXC_ERROR_BAD_BLOCK_TYPE";
        case ZXC_ERROR_BAD_BLOCK_SIZE:
            return "ZXC_ERROR_BAD_BLOCK_SIZE";
        case ZXC_ERROR_DICT_REQUIRED:
            return "ZXC_ERROR_DICT_REQUIRED";
        case ZXC_ERROR_DICT_MISMATCH:
            return "ZXC_ERROR_DICT_MISMATCH";
        case ZXC_ERROR_DICT_TOO_LARGE:
            return "ZXC_ERROR_DICT_TOO_LARGE";
        default:
            return "ZXC_UNKNOWN_ERROR";
    }
}

/*
 * ============================================================================
 * LIBRARY INFORMATION
 * ============================================================================
 */

/**
 * @brief Returns the minimum supported compression level.
 *
 * Returns the value of ZXC_LEVEL_FASTEST (currently 1).
 * This allows integrators to discover the level range at runtime without relying on
 * compile-time macros alone.
 */
int zxc_min_level(void) { return ZXC_LEVEL_FASTEST; }

/**
 * @brief Returns the maximum supported compression level.
 *
 * Returns the value of ZXC_LEVEL_DENSITY (currently 6).
 */
int zxc_max_level(void) { return ZXC_LEVEL_DENSITY; }

/**
 * @brief Returns the default compression level.
 *
 * Returns the value of ZXC_LEVEL_DEFAULT (currently 3).
 */
int zxc_default_level(void) { return ZXC_LEVEL_DEFAULT; }

/**
 * @brief Returns the human-readable library version string.
 *
 * The returned pointer is a compile-time constant and must not be freed.
 * Format: "MAJOR.MINOR.PATCH" (e.g. "0.12.0").
 */
const char* zxc_version_string(void) { return ZXC_LIB_VERSION_STR; }
