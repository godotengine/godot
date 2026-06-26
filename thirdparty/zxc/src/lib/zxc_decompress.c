/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_decompress.c
 * @brief Block-level decompression: GLO / GHI / RAW decoding with
 *        SIMD-accelerated bit-unpacking and overlapping copies.
 *
 * Like @ref zxc_compress.c, this file is compiled multiple times with
 * @c ZXC_FUNCTION_SUFFIX to produce per-ISA variants.
 */

/*
 * Function Multi-Versioning Support
 * If ZXC_FUNCTION_SUFFIX is defined (e.g. _avx2, _neon), rename the public
 * entry point AND the Huffman decoder consumed by this TU. The defines sit
 * before zxc_internal.h so that the prototypes the header declares are also
 * rewritten with the suffix, keeping callers and callees consistent.
 */
#ifdef ZXC_FUNCTION_SUFFIX
#define ZXC_CAT_IMPL(x, y) x##y
#define ZXC_CAT(x, y) ZXC_CAT_IMPL(x, y)
#define zxc_decompress_chunk_wrapper ZXC_CAT(zxc_decompress_chunk_wrapper, ZXC_FUNCTION_SUFFIX)
#define zxc_decompress_chunk_wrapper_dict \
    ZXC_CAT(zxc_decompress_chunk_wrapper_dict, ZXC_FUNCTION_SUFFIX)
#define zxc_decompress_chunk_wrapper_safe \
    ZXC_CAT(zxc_decompress_chunk_wrapper_safe, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_decode_section ZXC_CAT(zxc_huf_decode_section, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_decode_section_dict ZXC_CAT(zxc_huf_decode_section_dict, ZXC_FUNCTION_SUFFIX)
#endif

#include "../../include/zxc_error.h"
#include "zxc_internal.h"

/**
 * @brief Reads a Prefix Varint encoded integer from a stream.
 *
 * This function decodes a 32-bit unsigned integer encoded in Prefix Varint format
 * from the provided byte stream. Unary prefix bits in the first byte determine
 * the total length (1-3 bytes).
 *
 * Format:
 * - 1 byte  (0xxxxxxx):  7-bit payload (val < 2^7  = 128)
 * - 2 bytes (10xxxxxx): 14-bit payload (val < 2^14 = 16384)
 * - 3 bytes (110xxxxx): 21-bit payload (val < 2^21 = 2097152)
 *
 * @param[in,out] ptr Pointer to a pointer to the current position in the stream.
 * @param[in] end Pointer to the end of the readable stream (for bounds checking).
 * @return The decoded 32-bit integer, or 0 if reading would overflow bounds (safe default).
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_read_varint(const uint8_t** ptr, const uint8_t* end) {
    const uint8_t* p = *ptr;
    // Bounds check: need at least 1 byte
    if (UNLIKELY(p >= end)) return 0;

    const uint32_t b0 = p[0];

    // 1 Byte: 0xxxxxxx (7 bits) -> val < 128 (2^7)
    if (LIKELY(b0 < 0x80)) {
        *ptr = p + 1;
        return b0;
    }

    // 2 Bytes: 10xxxxxx xxxxxxxx (14 bits) -> val < 16384 (2^14)
    if (LIKELY(b0 < 0xC0)) {
        if (UNLIKELY(p + 1 >= end)) {
            *ptr = end;
            return 0;
        }
        *ptr = p + 2;
        return (b0 & 0x3F) | ((uint32_t)p[1] << 6);
    }

    // 3 Bytes: 110xxxxx xxxxxxxx xxxxxxxx (21 bits) -> val < 2097152 (2^21).
    // This is the largest length a legitimate varint can take: block_size_max
    // is 2^21 and varint values represent (ll - MASK) or (ml - MASK), which is
    // always strictly less than block_size_max.
    if (LIKELY(b0 < 0xE0)) {
        if (UNLIKELY(p + 2 >= end)) {
            *ptr = end;
            return 0;
        }
        *ptr = p + 3;
        return (b0 & 0x1F) | ((uint32_t)p[1] << 5) | ((uint32_t)p[2] << 13);
    }

    // extra encoding: out-of-spec for the current format, reject.
    *ptr = end;
    return 0;
}

/**
 * @brief Shuffle masks for overlapping copies with small offsets (0-15).
 *
 * Shared between ARM NEON and x86 SSSE3. Each row defines how to replicate
 * source bytes to fill 16 bytes when offset < 16.
 */
#if defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32) || defined(ZXC_USE_AVX2) || \
    defined(ZXC_USE_AVX512)
/**
 * @brief Precomputed masks for handling overlapping data during decompression.
 *
 * This 16x16 lookup table contains 128-bit aligned masks used to efficiently
 * mask off or combine bytes when processing overlapping copy operations or
 * boundary conditions in the ZXC decompression algorithm.
 *
 * The alignment to 16 bytes ensures compatibility with SIMD instructions
 * (like SSE/AVX) for optimized memory operations.
 */
static const ZXC_ALIGN(16) uint8_t zxc_overlap_masks[16][16] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},      // off=0 (unused)
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},      // off=1 (RLE handled separately)
    {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},      // off=2
    {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0},      // off=3
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},      // off=4
    {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0},      // off=5
    {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3},      // off=6
    {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1},      // off=7
    {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},      // off=8
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6},      // off=9
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5},      // off=10
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4},     // off=11
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3},    // off=12
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2},   // off=13
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1},  // off=14
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0}  // off=15
};
#endif

/**
 * @brief Per-offset store stride for periodic overlap runs: the largest
 *        multiple of @c off that fits in 16 bytes, i.e. `16 - (16 % off)`.
 *
 * Advancing the output cursor by a multiple of @c off keeps the 16-byte
 * pattern vector phase-aligned, so the run is emitted with pure stores of a
 * single register. Entries 0 and 1 are unused (RLE handled separately).
 */
static const uint8_t zxc_overlap_strides[16] = {16, 16, 16, 15, 16, 15, 12, 14,
                                                16, 9,  10, 11, 12, 13, 14, 15};

/**
 * @brief Copies an @p ml-byte LZ run whose pattern repeats with period @p off (2..15).
 *
 * Builds the 16-byte periodic pattern `out[i] = dst[-off + (i % off)]` once
 * (one shuffle on NEON/SSSE3, a wrap-counter byte loop on the SSE2/scalar
 * tier), then emits 16-byte stores advancing by @ref zxc_overlap_strides so
 * the pattern never needs re-shuffling. May overshoot up to 15 bytes past
 * @p ml; the caller must guarantee @ref ZXC_PAD_SIZE bytes of headroom.
 *
 * @param[out] dst Output cursor; the run source is `dst - off`.
 * @param[in]  off Back-reference distance, in [2, 15].
 * @param[in]  ml  Run length in bytes (>= 1).
 */
// codeql[cpp/unused-static-function] : False positive
static ZXC_ALWAYS_INLINE void zxc_decode_copy_overlap_run(uint8_t* dst, const uint32_t off,
                                                          const uint64_t ml) {
    const size_t stride = zxc_overlap_strides[off];
    size_t copied = 0;
#if defined(ZXC_USE_NEON64)
    const uint8x16_t mask = vld1q_u8(zxc_overlap_masks[off]);
    const uint8x16_t pat = vqtbl1q_u8(vld1q_u8(dst - off), mask);
    do {
        vst1q_u8(dst + copied, pat);
        copied += stride;
    } while (copied < ml);

#elif defined(ZXC_USE_NEON32)
    uint8x8x2_t src_tbl;
    src_tbl.val[0] = vld1_u8(dst - off);
    src_tbl.val[1] = vld1_u8(dst - off + 8);
    const uint8x8_t pat_lo = vtbl2_u8(src_tbl, vld1_u8(zxc_overlap_masks[off]));
    const uint8x8_t pat_hi = vtbl2_u8(src_tbl, vld1_u8(zxc_overlap_masks[off] + 8));
    do {
        vst1_u8(dst + copied, pat_lo);
        vst1_u8(dst + copied + 8, pat_hi);
        copied += stride;
    } while (copied < ml);

#elif defined(ZXC_USE_AVX2) || defined(ZXC_USE_AVX512)
    const __m128i mask = _mm_load_si128((const __m128i*)zxc_overlap_masks[off]);
    const __m128i src_data = _mm_loadu_si128((const __m128i*)(dst - off));
    const __m128i pat = _mm_shuffle_epi8(src_data, mask);
    do {
        _mm_storeu_si128((__m128i*)(dst + copied), pat);
        copied += stride;
    } while (copied < ml);

#else
    // SSE2-only tier and non-SIMD builds: no PSHUFB, build the pattern with a
    // wrap counter (no per-byte modulo), then store it via zxc_copy16.
    const uint8_t* src = dst - off;
    uint8_t pat[16];
    uint32_t k = 0;
    for (size_t i = 0; i < 16; i++) {
        pat[i] = src[k];
        if (++k == off) k = 0;
    }
    do {
        zxc_copy16(dst + copied, pat);
        copied += stride;
    } while (copied < ml);
#endif
}

/**
 * @brief Fills an @p ml-byte single-byte run (LZ offset == 1) with wild stores.
 *
 * Splats @p byte into a vector register and emits @ref ZXC_PAD_SIZE-byte
 * chunks, avoiding a libc memset call on the typically short runs of the hot
 * path. Like the other run copiers it may **overshoot** up to
 * @ref ZXC_PAD_SIZE - 1 bytes past @p ml; the caller must guarantee
 * @ref ZXC_PAD_SIZE bytes of headroom. Falls back to @ref ZXC_MEMSET on
 * non-SIMD builds.
 *
 * @param[out] dst  Output cursor.
 * @param[in]  byte Byte value to replicate.
 * @param[in]  ml   Run length in bytes (>= 1).
 */
// codeql[cpp/unused-static-function] : False positive, used in DECODE_SEQ_SAFE/FAST macros
static ZXC_ALWAYS_INLINE void zxc_decode_fill_run(uint8_t* dst, const uint8_t byte,
                                                  const uint64_t ml) {
#if defined(ZXC_USE_AVX2) || defined(ZXC_USE_AVX512)
    const __m256i v = _mm256_set1_epi8((char)byte);
    _mm256_storeu_si256((__m256i*)dst, v);
    if (UNLIKELY(ml > ZXC_PAD_SIZE)) {
        uint8_t* out = dst + ZXC_PAD_SIZE;
        size_t rem = ml - ZXC_PAD_SIZE;
        while (rem > ZXC_PAD_SIZE) {
            _mm256_storeu_si256((__m256i*)out, v);
            out += ZXC_PAD_SIZE;
            rem -= ZXC_PAD_SIZE;
        }
        _mm256_storeu_si256((__m256i*)out, v);
    }
#elif defined(ZXC_USE_SSE2)
    const __m128i v = _mm_set1_epi8((char)byte);
    _mm_storeu_si128((__m128i*)dst, v);
    _mm_storeu_si128((__m128i*)(dst + 16), v);
    if (UNLIKELY(ml > ZXC_PAD_SIZE)) {
        uint8_t* out = dst + ZXC_PAD_SIZE;
        size_t rem = ml - ZXC_PAD_SIZE;
        while (rem > ZXC_PAD_SIZE) {
            _mm_storeu_si128((__m128i*)out, v);
            _mm_storeu_si128((__m128i*)(out + 16), v);
            out += ZXC_PAD_SIZE;
            rem -= ZXC_PAD_SIZE;
        }
        _mm_storeu_si128((__m128i*)out, v);
        _mm_storeu_si128((__m128i*)(out + 16), v);
    }
#elif defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32)
    const uint8x16_t v = vdupq_n_u8(byte);
    vst1q_u8(dst, v);
    vst1q_u8(dst + 16, v);
    if (UNLIKELY(ml > ZXC_PAD_SIZE)) {
        uint8_t* out = dst + ZXC_PAD_SIZE;
        size_t rem = ml - ZXC_PAD_SIZE;
        while (rem > ZXC_PAD_SIZE) {
            vst1q_u8(out, v);
            vst1q_u8(out + 16, v);
            out += ZXC_PAD_SIZE;
            rem -= ZXC_PAD_SIZE;
        }
        vst1q_u8(out, v);
        vst1q_u8(out + 16, v);
    }
#else
    ZXC_MEMSET(dst, byte, ml);
#endif
}

/* ==========================================================================
 * Shared decode macros for the GLO and GHI decoders (fast + safe variants).
 * Defined at file scope to avoid four identical copies inside each function.
 * They reference the local names l_ptr, d_ptr, written that every call site
 * has in scope. #undef-ed at the end of the last consumer.
 * ========================================================================== */

/**
 * @brief Copies @p ll literal bytes from @p src to @p dst using 32-byte wild copies.
 *
 * Writes in @ref ZXC_PAD_SIZE-byte chunks and may **overshoot** by up to
 * @ref ZXC_PAD_SIZE - 1 bytes past @p ll; the caller must guarantee @p dst has at
 * least @ref ZXC_PAD_SIZE bytes of writable headroom (the unrolled loops and the
 * trailing-literal margins ensure this). Pointers are taken by value and the
 * caller advances its own cursors by @p ll, keeping them in registers on the hot
 * path.
 *
 * @param[out] dst Output cursor. Must not overlap @p src and must have
 *                 @ref ZXC_PAD_SIZE bytes of overshoot headroom.
 * @param[in]  src Literal-stream source. Must not overlap @p dst (RESTRICT).
 * @param[in]  ll  Number of literal bytes to copy.
 */
static ZXC_ALWAYS_INLINE void zxc_decode_copy_literals(uint8_t* RESTRICT dst,
                                                       const uint8_t* RESTRICT src,
                                                       const uint64_t ll) {
    zxc_copy32(dst, src);
    if (UNLIKELY(ll > ZXC_PAD_SIZE)) {
        dst += ZXC_PAD_SIZE;
        src += ZXC_PAD_SIZE;
        size_t rem = ll - ZXC_PAD_SIZE;
        while (rem > ZXC_PAD_SIZE) {
            zxc_copy32(dst, src);
            dst += ZXC_PAD_SIZE;
            src += ZXC_PAD_SIZE;
            rem -= ZXC_PAD_SIZE;
        }
        zxc_copy32(dst, src);
    }
}

/**
 * @brief Copies an @p ml-byte LZ match from @c d_ptr-off to @p d_ptr, handling overlap.
 *
 * The source @c d_ptr-off may overlap the destination (the LZ repeat case), so the
 * copy strategy is chosen by back-reference distance:
 *  - @p off >= @ref ZXC_PAD_SIZE      : 32-byte wild copies (no overlap within a chunk);
 *  - @p off >= @ref ZXC_PAD_SIZE / 2  : 16-byte wild copies;
 *  - @p off == 1                      : single-byte run via @ref zxc_decode_fill_run;
 *  - otherwise (2..15)                : pattern-replicating overlap copy.
 *
 * Like @ref zxc_decode_copy_literals it may **overshoot** up to @ref ZXC_PAD_SIZE - 1
 * bytes past @p ml, so @p d_ptr must have @ref ZXC_PAD_SIZE bytes of headroom. @p d_ptr
 * is taken by value; the caller advances its cursor by @p ml.
 *
 * @param[in,out] d_ptr Output cursor; the match source is @c d_ptr-off. Must have
 *                      @ref ZXC_PAD_SIZE bytes of overshoot headroom.
 * @param[in]     off   Resolved (bias-removed) back-reference distance, @c >= 1.
 * @param[in]     ml    Match length in bytes (@c >= ZXC_LZ_MIN_MATCH_LEN).
 */
static ZXC_ALWAYS_INLINE void zxc_decode_copy_match(uint8_t* RESTRICT d_ptr, const uint32_t off,
                                                    const uint64_t ml) {
    const uint8_t* match_src = d_ptr - off;
    if (LIKELY(off >= ZXC_PAD_SIZE)) {
        zxc_copy32(d_ptr, match_src);
        if (UNLIKELY(ml > ZXC_PAD_SIZE)) {
            uint8_t* out = d_ptr + ZXC_PAD_SIZE;
            const uint8_t* ref = match_src + ZXC_PAD_SIZE;
            size_t rem = ml - ZXC_PAD_SIZE;
            while (rem > ZXC_PAD_SIZE) {
                zxc_copy32(out, ref);
                out += ZXC_PAD_SIZE;
                ref += ZXC_PAD_SIZE;
                rem -= ZXC_PAD_SIZE;
            }
            zxc_copy32(out, ref);
        }
    } else if (off >= (ZXC_PAD_SIZE / 2)) {
        zxc_copy16(d_ptr, match_src);
        if (UNLIKELY(ml > (ZXC_PAD_SIZE / 2))) {
            uint8_t* out = d_ptr + (ZXC_PAD_SIZE / 2);
            const uint8_t* ref = match_src + (ZXC_PAD_SIZE / 2);
            size_t rem = ml - (ZXC_PAD_SIZE / 2);
            while (rem > (ZXC_PAD_SIZE / 2)) {
                zxc_copy16(out, ref);
                out += (ZXC_PAD_SIZE / 2);
                ref += (ZXC_PAD_SIZE / 2);
                rem -= (ZXC_PAD_SIZE / 2);
            }
            zxc_copy16(out, ref);
        }
    } else if (off == 1) {
        zxc_decode_fill_run(d_ptr, match_src[0], ml);
    } else {
        zxc_decode_copy_overlap_run(d_ptr, off, ml);
    }
}

// SAFE version: validates offset against written bytes
#define DECODE_SEQ_SAFE(ll, ml, off)                              \
    do {                                                          \
        zxc_decode_copy_literals(d_ptr, l_ptr, ll);               \
        l_ptr += ll;                                              \
        d_ptr += ll;                                              \
        written += ll;                                            \
        if (UNLIKELY(off > written)) return ZXC_ERROR_BAD_OFFSET; \
        zxc_decode_copy_match(d_ptr, off, ml);                    \
        d_ptr += ml;                                              \
        written += ml;                                            \
    } while (0)

// FAST version: no offset validation (for use after written >= 256 or 65536)
#define DECODE_SEQ_FAST(ll, ml, off)                \
    do {                                            \
        zxc_decode_copy_literals(d_ptr, l_ptr, ll); \
        l_ptr += ll;                                \
        d_ptr += ll;                                \
        zxc_decode_copy_match(d_ptr, off, ml);      \
        d_ptr += ml;                                \
    } while (0)

/**
 * @brief Unified GLO (General Low) block decoder body, shared by the fast, safe
 *        and dictionary variants.
 *
 * Decodes a block in the internal GLO format; the decompressed size is derived
 * from the Section Descriptors in the payload. @p safe and @p has_dict must be
 * compile-time constants (0 or 1): the 4x-unrolled loops are duplicated inside
 * @c if(safe)/else branches so each variant keeps single-assignment @c const
 * save pointers, and after constant propagation only one branch survives per
 * wrapper (codegen equivalent to a hand-written pair).
 *
 * @param[in,out] ctx          Decompression context (dict buffer, tables).
 * @param[in]     src          Compressed block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer for decoded bytes.
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @param[in]     safe         Compile-time flag: 1 = strict bounds-checked loop.
 * @param[in]     has_dict     Compile-time flag: 1 = resolve matches against a dict prefix.
 * @return Bytes written to @p dst on success, or a negative @ref zxc_error_t.
 */
static ZXC_ALWAYS_INLINE int zxc_decode_block_glo_impl(const zxc_cctx_t* RESTRICT ctx,
                                                       const uint8_t* RESTRICT src,
                                                       const size_t src_size, uint8_t* RESTRICT dst,
                                                       const size_t dst_capacity, const int safe,
                                                       const int has_dict) {
    zxc_gnr_header_t gh;

    /* Constant 0 when !has_dict, so `written` starts at 0 and `dst - dict_size`
     * folds to `dst` -- pre-dict codegen on the hot path. */
    const size_t dict_size = has_dict ? ctx->dict_size : 0;
    zxc_section_desc_t desc[ZXC_GLO_SECTIONS];

    if (UNLIKELY(zxc_read_glo_header_and_desc(src, src_size, &gh, desc) != ZXC_OK))
        return ZXC_ERROR_BAD_HEADER;

    const uint8_t* p_data =
        src + ZXC_GLO_HEADER_BINARY_SIZE + ZXC_GLO_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;
    const uint8_t* p_curr = p_data;

    // --- Literal Stream Setup ---
    const uint8_t* l_ptr;
    const uint8_t* l_end;
    uint8_t* rle_buf = NULL;

    size_t lit_stream_size = (size_t)(desc[0].sizes & ZXC_SECTION_SIZE_MASK);

    if (gh.enc_lit == ZXC_SECTION_ENCODING_HUFFMAN) {
        const size_t required_size = (size_t)(desc[0].sizes >> 32);
        if (UNLIKELY(lit_stream_size > (size_t)(src + src_size - p_curr)))
            return ZXC_ERROR_CORRUPT_DATA;
        if (required_size == 0) {
            l_ptr = p_curr;
            l_end = p_curr;
        } else {
            if (UNLIKELY(required_size > dst_capacity || required_size > SIZE_MAX - ZXC_PAD_SIZE))
                return ZXC_ERROR_DST_TOO_SMALL;
            const size_t alloc_size = required_size + ZXC_PAD_SIZE;
            /* lit_buffer is pre-allocated to chunk_size + ZXC_PAD_SIZE by
             * zxc_cctx_init (mode == 0). */
            if (UNLIKELY(ctx->lit_buffer_cap < alloc_size)) return ZXC_ERROR_CORRUPT_DATA;
            const int rc =
                zxc_huf_decode_section(p_curr, lit_stream_size, ctx->lit_buffer, required_size);
            if (UNLIKELY(rc != ZXC_OK)) return rc;
            l_ptr = ctx->lit_buffer;
            l_end = ctx->lit_buffer + required_size;
        }
    } else if (gh.enc_lit == ZXC_SECTION_ENCODING_HUFFMAN_DICT) {
        /* Shared dictionary table: no inline lengths header; the prebuilt
         * decode table was attached to the context with the dictionary. */
        const size_t required_size = (size_t)(desc[0].sizes >> 32);
        if (UNLIKELY(lit_stream_size > (size_t)(src + src_size - p_curr)))
            return ZXC_ERROR_CORRUPT_DATA;
        if (required_size == 0) {
            l_ptr = p_curr;
            l_end = p_curr;
        } else {
            if (UNLIKELY(ctx->dict_huf_table == NULL)) return ZXC_ERROR_DICT_REQUIRED;
            if (UNLIKELY(required_size > dst_capacity || required_size > SIZE_MAX - ZXC_PAD_SIZE))
                return ZXC_ERROR_DST_TOO_SMALL;
            const size_t alloc_size = required_size + ZXC_PAD_SIZE;
            /* lit_buffer is pre-allocated to chunk_size + ZXC_PAD_SIZE by
             * zxc_cctx_init (mode == 0). */
            if (UNLIKELY(ctx->lit_buffer_cap < alloc_size)) return ZXC_ERROR_CORRUPT_DATA;
            const int rc = zxc_huf_decode_section_dict(p_curr, lit_stream_size, ctx->lit_buffer,
                                                       required_size, ctx->dict_huf_table);
            if (UNLIKELY(rc != ZXC_OK)) return rc;
            l_ptr = ctx->lit_buffer;
            l_end = ctx->lit_buffer + required_size;
        }
    } else if (gh.enc_lit == ZXC_SECTION_ENCODING_RLE) {
        const size_t required_size = (size_t)(desc[0].sizes >> 32);

        if (required_size > 0) {
            if (UNLIKELY(required_size > dst_capacity || required_size > SIZE_MAX - ZXC_PAD_SIZE))
                return ZXC_ERROR_DST_TOO_SMALL;
            const size_t alloc_size = required_size + ZXC_PAD_SIZE;

            /* lit_buffer is pre-allocated to chunk_size + ZXC_PAD_SIZE by
             * zxc_cctx_init (mode == 0).*/
            if (UNLIKELY(ctx->lit_buffer_cap < alloc_size)) return ZXC_ERROR_CORRUPT_DATA;

            rle_buf = ctx->lit_buffer;
            if (UNLIKELY(!rle_buf || lit_stream_size > (size_t)(src + src_size - p_curr)))
                return ZXC_ERROR_CORRUPT_DATA;

            const uint8_t* r_ptr = p_curr;
            const uint8_t* r_end = r_ptr + lit_stream_size;
            uint8_t* w_ptr = rle_buf;
            const uint8_t* const w_end = rle_buf + required_size;

            while (r_ptr < r_end && w_ptr < w_end) {
                uint8_t token = *r_ptr++;
                if (LIKELY(!(token & ZXC_LIT_RLE_FLAG))) {
                    // Raw copy (most common path): use ZXC_PAD_SIZE-byte wild copies
                    // token is 7-bit (0-127), so len is 1-128 bytes
                    const uint32_t len = (uint32_t)token + 1;
                    if (UNLIKELY(w_ptr + len > w_end || r_ptr + len > r_end))
                        return ZXC_ERROR_CORRUPT_DATA;

                    // Destination has ZXC_PAD_SIZE bytes of safe overrun space.
                    // Source may not - check before wild copy.
                    // Fast path: source has ZXC_PAD_SIZE-byte read headroom (most common)
                    if (LIKELY(r_ptr + ZXC_PAD_SIZE <= r_end)) {
                        // Single 32-byte copy covers len <= ZXC_PAD_SIZE (most tokens)
                        zxc_copy32(w_ptr, r_ptr);

                        if (UNLIKELY(len > ZXC_PAD_SIZE)) {
                            // Unroll: max len=128, so max 4 copies total
                            // Use unconditional stores with overlap - faster than branches
                            if (len <= 2 * ZXC_PAD_SIZE) {
                                zxc_copy32(w_ptr + len - ZXC_PAD_SIZE, r_ptr + len - ZXC_PAD_SIZE);
                            } else if (len <= 3 * ZXC_PAD_SIZE) {
                                zxc_copy32(w_ptr + ZXC_PAD_SIZE, r_ptr + ZXC_PAD_SIZE);
                                zxc_copy32(w_ptr + len - ZXC_PAD_SIZE, r_ptr + len - ZXC_PAD_SIZE);
                            } else {
                                zxc_copy32(w_ptr + ZXC_PAD_SIZE, r_ptr + ZXC_PAD_SIZE);
                                zxc_copy32(w_ptr + 2 * ZXC_PAD_SIZE, r_ptr + 2 * ZXC_PAD_SIZE);
                                zxc_copy32(w_ptr + len - ZXC_PAD_SIZE, r_ptr + len - ZXC_PAD_SIZE);
                            }
                        }
                    } else {
                        // Near end of source: safe copy (rare cold path)
                        ZXC_MEMCPY(w_ptr, r_ptr, len);
                    }

                    w_ptr += len;
                    r_ptr += len;
                } else {
                    // RLE run: fill with single byte
                    const uint32_t len = (token & ZXC_LIT_LEN_MASK) + 4;
                    if (UNLIKELY(w_ptr + len > w_end || r_ptr >= r_end))
                        return ZXC_ERROR_CORRUPT_DATA;
                    ZXC_MEMSET(w_ptr, *r_ptr++, len);
                    w_ptr += len;
                }
            }
            if (UNLIKELY(w_ptr != w_end)) return ZXC_ERROR_CORRUPT_DATA;
            l_ptr = rle_buf;
            l_end = rle_buf + required_size;
        } else {
            l_ptr = p_curr;
            l_end = p_curr;
        }
    } else if (gh.enc_lit == ZXC_SECTION_ENCODING_RAW) {
        l_ptr = p_curr;
        l_end = p_curr + lit_stream_size;
    } else {
        return ZXC_ERROR_CORRUPT_DATA;
    }

    p_curr += lit_stream_size;

    // --- Stream Pointers & Validation ---
    const size_t sz_tokens = (size_t)(desc[1].sizes & ZXC_SECTION_SIZE_MASK);
    const size_t sz_offsets = (size_t)(desc[2].sizes & ZXC_SECTION_SIZE_MASK);
    const size_t sz_extras = (size_t)(desc[3].sizes & ZXC_SECTION_SIZE_MASK);

    // Validate stream sizes match sequence count (early rejection of malformed data)
    const uint64_t expected_off_size =
        (gh.enc_off == 1) ? (uint64_t)gh.n_sequences : (uint64_t)gh.n_sequences * 2;

    const uint8_t* t_ptr = p_curr;
    const uint8_t* o_ptr = t_ptr + sz_tokens;
    const uint8_t* e_ptr = o_ptr + sz_offsets;
    const uint8_t* const e_end = e_ptr + sz_extras;  // For vbyte overflow detection

    // Validate streams don't overflow source buffer +
    // Validate stream sizes match sequence count (early rejection of malformed data)
    if (UNLIKELY((e_end != src + src_size) || sz_tokens < gh.n_sequences ||
                 (uint64_t)sz_offsets < expected_off_size))
        return ZXC_ERROR_CORRUPT_DATA;

    uint8_t* d_ptr = dst;
    const uint8_t* const d_end = dst + dst_capacity;
    // Destination safe margin for 4x loop: max output without varint extension.
    // ll_max = 14, ml_max = 14 + 5 = 19, per-seq = 33, 4x = 132.
    // Add ZXC_PAD_SIZE (32) for the wild zxc_copy32 overshoot + 4 safety = 168.
    const uint8_t* const d_end_safe = d_end - (132 + ZXC_PAD_SIZE + 4);

    // Literal stream safe threshold for 4x-unrolled loops.
    // Without varint extension, max ll per sequence = ZXC_TOKEN_LL_MASK - 1 = 14.
    // For 4 sequences: 4 * 14 = 56. With this margin, l_ptr checks are only needed
    // on the cold varint path, keeping the hot path free of l_ptr overhead.
    const size_t glo_sz_lit = (size_t)(l_end - l_ptr);
    const size_t glo_margin_4x = 4 * (ZXC_TOKEN_LL_MASK - 1);  // 56
    const size_t glo_margin_1x = ZXC_TOKEN_LL_MASK - 1;        // 14
    const uint8_t* const l_end_safe_4x =
        (glo_sz_lit > glo_margin_4x) ? l_end - glo_margin_4x : l_ptr;
    const uint8_t* const l_end_safe_1x =
        (glo_sz_lit > glo_margin_1x) ? l_end - glo_margin_1x : l_ptr;

    uint32_t n_seq = gh.n_sequences;

    // Track bytes written for offset validation
    // For 1-byte offsets (enc_off==1): validate until 256 bytes written (max 8-bit offset)
    // For 2-byte offsets (enc_off==0): validate until 65536 bytes written (max 16-bit offset)
    // After threshold, all offsets are guaranteed valid (can't exceed written bytes)
    // When a dictionary is active, dict_size bytes are logically "already written"
    // (prepended by the caller), so the SAFE loop may be skipped entirely.
    size_t written = dict_size;

    // --- SAFE Loop: offset validation until threshold (4x unroll) ---
    // For 1-byte offsets: bounds check until 256 bytes written
    // For 2-byte offsets: bounds check until 65536 bytes written
    const size_t bounds_threshold = (gh.enc_off == 1) ? (1U << 8) : (1U << 16);

    if (safe) {
        /* SAFE variant: save per-batch state so overflow can rollback. */
        while (n_seq >= 4 && d_ptr < d_end_safe && l_ptr < l_end_safe_4x &&
               written < bounds_threshold) {
            const uint8_t* const t_save = t_ptr;
            const uint8_t* const o_save = o_ptr;
            const uint8_t* const e_save = e_ptr;
            uint8_t* const d_save = d_ptr;
            const uint8_t* const l_save = l_ptr;
            const size_t w_save = written;
            uint32_t tokens = zxc_le32(t_ptr);
            t_ptr += sizeof(uint32_t);

            uint32_t off1 = ZXC_LZ_OFFSET_BIAS, off2 = ZXC_LZ_OFFSET_BIAS,
                     off3 = ZXC_LZ_OFFSET_BIAS, off4 = ZXC_LZ_OFFSET_BIAS;
            if (gh.enc_off == 1) {
                uint32_t offsets = zxc_le32(o_ptr);
                o_ptr += sizeof(uint32_t);
                off1 += offsets & 0xFF;
                off2 += (offsets >> 8) & 0xFF;
                off3 += (offsets >> 16) & 0xFF;
                off4 += (offsets >> 24) & 0xFF;
            } else {
                uint64_t offsets = zxc_le64(o_ptr);
                o_ptr += sizeof(uint64_t);
                off1 += (uint32_t)(offsets & 0xFFFF);
                off2 += (uint32_t)((offsets >> 16) & 0xFFFF);
                off3 += (uint32_t)((offsets >> 32) & 0xFFFF);
                off4 += (uint32_t)((offsets >> 48) & 0xFFFF);
            }

            uint64_t ll1 = (tokens & 0x0F0) >> 4;
            uint64_t ml1 = (tokens & 0x00F);
            if (UNLIKELY(ll1 == ZXC_TOKEN_LL_MASK)) {
                ll1 += zxc_read_varint(&e_ptr, e_end);
                const uint64_t reserve =
                    ((tokens >> 12) & 0xF) + ((tokens >> 20) & 0xF) + (tokens >> 28);
                if (UNLIKELY(ll1 + reserve > (size_t)(l_end - l_ptr) ||
                             ll1 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            if (UNLIKELY(ml1 == ZXC_TOKEN_ML_MASK)) {
                ml1 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll1 + ml1 + ZXC_LZ_MIN_MATCH_LEN +
                                 3U * ZXC_GLO_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            ml1 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_SAFE(ll1, ml1, off1);

            uint64_t ll2 = (tokens & 0x0F000) >> 12;
            uint64_t ml2 = (tokens & 0x00F00) >> 8;
            if (UNLIKELY(ll2 == ZXC_TOKEN_LL_MASK)) {
                ll2 += zxc_read_varint(&e_ptr, e_end);
                const uint64_t reserve = ((tokens >> 20) & 0xF) + (tokens >> 28);
                if (UNLIKELY(ll2 + reserve > (size_t)(l_end - l_ptr) ||
                             ll2 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            if (UNLIKELY(ml2 == ZXC_TOKEN_ML_MASK)) {
                ml2 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll2 + ml2 + ZXC_LZ_MIN_MATCH_LEN +
                                 2U * ZXC_GLO_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            ml2 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_SAFE(ll2, ml2, off2);

            uint64_t ll3 = (tokens & 0x0F00000) >> 20;
            uint64_t ml3 = (tokens & 0x00F0000) >> 16;
            if (UNLIKELY(ll3 == ZXC_TOKEN_LL_MASK)) {
                ll3 += zxc_read_varint(&e_ptr, e_end);
                const uint64_t reserve = (tokens >> 28);
                if (UNLIKELY(ll3 + reserve > (size_t)(l_end - l_ptr) ||
                             ll3 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            if (UNLIKELY(ml3 == ZXC_TOKEN_ML_MASK)) {
                ml3 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll3 + ml3 + ZXC_LZ_MIN_MATCH_LEN + ZXC_GLO_MAX_INLINE_OUT_PER_SEQ +
                                 ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            ml3 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_SAFE(ll3, ml3, off3);

            uint64_t ll4 = (tokens >> 28);
            uint64_t ml4 = (tokens >> 24) & 0x0F;
            if (UNLIKELY(ll4 == ZXC_TOKEN_LL_MASK)) {
                ll4 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll4 > (size_t)(l_end - l_ptr) ||
                             ll4 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            if (UNLIKELY(ml4 == ZXC_TOKEN_ML_MASK)) {
                ml4 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll4 + ml4 + ZXC_LZ_MIN_MATCH_LEN + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            ml4 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_SAFE(ll4, ml4, off4);

            n_seq -= 4;
            continue;

        rollback_safe_4x:
            t_ptr = t_save;
            o_ptr = o_save;
            e_ptr = e_save;
            d_ptr = d_save;
            l_ptr = l_save;
            written = w_save;
            break;
        }
    } else {
        while (n_seq >= 4 && d_ptr < d_end_safe && l_ptr < l_end_safe_4x &&
               written < bounds_threshold) {
            uint32_t tokens = zxc_le32(t_ptr);
            t_ptr += sizeof(uint32_t);

            uint32_t off1 = ZXC_LZ_OFFSET_BIAS;
            uint32_t off2 = ZXC_LZ_OFFSET_BIAS;
            uint32_t off3 = ZXC_LZ_OFFSET_BIAS;
            uint32_t off4 = ZXC_LZ_OFFSET_BIAS;

            if (gh.enc_off == 1) {
                // Read 4 x 1-byte offsets
                uint32_t offsets = zxc_le32(o_ptr);
                o_ptr += sizeof(uint32_t);
                off1 += offsets & 0xFF;
                off2 += (offsets >> 8) & 0xFF;
                off3 += (offsets >> 16) & 0xFF;
                off4 += (offsets >> 24) & 0xFF;
            } else {
                // Read 4 x 2-byte offsets
                uint64_t offsets = zxc_le64(o_ptr);
                o_ptr += sizeof(uint64_t);
                off1 += (uint32_t)(offsets & 0xFFFF);
                off2 += (uint32_t)((offsets >> 16) & 0xFFFF);
                off3 += (uint32_t)((offsets >> 32) & 0xFFFF);
                off4 += (uint32_t)((offsets >> 48) & 0xFFFF);
            }

            uint64_t ll1 = (tokens & 0x0F0) >> 4;
            uint64_t ml1 = (tokens & 0x00F);
            if (UNLIKELY(ll1 == ZXC_TOKEN_LL_MASK)) {
                ll1 += zxc_read_varint(&e_ptr, e_end);
                const uint64_t reserve =
                    ((tokens >> 12) & 0xF) + ((tokens >> 20) & 0xF) + (tokens >> 28);
                if (UNLIKELY(ll1 + reserve > (size_t)(l_end - l_ptr) ||
                             ll1 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            if (UNLIKELY(ml1 == ZXC_TOKEN_ML_MASK)) {
                ml1 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll1 + ml1 + ZXC_LZ_MIN_MATCH_LEN +
                                 3U * ZXC_GLO_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            ml1 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_SAFE(ll1, ml1, off1);

            uint64_t ll2 = (tokens & 0x0F000) >> 12;
            uint64_t ml2 = (tokens & 0x00F00) >> 8;
            if (UNLIKELY(ll2 == ZXC_TOKEN_LL_MASK)) {
                ll2 += zxc_read_varint(&e_ptr, e_end);
                const uint64_t reserve = ((tokens >> 20) & 0xF) + (tokens >> 28);
                if (UNLIKELY(ll2 + reserve > (size_t)(l_end - l_ptr) ||
                             ll2 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            if (UNLIKELY(ml2 == ZXC_TOKEN_ML_MASK)) {
                ml2 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll2 + ml2 + ZXC_LZ_MIN_MATCH_LEN +
                                 2U * ZXC_GLO_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            ml2 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_SAFE(ll2, ml2, off2);

            uint64_t ll3 = (tokens & 0x0F00000) >> 20;
            uint64_t ml3 = (tokens & 0x00F0000) >> 16;
            if (UNLIKELY(ll3 == ZXC_TOKEN_LL_MASK)) {
                ll3 += zxc_read_varint(&e_ptr, e_end);
                const uint64_t reserve = (tokens >> 28);
                if (UNLIKELY(ll3 + reserve > (size_t)(l_end - l_ptr) ||
                             ll3 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            if (UNLIKELY(ml3 == ZXC_TOKEN_ML_MASK)) {
                ml3 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll3 + ml3 + ZXC_LZ_MIN_MATCH_LEN + ZXC_GLO_MAX_INLINE_OUT_PER_SEQ +
                                 ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            ml3 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_SAFE(ll3, ml3, off3);

            uint64_t ll4 = (tokens >> 28);
            uint64_t ml4 = (tokens >> 24) & 0x0F;
            if (UNLIKELY(ll4 == ZXC_TOKEN_LL_MASK)) {
                ll4 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll4 > (size_t)(l_end - l_ptr) ||
                             ll4 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            if (UNLIKELY(ml4 == ZXC_TOKEN_ML_MASK)) {
                ml4 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll4 + ml4 + ZXC_LZ_MIN_MATCH_LEN + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            ml4 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_SAFE(ll4, ml4, off4);

            n_seq -= 4;
        }
    }

    // --- FAST Loop: After threshold, no offset validation needed (4x unroll) ---
    if (safe) {
        while (n_seq >= 4 && d_ptr < d_end_safe && l_ptr < l_end_safe_4x) {
            const uint8_t* const t_save = t_ptr;
            const uint8_t* const o_save = o_ptr;
            const uint8_t* const e_save = e_ptr;
            uint8_t* const d_save = d_ptr;
            const uint8_t* const l_save = l_ptr;
            uint32_t tokens = zxc_le32(t_ptr);
            t_ptr += sizeof(uint32_t);

            uint32_t off1 = ZXC_LZ_OFFSET_BIAS;
            uint32_t off2 = ZXC_LZ_OFFSET_BIAS;
            uint32_t off3 = ZXC_LZ_OFFSET_BIAS;
            uint32_t off4 = ZXC_LZ_OFFSET_BIAS;

            if (gh.enc_off == 1) {
                uint32_t offsets = zxc_le32(o_ptr);
                o_ptr += sizeof(uint32_t);
                off1 += offsets & 0xFF;
                off2 += (offsets >> 8) & 0xFF;
                off3 += (offsets >> 16) & 0xFF;
                off4 += (offsets >> 24) & 0xFF;
            } else {
                uint64_t offsets = zxc_le64(o_ptr);
                o_ptr += sizeof(uint64_t);
                off1 += (uint32_t)(offsets & 0xFFFF);
                off2 += (uint32_t)((offsets >> 16) & 0xFFFF);
                off3 += (uint32_t)((offsets >> 32) & 0xFFFF);
                off4 += (uint32_t)((offsets >> 48) & 0xFFFF);
            }

            uint64_t ll1 = (tokens & 0x0F0) >> 4;
            uint64_t ml1 = (tokens & 0x00F);
            if (UNLIKELY(ll1 == ZXC_TOKEN_LL_MASK)) {
                ll1 += zxc_read_varint(&e_ptr, e_end);
                const uint64_t reserve =
                    ((tokens >> 12) & 0xF) + ((tokens >> 20) & 0xF) + (tokens >> 28);
                if (UNLIKELY(ll1 + reserve > (size_t)(l_end - l_ptr) ||
                             ll1 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            if (UNLIKELY(ml1 == ZXC_TOKEN_ML_MASK)) {
                ml1 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll1 + ml1 + ZXC_LZ_MIN_MATCH_LEN +
                                 3U * ZXC_GLO_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            ml1 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_FAST(ll1, ml1, off1);

            uint64_t ll2 = (tokens & 0x0F000) >> 12;
            uint64_t ml2 = (tokens & 0x00F00) >> 8;
            if (UNLIKELY(ll2 == ZXC_TOKEN_LL_MASK)) {
                ll2 += zxc_read_varint(&e_ptr, e_end);
                const uint64_t reserve = ((tokens >> 20) & 0xF) + (tokens >> 28);
                if (UNLIKELY(ll2 + reserve > (size_t)(l_end - l_ptr) ||
                             ll2 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            if (UNLIKELY(ml2 == ZXC_TOKEN_ML_MASK)) {
                ml2 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll2 + ml2 + ZXC_LZ_MIN_MATCH_LEN +
                                 2U * ZXC_GLO_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            ml2 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_FAST(ll2, ml2, off2);

            uint64_t ll3 = (tokens & 0x0F00000) >> 20;
            uint64_t ml3 = (tokens & 0x00F0000) >> 16;
            if (UNLIKELY(ll3 == ZXC_TOKEN_LL_MASK)) {
                ll3 += zxc_read_varint(&e_ptr, e_end);
                const uint64_t reserve = (tokens >> 28);
                if (UNLIKELY(ll3 + reserve > (size_t)(l_end - l_ptr) ||
                             ll3 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            if (UNLIKELY(ml3 == ZXC_TOKEN_ML_MASK)) {
                ml3 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll3 + ml3 + ZXC_LZ_MIN_MATCH_LEN + ZXC_GLO_MAX_INLINE_OUT_PER_SEQ +
                                 ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            ml3 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_FAST(ll3, ml3, off3);

            uint64_t ll4 = (tokens >> 28);
            uint64_t ml4 = (tokens >> 24) & 0x0F;
            if (UNLIKELY(ll4 == ZXC_TOKEN_LL_MASK)) {
                ll4 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll4 > (size_t)(l_end - l_ptr) ||
                             ll4 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            if (UNLIKELY(ml4 == ZXC_TOKEN_ML_MASK)) {
                ml4 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll4 + ml4 + ZXC_LZ_MIN_MATCH_LEN + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            ml4 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_FAST(ll4, ml4, off4);

            n_seq -= 4;
            continue;

        rollback_fast_4x:
            t_ptr = t_save;
            o_ptr = o_save;
            e_ptr = e_save;
            d_ptr = d_save;
            l_ptr = l_save;
            break;
        }
    } else {
        while (n_seq >= 4 && d_ptr < d_end_safe && l_ptr < l_end_safe_4x) {
            uint32_t tokens = zxc_le32(t_ptr);
            t_ptr += sizeof(uint32_t);

            uint32_t off1 = ZXC_LZ_OFFSET_BIAS;
            uint32_t off2 = ZXC_LZ_OFFSET_BIAS;
            uint32_t off3 = ZXC_LZ_OFFSET_BIAS;
            uint32_t off4 = ZXC_LZ_OFFSET_BIAS;
            if (gh.enc_off == 1) {
                // Read 4 x 1-byte offsets
                uint32_t offsets = zxc_le32(o_ptr);
                o_ptr += sizeof(uint32_t);
                off1 += offsets & 0xFF;
                off2 += (offsets >> 8) & 0xFF;
                off3 += (offsets >> 16) & 0xFF;
                off4 += (offsets >> 24) & 0xFF;
            } else {
                // Read 4 x 2-byte offsets
                uint64_t offsets = zxc_le64(o_ptr);
                o_ptr += sizeof(uint64_t);
                off1 += (uint32_t)(offsets & 0xFFFF);
                off2 += (uint32_t)((offsets >> 16) & 0xFFFF);
                off3 += (uint32_t)((offsets >> 32) & 0xFFFF);
                off4 += (uint32_t)((offsets >> 48) & 0xFFFF);
            }

            uint64_t ll1 = (tokens & 0x0F0) >> 4;
            uint64_t ml1 = (tokens & 0x00F);
            if (UNLIKELY(ll1 == ZXC_TOKEN_LL_MASK)) {
                ll1 += zxc_read_varint(&e_ptr, e_end);
                const uint64_t reserve =
                    ((tokens >> 12) & 0xF) + ((tokens >> 20) & 0xF) + (tokens >> 28);
                if (UNLIKELY(ll1 + reserve > (size_t)(l_end - l_ptr) ||
                             ll1 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            if (UNLIKELY(ml1 == ZXC_TOKEN_ML_MASK)) {
                ml1 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll1 + ml1 + ZXC_LZ_MIN_MATCH_LEN +
                                 3U * ZXC_GLO_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            ml1 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_FAST(ll1, ml1, off1);

            uint64_t ll2 = (tokens & 0x0F000) >> 12;
            uint64_t ml2 = (tokens & 0x00F00) >> 8;
            if (UNLIKELY(ll2 == ZXC_TOKEN_LL_MASK)) {
                ll2 += zxc_read_varint(&e_ptr, e_end);
                const uint64_t reserve = ((tokens >> 20) & 0xF) + (tokens >> 28);
                if (UNLIKELY(ll2 + reserve > (size_t)(l_end - l_ptr) ||
                             ll2 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            if (UNLIKELY(ml2 == ZXC_TOKEN_ML_MASK)) {
                ml2 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll2 + ml2 + ZXC_LZ_MIN_MATCH_LEN +
                                 2U * ZXC_GLO_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            ml2 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_FAST(ll2, ml2, off2);

            uint64_t ll3 = (tokens & 0x0F00000) >> 20;
            uint64_t ml3 = (tokens & 0x00F0000) >> 16;
            if (UNLIKELY(ll3 == ZXC_TOKEN_LL_MASK)) {
                ll3 += zxc_read_varint(&e_ptr, e_end);
                const uint64_t reserve = (tokens >> 28);
                if (UNLIKELY(ll3 + reserve > (size_t)(l_end - l_ptr) ||
                             ll3 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            if (UNLIKELY(ml3 == ZXC_TOKEN_ML_MASK)) {
                ml3 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll3 + ml3 + ZXC_LZ_MIN_MATCH_LEN + ZXC_GLO_MAX_INLINE_OUT_PER_SEQ +
                                 ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            ml3 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_FAST(ll3, ml3, off3);

            uint64_t ll4 = (tokens >> 28);
            uint64_t ml4 = (tokens >> 24) & 0x0F;
            if (UNLIKELY(ll4 == ZXC_TOKEN_LL_MASK)) {
                ll4 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll4 > (size_t)(l_end - l_ptr) ||
                             ll4 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            if (UNLIKELY(ml4 == ZXC_TOKEN_ML_MASK)) {
                ml4 += zxc_read_varint(&e_ptr, e_end);
                if (UNLIKELY(ll4 + ml4 + ZXC_LZ_MIN_MATCH_LEN + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            ml4 += ZXC_LZ_MIN_MATCH_LEN;
            DECODE_SEQ_FAST(ll4, ml4, off4);

            n_seq -= 4;
        }
    }

    // Validate vbyte reads didn't overflow
    if (UNLIKELY(e_ptr > e_end)) return ZXC_ERROR_CORRUPT_DATA;

    // --- Remaining 1 sequence (Fast Path) ---
    while (n_seq > 0 && d_ptr < d_end_safe && l_ptr < l_end_safe_1x) {
        // Save pointers before reading (in case we need to fall back to Safe Path)
        const uint8_t* t_save = t_ptr;
        const uint8_t* o_save = o_ptr;
        const uint8_t* e_save = e_ptr;

        uint8_t token = *t_ptr++;
        uint64_t ll = token >> ZXC_TOKEN_LIT_BITS;
        uint64_t ml = token & ZXC_TOKEN_ML_MASK;
        uint32_t offset = ZXC_LZ_OFFSET_BIAS;
        if (gh.enc_off == 1) {
            offset += *o_ptr++;  // 1-byte offset (biased)
        } else {
            offset += zxc_le16(o_ptr);  // 2-byte offset (biased)
            o_ptr += sizeof(uint16_t);
        }

        if (UNLIKELY(ll == ZXC_TOKEN_LL_MASK)) {
            ll += zxc_read_varint(&e_ptr, e_end);
            if (UNLIKELY(l_ptr + ll > l_end)) {
                t_ptr = t_save;
                o_ptr = o_save;
                e_ptr = e_save;
                break;
            }
        }
        if (UNLIKELY(ml == ZXC_TOKEN_ML_MASK)) ml += zxc_read_varint(&e_ptr, e_end);
        ml += ZXC_LZ_MIN_MATCH_LEN;

        // Check bounds before wild copies - if too close to end, fall back to Safe Path
        if (UNLIKELY(ll + ml + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr))) {
            // Restore pointers and let Safe Path handle this sequence
            t_ptr = t_save;
            o_ptr = o_save;
            e_ptr = e_save;
            break;
        }

        {
            const uint8_t* src_lit = l_ptr;
            uint8_t* dst_lit = d_ptr;
            zxc_copy32(dst_lit, src_lit);
            if (UNLIKELY(ll > ZXC_PAD_SIZE)) {
                dst_lit += ZXC_PAD_SIZE;
                src_lit += ZXC_PAD_SIZE;
                size_t rem = ll - ZXC_PAD_SIZE;
                while (rem > ZXC_PAD_SIZE) {
                    zxc_copy32(dst_lit, src_lit);
                    dst_lit += ZXC_PAD_SIZE;
                    src_lit += ZXC_PAD_SIZE;
                    rem -= ZXC_PAD_SIZE;
                }
                zxc_copy32(dst_lit, src_lit);
            }
            l_ptr += ll;
            d_ptr += ll;
            written += ll;
        }

        {
            // Skip check if written >= bounds_threshold (256 for 8-bit, 65536 for 16-bit)
            if (UNLIKELY(written < bounds_threshold && offset > written))
                return ZXC_ERROR_BAD_OFFSET;

            /* The loop entry check guarantees ll + ml + ZXC_PAD_SIZE bytes of
             * headroom, so the wild-copy ladder (incl. overlap/fill runs) is safe. */
            zxc_decode_copy_match(d_ptr, offset, ml);
            d_ptr += ml;
            written += ml;
        }
        n_seq--;
    }

    // --- Safe Path for Remaining Sequences ---
    while (n_seq > 0) {
        uint8_t token = *t_ptr++;
        uint64_t ll = token >> ZXC_TOKEN_LIT_BITS;
        uint64_t ml = token & ZXC_TOKEN_ML_MASK;
        uint32_t offset = ZXC_LZ_OFFSET_BIAS;
        if (gh.enc_off == 1) {
            offset += *o_ptr++;  // 1-byte offset (biased)
        } else {
            offset += zxc_le16(o_ptr);  // 2-byte offset (biased)
            o_ptr += sizeof(uint16_t);
        }

        if (UNLIKELY(ll == ZXC_TOKEN_LL_MASK)) ll += zxc_read_varint(&e_ptr, e_end);
        if (UNLIKELY(ml == ZXC_TOKEN_ML_MASK)) ml += zxc_read_varint(&e_ptr, e_end);
        ml += ZXC_LZ_MIN_MATCH_LEN;

        if (UNLIKELY(ll + ml > (size_t)(d_end - d_ptr) || l_ptr + ll > l_end))
            return ZXC_ERROR_OVERFLOW;
        ZXC_MEMCPY(d_ptr, l_ptr, ll);
        l_ptr += ll;
        d_ptr += ll;

        const uint8_t* match_src = d_ptr - offset;
        if (UNLIKELY(match_src < dst - dict_size)) return ZXC_ERROR_BAD_OFFSET;

        if (offset < ml) {
            for (size_t i = 0; i < ml; i++) d_ptr[i] = match_src[i];
        } else {
            ZXC_MEMCPY(d_ptr, match_src, ml);
        }
        d_ptr += ml;
        n_seq--;
    }

    // --- Trailing Literals ---
    // Copy remaining literals from source stream (literal exhaustion)
    if (UNLIKELY(l_ptr > l_end)) return ZXC_ERROR_CORRUPT_DATA;
    const size_t remaining_literals = (size_t)(l_end - l_ptr);
    if (remaining_literals > 0) {
        if (UNLIKELY(d_ptr + remaining_literals > d_end)) return ZXC_ERROR_OVERFLOW;
        ZXC_MEMCPY(d_ptr, l_ptr, remaining_literals);
        d_ptr += remaining_literals;
    }

    return (int)(d_ptr - dst);
}

/**
 * @brief Unified GHI (General High) block decoder body, shared by the fast, safe
 *        and dictionary variants.
 *
 * Decodes a block in the internal GHI format; the decompressed size is derived
 * from the Section Descriptors in the payload. @p safe and @p has_dict must be
 * compile-time constants (0 or 1): the 4x-unrolled loops are duplicated inside
 * @c if(safe)/else branches so each variant keeps single-assignment @c const
 * save pointers, and after constant propagation only one branch survives per
 * wrapper.
 *
 * @param[in,out] ctx          Decompression context (dict buffer, tables).
 * @param[in]     src          Compressed block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer for decoded bytes.
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @param[in]     safe         Compile-time flag: 1 = strict bounds-checked loop.
 * @param[in]     has_dict     Compile-time flag: 1 = resolve matches against a dict prefix.
 * @return Bytes written to @p dst on success, or a negative @ref zxc_error_t.
 */
static ZXC_ALWAYS_INLINE int zxc_decode_block_ghi_impl(const zxc_cctx_t* RESTRICT ctx,
                                                       const uint8_t* RESTRICT src,
                                                       const size_t src_size, uint8_t* RESTRICT dst,
                                                       const size_t dst_capacity, const int safe,
                                                       const int has_dict) {
    zxc_gnr_header_t gh;

    /* 0 when !has_dict (safe path) -> folds `written`/`dst - dict_size`. */
    const size_t dict_size = has_dict ? ctx->dict_size : 0;
    zxc_section_desc_t desc[ZXC_GHI_SECTIONS];

    if (UNLIKELY(zxc_read_ghi_header_and_desc(src, src_size, &gh, desc) != ZXC_OK))
        return ZXC_ERROR_BAD_HEADER;

    const uint8_t* p_curr =
        src + ZXC_GHI_HEADER_BINARY_SIZE + ZXC_GHI_SECTIONS * ZXC_SECTION_DESC_BINARY_SIZE;

    // --- Stream Pointers & Validation ---
    const size_t sz_lit = (uint32_t)desc[0].sizes;
    const size_t sz_seqs = (uint32_t)desc[1].sizes;
    const size_t sz_exts = (uint32_t)desc[2].sizes;
    const uint8_t* l_ptr = p_curr;
    const uint8_t* l_end = l_ptr + sz_lit;
    p_curr += sz_lit;

    const uint8_t* seq_ptr = p_curr;
    const uint8_t* extras_ptr = p_curr + sz_seqs;
    const uint8_t* const extras_end = extras_ptr + sz_exts;

    // Validate streams don't overflow source buffer +
    // Validate sequence stream size matches sequence count
    if (UNLIKELY((extras_end != src + src_size) ||
                 ((uint64_t)sz_seqs < (uint64_t)gh.n_sequences * 4)))
        return ZXC_ERROR_CORRUPT_DATA;

    uint8_t* d_ptr = dst;
    const uint8_t* const d_end = dst + dst_capacity;
    const uint8_t* const d_end_safe = d_end - (ZXC_PAD_SIZE * 4);  // 128
    // Safety margin for 4x unrolled loop: 4 * (ZXC_SEQ_LL_MASK LL +
    // ZXC_SEQ_ML_MASK+ZXC_LZ_MIN_MATCH_LEN ML) + ZXC_PAD_SIZE Pad = 4 x (255 + 255 + 5) + 32 = 2092
    const uint8_t* const d_end_fast = d_end - ZXC_DECOMPRESS_TAIL_PAD;  // 2112

    // Literal stream safe thresholds for GHI loops.
    // Without varint extension, max ll per sequence = ZXC_SEQ_LL_MASK - 1 = 254.
    // For 4 sequences: 4 * 254 = 1016. With this margin, l_ptr checks are only needed
    // on the cold varint path, keeping the hot path free of l_ptr overhead.
    const size_t ghi_margin_4x = 4 * (ZXC_SEQ_LL_MASK - 1);  // 1016
    const size_t ghi_margin_1x = ZXC_SEQ_LL_MASK - 1;        // 254
    const uint8_t* const l_end_safe_4x = (sz_lit > ghi_margin_4x) ? l_end - ghi_margin_4x : l_ptr;
    const uint8_t* const l_end_safe_1x = (sz_lit > ghi_margin_1x) ? l_end - ghi_margin_1x : l_ptr;

    uint32_t n_seq = gh.n_sequences;

    // Track bytes written for offset validation
    // For 1-byte offsets (enc_off==1): validate until 256 bytes written (max 8-bit offset)
    // For 2-byte offsets (enc_off==0): validate until 65536 bytes written (max 16-bit offset)
    // After threshold, all offsets are guaranteed valid (can't exceed written bytes)
    // When a dictionary is active, dict_size bytes are logically "already written"
    // (prepended by the caller), so the SAFE loop may be skipped entirely.
    size_t written = dict_size;

    // --- SAFE Loop: offset validation until threshold (4x unroll) ---
    // Since offset is 16-bit, threshold is 65536.
    // For 1-byte offsets (enc_off==1): validate until 256 bytes written
    // For 2-byte offsets (enc_off==0): validate until 65536 bytes written
    const size_t bounds_threshold = (gh.enc_off == 1) ? (1U << 8) : (1U << 16);

    if (safe) {
        /* SAFE variant: save per-batch state so an OVERFLOW can rollback and
         * hand over to the 1x loop / Safe Path. Wild writes already committed
         * are deterministically overwritten when the 1x loop replays. */
        while (n_seq >= 4 && d_ptr < d_end_fast && l_ptr < l_end_safe_4x &&
               written < bounds_threshold) {
            const uint8_t* const t_save = seq_ptr;
            const uint8_t* const e_save = extras_ptr;
            uint8_t* const d_save = d_ptr;
            const uint8_t* const l_save = l_ptr;
            const size_t w_save = written;
            uint32_t s1 = zxc_le32(seq_ptr);
            uint32_t s2 = zxc_le32(seq_ptr + sizeof(uint32_t));
            uint32_t s3 = zxc_le32(seq_ptr + 2 * sizeof(uint32_t));
            uint32_t s4 = zxc_le32(seq_ptr + 3 * sizeof(uint32_t));
            seq_ptr += 4 * sizeof(uint32_t);

            uint64_t ll1 = s1 >> 24;
            if (UNLIKELY(ll1 == ZXC_SEQ_LL_MASK)) {
                ll1 += zxc_read_varint(&extras_ptr, extras_end);
                const uint64_t reserve = (s2 >> 24) + (s3 >> 24) + (s4 >> 24);
                if (UNLIKELY(ll1 + reserve > (size_t)(l_end - l_ptr) ||
                             ll1 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            uint32_t m1b = (s1 >> 16) & 0xFF;
            uint64_t ml1 = m1b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m1b == ZXC_SEQ_ML_MASK)) {
                ml1 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll1 + ml1 + 3U * ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            uint32_t off1 = (s1 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_SAFE(ll1, ml1, off1);

            uint64_t ll2 = s2 >> 24;
            if (UNLIKELY(ll2 == ZXC_SEQ_LL_MASK)) {
                ll2 += zxc_read_varint(&extras_ptr, extras_end);
                const uint64_t reserve = (s3 >> 24) + (s4 >> 24);
                if (UNLIKELY(ll2 + reserve > (size_t)(l_end - l_ptr) ||
                             ll2 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            uint32_t m2b = (s2 >> 16) & 0xFF;
            uint64_t ml2 = m2b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m2b == ZXC_SEQ_ML_MASK)) {
                ml2 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll2 + ml2 + 2U * ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            uint32_t off2 = (s2 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_SAFE(ll2, ml2, off2);

            uint64_t ll3 = s3 >> 24;
            if (UNLIKELY(ll3 == ZXC_SEQ_LL_MASK)) {
                ll3 += zxc_read_varint(&extras_ptr, extras_end);
                const uint64_t reserve = (s4 >> 24);
                if (UNLIKELY(ll3 + reserve > (size_t)(l_end - l_ptr) ||
                             ll3 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            uint32_t m3b = (s3 >> 16) & 0xFF;
            uint64_t ml3 = m3b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m3b == ZXC_SEQ_ML_MASK)) {
                ml3 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll3 + ml3 + ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            uint32_t off3 = (s3 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_SAFE(ll3, ml3, off3);

            uint64_t ll4 = s4 >> 24;
            if (UNLIKELY(ll4 == ZXC_SEQ_LL_MASK)) {
                ll4 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll4 > (size_t)(l_end - l_ptr) ||
                             ll4 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            uint32_t m4b = (s4 >> 16) & 0xFF;
            uint64_t ml4 = m4b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m4b == ZXC_SEQ_ML_MASK)) {
                ml4 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll4 + ml4 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_safe_4x;
            }
            uint32_t off4 = (s4 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_SAFE(ll4, ml4, off4);

            n_seq -= 4;
            continue;

        rollback_safe_4x:
            seq_ptr = t_save;
            extras_ptr = e_save;
            d_ptr = d_save;
            l_ptr = l_save;
            written = w_save;
            break;
        }
    } else {
        while (n_seq >= 4 && d_ptr < d_end_fast && l_ptr < l_end_safe_4x &&
               written < bounds_threshold) {
            uint32_t s1 = zxc_le32(seq_ptr);
            uint32_t s2 = zxc_le32(seq_ptr + sizeof(uint32_t));
            uint32_t s3 = zxc_le32(seq_ptr + 2 * sizeof(uint32_t));
            uint32_t s4 = zxc_le32(seq_ptr + 3 * sizeof(uint32_t));
            seq_ptr += 4 * sizeof(uint32_t);

            uint64_t ll1 = s1 >> 24;
            if (UNLIKELY(ll1 == ZXC_SEQ_LL_MASK)) {
                ll1 += zxc_read_varint(&extras_ptr, extras_end);
                const uint64_t reserve = (s2 >> 24) + (s3 >> 24) + (s4 >> 24);
                if (UNLIKELY(ll1 + reserve > (size_t)(l_end - l_ptr) ||
                             ll1 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t m1b = (s1 >> 16) & 0xFF;
            uint64_t ml1 = m1b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m1b == ZXC_SEQ_ML_MASK)) {
                ml1 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll1 + ml1 + 3U * ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t off1 = (s1 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_SAFE(ll1, ml1, off1);

            uint64_t ll2 = s2 >> 24;
            if (UNLIKELY(ll2 == ZXC_SEQ_LL_MASK)) {
                ll2 += zxc_read_varint(&extras_ptr, extras_end);
                const uint64_t reserve = (s3 >> 24) + (s4 >> 24);
                if (UNLIKELY(ll2 + reserve > (size_t)(l_end - l_ptr) ||
                             ll2 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t m2b = (s2 >> 16) & 0xFF;
            uint64_t ml2 = m2b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m2b == ZXC_SEQ_ML_MASK)) {
                ml2 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll2 + ml2 + 2U * ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t off2 = (s2 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_SAFE(ll2, ml2, off2);

            uint64_t ll3 = s3 >> 24;
            if (UNLIKELY(ll3 == ZXC_SEQ_LL_MASK)) {
                ll3 += zxc_read_varint(&extras_ptr, extras_end);
                const uint64_t reserve = (s4 >> 24);
                if (UNLIKELY(ll3 + reserve > (size_t)(l_end - l_ptr) ||
                             ll3 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t m3b = (s3 >> 16) & 0xFF;
            uint64_t ml3 = m3b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m3b == ZXC_SEQ_ML_MASK)) {
                ml3 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll3 + ml3 + ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t off3 = (s3 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_SAFE(ll3, ml3, off3);

            uint64_t ll4 = s4 >> 24;
            if (UNLIKELY(ll4 == ZXC_SEQ_LL_MASK)) {
                ll4 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll4 > (size_t)(l_end - l_ptr) ||
                             ll4 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t m4b = (s4 >> 16) & 0xFF;
            uint64_t ml4 = m4b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m4b == ZXC_SEQ_ML_MASK)) {
                ml4 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll4 + ml4 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t off4 = (s4 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_SAFE(ll4, ml4, off4);

            n_seq -= 4;
        }
    }

    // --- SAFE Loop tail: remaining sequences with offset validation (1x) ---
    while (n_seq > 0 && d_ptr < d_end_safe && written < bounds_threshold) {
        uint32_t seq = zxc_le32(seq_ptr);
        seq_ptr += sizeof(uint32_t);

        uint64_t ll = seq >> 24;
        if (UNLIKELY(ll == ZXC_SEQ_LL_MASK)) ll += zxc_read_varint(&extras_ptr, extras_end);

        uint32_t m_bits = (seq >> 16) & 0xFF;
        uint64_t ml = m_bits + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m_bits == ZXC_SEQ_ML_MASK)) ml += zxc_read_varint(&extras_ptr, extras_end);

        uint32_t offset = (seq & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;

        // Strict bounds check: sequence must fit, AND wild copies must not overshoot
        // Check both destination (d_ptr) and source literal stream (l_ptr)
        if (UNLIKELY(ll + ml + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr) ||
                     ll + ZXC_PAD_SIZE > (size_t)(l_end - l_ptr))) {
            // Fallback to exact copy (slow but safe)
            if (UNLIKELY(d_ptr + ll > d_end || l_ptr + ll > l_end)) return ZXC_ERROR_OVERFLOW;
            ZXC_MEMCPY(d_ptr, l_ptr, ll);
            l_ptr += ll;
            d_ptr += ll;
            written += ll;

            if (UNLIKELY(offset > written || d_ptr + ml > d_end)) return ZXC_ERROR_BAD_OFFSET;
            const uint8_t* match_src = d_ptr - offset;

            if (offset < ml) {
                for (size_t i = 0; i < ml; i++) d_ptr[i] = match_src[i];
            } else {
                ZXC_MEMCPY(d_ptr, match_src, ml);
            }
            d_ptr += ml;
            written += ml;
        } else {
            // Safe to process with wild copies
            DECODE_SEQ_SAFE(ll, ml, offset);
        }
        n_seq--;
    }

    // --- FAST Loop: After threshold, check large margin to avoid individual bounds checks ---
    if (safe) {
        while (n_seq >= 4 && d_ptr < d_end_fast && l_ptr < l_end_safe_4x) {
            const uint8_t* const t_save = seq_ptr;
            const uint8_t* const e_save = extras_ptr;
            uint8_t* const d_save = d_ptr;
            const uint8_t* const l_save = l_ptr;
            uint32_t s1 = zxc_le32(seq_ptr);
            uint32_t s2 = zxc_le32(seq_ptr + sizeof(uint32_t));
            uint32_t s3 = zxc_le32(seq_ptr + 2 * sizeof(uint32_t));
            uint32_t s4 = zxc_le32(seq_ptr + 3 * sizeof(uint32_t));
            seq_ptr += 4 * sizeof(uint32_t);

            // Prefetch ahead in literal and extras streams to hide memory latency
            ZXC_PREFETCH_READ(l_ptr + ZXC_CACHE_LINE_SIZE);

            uint64_t ll1 = s1 >> 24;
            if (UNLIKELY(ll1 == ZXC_SEQ_LL_MASK)) {
                ll1 += zxc_read_varint(&extras_ptr, extras_end);
                const uint64_t reserve = (s2 >> 24) + (s3 >> 24) + (s4 >> 24);
                if (UNLIKELY(ll1 + reserve > (size_t)(l_end - l_ptr) ||
                             ll1 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            uint32_t m1b = (s1 >> 16) & 0xFF;
            uint64_t ml1 = m1b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m1b == ZXC_SEQ_ML_MASK)) {
                ml1 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll1 + ml1 + 3U * ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            uint32_t off1 = (s1 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_FAST(ll1, ml1, off1);

            uint64_t ll2 = s2 >> 24;
            if (UNLIKELY(ll2 == ZXC_SEQ_LL_MASK)) {
                ll2 += zxc_read_varint(&extras_ptr, extras_end);
                const uint64_t reserve = (s3 >> 24) + (s4 >> 24);
                if (UNLIKELY(ll2 + reserve > (size_t)(l_end - l_ptr) ||
                             ll2 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            uint32_t m2b = (s2 >> 16) & 0xFF;
            uint64_t ml2 = m2b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m2b == ZXC_SEQ_ML_MASK)) {
                ml2 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll2 + ml2 + 2U * ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            uint32_t off2 = (s2 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_FAST(ll2, ml2, off2);

            uint64_t ll3 = s3 >> 24;
            if (UNLIKELY(ll3 == ZXC_SEQ_LL_MASK)) {
                ll3 += zxc_read_varint(&extras_ptr, extras_end);
                const uint64_t reserve = (s4 >> 24);
                if (UNLIKELY(ll3 + reserve > (size_t)(l_end - l_ptr) ||
                             ll3 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            uint32_t m3b = (s3 >> 16) & 0xFF;
            uint64_t ml3 = m3b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m3b == ZXC_SEQ_ML_MASK)) {
                ml3 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll3 + ml3 + ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            uint32_t off3 = (s3 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_FAST(ll3, ml3, off3);

            uint64_t ll4 = s4 >> 24;
            if (UNLIKELY(ll4 == ZXC_SEQ_LL_MASK)) {
                ll4 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll4 > (size_t)(l_end - l_ptr) ||
                             ll4 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            uint32_t m4b = (s4 >> 16) & 0xFF;
            uint64_t ml4 = m4b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m4b == ZXC_SEQ_ML_MASK)) {
                ml4 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll4 + ml4 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    goto rollback_fast_4x;
            }
            uint32_t off4 = (s4 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_FAST(ll4, ml4, off4);

            n_seq -= 4;
            continue;

        rollback_fast_4x:
            seq_ptr = t_save;
            extras_ptr = e_save;
            d_ptr = d_save;
            l_ptr = l_save;
            break;
        }
    } else {
        while (n_seq >= 4 && d_ptr < d_end_fast && l_ptr < l_end_safe_4x) {
            uint32_t s1 = zxc_le32(seq_ptr);
            uint32_t s2 = zxc_le32(seq_ptr + sizeof(uint32_t));
            uint32_t s3 = zxc_le32(seq_ptr + 2 * sizeof(uint32_t));
            uint32_t s4 = zxc_le32(seq_ptr + 3 * sizeof(uint32_t));
            seq_ptr += 4 * sizeof(uint32_t);

            // Prefetch ahead in literal and extras streams to hide memory latency
            ZXC_PREFETCH_READ(l_ptr + ZXC_CACHE_LINE_SIZE);

            uint64_t ll1 = s1 >> 24;
            if (UNLIKELY(ll1 == ZXC_SEQ_LL_MASK)) {
                ll1 += zxc_read_varint(&extras_ptr, extras_end);
                const uint64_t reserve = (s2 >> 24) + (s3 >> 24) + (s4 >> 24);
                if (UNLIKELY(ll1 + reserve > (size_t)(l_end - l_ptr) ||
                             ll1 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t m1b = (s1 >> 16) & 0xFF;
            uint64_t ml1 = m1b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m1b == ZXC_SEQ_ML_MASK)) {
                ml1 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll1 + ml1 + 3U * ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t off1 = (s1 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_FAST(ll1, ml1, off1);

            uint64_t ll2 = s2 >> 24;
            if (UNLIKELY(ll2 == ZXC_SEQ_LL_MASK)) {
                ll2 += zxc_read_varint(&extras_ptr, extras_end);
                const uint64_t reserve = (s3 >> 24) + (s4 >> 24);
                if (UNLIKELY(ll2 + reserve > (size_t)(l_end - l_ptr) ||
                             ll2 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t m2b = (s2 >> 16) & 0xFF;
            uint64_t ml2 = m2b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m2b == ZXC_SEQ_ML_MASK)) {
                ml2 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll2 + ml2 + 2U * ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t off2 = (s2 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_FAST(ll2, ml2, off2);

            uint64_t ll3 = s3 >> 24;
            if (UNLIKELY(ll3 == ZXC_SEQ_LL_MASK)) {
                ll3 += zxc_read_varint(&extras_ptr, extras_end);
                const uint64_t reserve = (s4 >> 24);
                if (UNLIKELY(ll3 + reserve > (size_t)(l_end - l_ptr) ||
                             ll3 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t m3b = (s3 >> 16) & 0xFF;
            uint64_t ml3 = m3b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m3b == ZXC_SEQ_ML_MASK)) {
                ml3 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll3 + ml3 + ZXC_GHI_MAX_INLINE_OUT_PER_SEQ + ZXC_PAD_SIZE >
                             (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t off3 = (s3 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_FAST(ll3, ml3, off3);

            uint64_t ll4 = s4 >> 24;
            if (UNLIKELY(ll4 == ZXC_SEQ_LL_MASK)) {
                ll4 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll4 > (size_t)(l_end - l_ptr) ||
                             ll4 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t m4b = (s4 >> 16) & 0xFF;
            uint64_t ml4 = m4b + ZXC_LZ_MIN_MATCH_LEN;
            if (UNLIKELY(m4b == ZXC_SEQ_ML_MASK)) {
                ml4 += zxc_read_varint(&extras_ptr, extras_end);
                if (UNLIKELY(ll4 + ml4 + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr)))
                    return ZXC_ERROR_OVERFLOW;
            }
            uint32_t off4 = (s4 & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;
            DECODE_SEQ_FAST(ll4, ml4, off4);

            n_seq -= 4;
        }
    }

    // --- Remaining 1 sequence (Fast Path) ---
    while (n_seq > 0 && d_ptr < d_end_safe && l_ptr < l_end_safe_1x) {
        // Save state for fallback
        const uint8_t* seq_save = seq_ptr;
        const uint8_t* ext_save = extras_ptr;

        const uint32_t seq = zxc_le32(seq_ptr);
        seq_ptr += sizeof(uint32_t);

        uint64_t ll = seq >> 24;
        if (UNLIKELY(ll == ZXC_SEQ_LL_MASK)) {
            ll += zxc_read_varint(&extras_ptr, extras_end);
            if (UNLIKELY(l_ptr + ll > l_end)) {
                seq_ptr = seq_save;
                extras_ptr = ext_save;
                break;
            }
        }

        uint32_t m_bits = (seq >> 16) & 0xFF;
        uint64_t ml = m_bits + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m_bits == ZXC_SEQ_ML_MASK)) ml += zxc_read_varint(&extras_ptr, extras_end);

        // Strict bounds checks (including wild copy overrun safety)
        if (UNLIKELY(ll + ml + ZXC_PAD_SIZE > (size_t)(d_end - d_ptr) ||
                     ll + ZXC_PAD_SIZE > (size_t)(l_end - l_ptr))) {
            // Restore state and break to Safe Path
            seq_ptr = seq_save;
            extras_ptr = ext_save;
            break;
        }
        uint32_t offset = (seq & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;

        {
            const uint8_t* src_lit = l_ptr;
            uint8_t* dst_lit = d_ptr;
            zxc_copy32(dst_lit, src_lit);
            if (UNLIKELY(ll > ZXC_PAD_SIZE)) {
                dst_lit += ZXC_PAD_SIZE;
                src_lit += ZXC_PAD_SIZE;
                size_t rem = ll - ZXC_PAD_SIZE;
                while (rem > ZXC_PAD_SIZE) {
                    zxc_copy32(dst_lit, src_lit);
                    dst_lit += ZXC_PAD_SIZE;
                    src_lit += ZXC_PAD_SIZE;
                    rem -= ZXC_PAD_SIZE;
                }
                zxc_copy32(dst_lit, src_lit);
            }
            l_ptr += ll;
            d_ptr += ll;
            written += ll;
        }

        {
            // Skip check if written >= bounds_threshold (256 for 8-bit, 65536 for 16-bit)
            if (UNLIKELY(written < bounds_threshold && offset > written))
                return ZXC_ERROR_BAD_OFFSET;

            /* The loop entry check guarantees ll + ml + ZXC_PAD_SIZE bytes of
             * headroom, so the wild-copy ladder (incl. overlap/fill runs) is safe. */
            zxc_decode_copy_match(d_ptr, offset, ml);
            d_ptr += ml;
            written += ml;
        }
        n_seq--;
    }

    // --- Safe Path for Remaining Sequences ---
    while (n_seq > 0) {
        uint32_t seq = zxc_le32(seq_ptr);
        seq_ptr += sizeof(uint32_t);

        uint64_t ll = seq >> 24;
        if (UNLIKELY(ll == ZXC_SEQ_LL_MASK)) ll += zxc_read_varint(&extras_ptr, extras_end);

        uint32_t m_bits = (seq >> 16) & 0xFF;
        uint64_t ml = m_bits + ZXC_LZ_MIN_MATCH_LEN;
        if (UNLIKELY(m_bits == ZXC_SEQ_ML_MASK)) ml += zxc_read_varint(&extras_ptr, extras_end);
        uint32_t offset = (seq & 0xFFFF) + ZXC_LZ_OFFSET_BIAS;

        if (UNLIKELY(ll + ml > (size_t)(d_end - d_ptr) || l_ptr + ll > l_end))
            return ZXC_ERROR_OVERFLOW;
        ZXC_MEMCPY(d_ptr, l_ptr, ll);
        l_ptr += ll;
        d_ptr += ll;

        const uint8_t* match_src = d_ptr - offset;
        if (UNLIKELY(match_src < dst - dict_size)) return ZXC_ERROR_BAD_OFFSET;

        if (offset < ml) {
            for (size_t i = 0; i < ml; i++) d_ptr[i] = match_src[i];
        } else {
            ZXC_MEMCPY(d_ptr, match_src, ml);
        }
        d_ptr += ml;
        n_seq--;
    }

    // --- Trailing Literals ---
    // Copy remaining literals from source stream (literal exhaustion)
    if (UNLIKELY(l_ptr > l_end)) return ZXC_ERROR_CORRUPT_DATA;
    const size_t remaining_literals = (size_t)(l_end - l_ptr);
    if (remaining_literals > 0) {
        if (UNLIKELY(d_ptr + remaining_literals > d_end)) return ZXC_ERROR_OVERFLOW;
        ZXC_MEMCPY(d_ptr, l_ptr, remaining_literals);
        d_ptr += remaining_literals;
    }

    return (int)(d_ptr - dst);
}

/**
 * @brief Decode a no-dict GLO block (plain, inlinable path).
 *
 * Wrapper over @ref zxc_decode_block_glo_impl with @c safe=0, @c has_dict=0, so
 * the no-dict chunk wrapper inlines it exactly like the dict-free build.
 *
 * @param[in,out] ctx          Decompression context.
 * @param[in]     src          Compressed GLO block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static int zxc_decode_block_glo(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                const size_t src_size, uint8_t* RESTRICT dst,
                                const size_t dst_capacity) {
    return zxc_decode_block_glo_impl(ctx, src, src_size, dst, dst_capacity, 0, 0);
}

/**
 * @brief Decode a no-dict GHI block (plain, inlinable path).
 *
 * Wrapper over @ref zxc_decode_block_ghi_impl with @c safe=0, @c has_dict=0.
 *
 * @param[in,out] ctx          Decompression context.
 * @param[in]     src          Compressed GHI block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static int zxc_decode_block_ghi(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                const size_t src_size, uint8_t* RESTRICT dst,
                                const size_t dst_capacity) {
    return zxc_decode_block_ghi_impl(ctx, src, src_size, dst, dst_capacity, 0, 0);
}

/**
 * @brief Decode a GLO block against a dictionary prefix (cold path).
 *
 * Wrapper over @ref zxc_decode_block_glo_impl with @c safe=0, @c has_dict=1.
 * NOINLINE: only reached on the cold dict path (@ref zxc_decompress_chunk_wrapper_dict),
 * so it never loads into I-cache on a no-dict stream.
 *
 * @param[in,out] ctx          Decompression context (dict prefix in its buffer).
 * @param[in]     src          Compressed GLO block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static ZXC_NOINLINE int zxc_decode_block_glo_dict(const zxc_cctx_t* RESTRICT ctx,
                                                  const uint8_t* RESTRICT src,
                                                  const size_t src_size, uint8_t* RESTRICT dst,
                                                  const size_t dst_capacity) {
    return zxc_decode_block_glo_impl(ctx, src, src_size, dst, dst_capacity, 0, 1);
}

/**
 * @brief Decode a GHI block against a dictionary prefix (cold path).
 *
 * Wrapper over @ref zxc_decode_block_ghi_impl with @c safe=0, @c has_dict=1
 * (NOINLINE; see @ref zxc_decode_block_glo_dict).
 *
 * @param[in,out] ctx          Decompression context (dict prefix in its buffer).
 * @param[in]     src          Compressed GHI block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static ZXC_NOINLINE int zxc_decode_block_ghi_dict(const zxc_cctx_t* RESTRICT ctx,
                                                  const uint8_t* RESTRICT src,
                                                  const size_t src_size, uint8_t* RESTRICT dst,
                                                  const size_t dst_capacity) {
    return zxc_decode_block_ghi_impl(ctx, src, src_size, dst, dst_capacity, 0, 1);
}

/**
 * @brief Decode a GLO block with the strict-tail safe loop (no wild copies).
 *
 * Wrapper over @ref zxc_decode_block_glo_impl with @c safe=1, @c has_dict=0.
 * The safe path never carries a dict (block_safe routes dict inputs to the
 * bounce path), so @c has_dict=0 folds the dead dict handling.
 *
 * @param[in,out] ctx          Decompression context.
 * @param[in]     src          Compressed GLO block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer (capacity == exact decoded size).
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static ZXC_NOINLINE int zxc_decode_block_glo_safe(const zxc_cctx_t* RESTRICT ctx,
                                                  const uint8_t* RESTRICT src,
                                                  const size_t src_size, uint8_t* RESTRICT dst,
                                                  const size_t dst_capacity) {
    return zxc_decode_block_glo_impl(ctx, src, src_size, dst, dst_capacity, 1, 0);
}

/**
 * @brief Decode a GHI block with the strict-tail safe loop (no wild copies).
 *
 * Wrapper over @ref zxc_decode_block_ghi_impl with @c safe=1, @c has_dict=0
 * (the strict-tail safe path never carries a dict; see
 * @ref zxc_decode_block_glo_safe).
 *
 * @param[in,out] ctx          Decompression context.
 * @param[in]     src          Compressed GHI block payload.
 * @param[in]     src_size     Size of @p src in bytes.
 * @param[out]    dst          Destination buffer (capacity == exact decoded size).
 * @param[in]     dst_capacity Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static ZXC_NOINLINE int zxc_decode_block_ghi_safe(const zxc_cctx_t* RESTRICT ctx,
                                                  const uint8_t* RESTRICT src,
                                                  const size_t src_size, uint8_t* RESTRICT dst,
                                                  const size_t dst_capacity) {
    return zxc_decode_block_ghi_impl(ctx, src, src_size, dst, dst_capacity, 1, 0);
}

#undef DECODE_SEQ_FAST
#undef DECODE_SEQ_SAFE

/**
 * @brief Shared chunk-decode body: validates the block header, verifies the
 *        optional checksum, then dispatches on block type.
 *
 * @p has_dict is a compile-time constant: the no-dict instantiation folds the
 * GLO/GHI selection to the plain (inlinable) decoders, so
 * @ref zxc_decompress_chunk_wrapper carries no dict code and matches the
 * dict-free build; the dict instantiation calls the NOINLINE @c _dict decoders.
 *
 * @param[in,out] ctx       Decompression context.
 * @param[in]     src       Compressed block (header + payload + optional checksum).
 * @param[in]     src_sz    Size of @p src in bytes.
 * @param[out]    dst       Destination buffer for the decoded block.
 * @param[in]     dst_cap   Capacity of @p dst in bytes.
 * @param[in]     has_dict  Compile-time flag: 1 = dictionary-aware decoders.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
static ZXC_ALWAYS_INLINE int zxc_decompress_chunk_wrapper_body(
    const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src, const size_t src_sz,
    uint8_t* RESTRICT dst, const size_t dst_cap, const int has_dict) {
    if (UNLIKELY(src_sz < ZXC_BLOCK_HEADER_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;

    const uint8_t type = src[0];
    const uint32_t comp_sz = zxc_le32(src + 3);
    const int has_crc = ctx->checksum_enabled;

    // Check bounds: Header + Body + Checksum(if any)
    const size_t expected_sz =
        (size_t)ZXC_BLOCK_HEADER_SIZE + comp_sz + (has_crc ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
    if (UNLIKELY(src_sz < expected_sz)) return ZXC_ERROR_SRC_TOO_SMALL;

    const uint8_t* data = src + ZXC_BLOCK_HEADER_SIZE;

    if (has_crc) {
        const uint32_t stored = zxc_le32(data + comp_sz);
        const uint32_t calc = zxc_checksum(data, comp_sz, ZXC_CHECKSUM_RAPIDHASH);
        if (UNLIKELY(stored != calc)) return ZXC_ERROR_BAD_CHECKSUM;
    }

    int decoded_sz = ZXC_ERROR_BAD_BLOCK_TYPE;

    switch (type) {
        case ZXC_BLOCK_GLO:
            decoded_sz = has_dict ? zxc_decode_block_glo_dict(ctx, data, comp_sz, dst, dst_cap)
                                  : zxc_decode_block_glo(ctx, data, comp_sz, dst, dst_cap);
            break;
        case ZXC_BLOCK_GHI:
            decoded_sz = has_dict ? zxc_decode_block_ghi_dict(ctx, data, comp_sz, dst, dst_cap)
                                  : zxc_decode_block_ghi(ctx, data, comp_sz, dst, dst_cap);
            break;
        case ZXC_BLOCK_RAW:
            // For RAW blocks, comp_sz == raw_sz (uncompressed data stored as-is)
            if (UNLIKELY(comp_sz > dst_cap)) return ZXC_ERROR_DST_TOO_SMALL;
            ZXC_MEMCPY(dst, data, comp_sz);
            decoded_sz = (int)comp_sz;
            break;
        case ZXC_BLOCK_EOF:
            // EOF should be handled by the dispatcher, not here
            return ZXC_ERROR_CORRUPT_DATA;
        default:
            return ZXC_ERROR_BAD_BLOCK_TYPE;
    }

    return decoded_sz;
}

/**
 * @brief Public no-dict chunk decoder (decompression hot path).
 *
 * Inlines the plain GLO/GHI decoders via @ref zxc_decompress_chunk_wrapper_body
 * with @c has_dict=0, so it carries no dict code and matches the dict-free build.
 *
 * @param[in,out] ctx     Decompression context.
 * @param[in]     src     Compressed block bytes.
 * @param[in]     src_sz  Size of @p src in bytes.
 * @param[out]    dst     Destination buffer for the decoded block.
 * @param[in]     dst_cap Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
// cppcheck-suppress unusedFunction
int zxc_decompress_chunk_wrapper(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                 const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap) {
    return zxc_decompress_chunk_wrapper_body(ctx, src, src_sz, dst, dst_cap, 0);
}

/**
 * @brief Public dictionary chunk decoder.
 *
 * Routes through @ref zxc_decompress_chunk_wrapper_body with @c has_dict=1,
 * which calls the NOINLINE @c _dict decoders (slower: dict back-refs read the
 * prepended dictionary). Used only when @c ctx->dict_size != 0.
 *
 * @param[in,out] ctx     Decompression context (dict prefix in its buffer).
 * @param[in]     src     Compressed block bytes.
 * @param[in]     src_sz  Size of @p src in bytes.
 * @param[out]    dst     Destination buffer for the decoded block.
 * @param[in]     dst_cap Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
// cppcheck-suppress unusedFunction
int zxc_decompress_chunk_wrapper_dict(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap) {
    return zxc_decompress_chunk_wrapper_body(ctx, src, src_sz, dst, dst_cap, 1);
}

/**
 * @brief Public strict-tail safe chunk decoder (dst_cap == exact decoded size).
 *
 * Validates the block header and optional checksum, then decodes via the
 * @c _safe decoders (no bounce buffer, no tail padding); RAW blocks are copied
 * directly. Dict inputs are not handled here (the caller routes them to the
 * bounce-capable path).
 *
 * @param[in,out] ctx     Decompression context.
 * @param[in]     src     Compressed block bytes.
 * @param[in]     src_sz  Size of @p src in bytes.
 * @param[out]    dst     Destination buffer (capacity == exact decoded size).
 * @param[in]     dst_cap Capacity of @p dst in bytes.
 * @return Bytes written on success, or a negative @ref zxc_error_t.
 */
// cppcheck-suppress unusedFunction
int zxc_decompress_chunk_wrapper_safe(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap) {
    if (UNLIKELY(src_sz < ZXC_BLOCK_HEADER_SIZE)) return ZXC_ERROR_SRC_TOO_SMALL;

    const uint8_t type = src[0];
    const uint32_t comp_sz = zxc_le32(src + 3);
    const int has_crc = ctx->checksum_enabled;

    const size_t expected_sz =
        (size_t)ZXC_BLOCK_HEADER_SIZE + comp_sz + (has_crc ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
    if (UNLIKELY(src_sz < expected_sz)) return ZXC_ERROR_SRC_TOO_SMALL;

    const uint8_t* data = src + ZXC_BLOCK_HEADER_SIZE;

    if (has_crc) {
        const uint32_t stored = zxc_le32(data + comp_sz);
        const uint32_t calc = zxc_checksum(data, comp_sz, ZXC_CHECKSUM_RAPIDHASH);
        if (UNLIKELY(stored != calc)) return ZXC_ERROR_BAD_CHECKSUM;
    }

    switch (type) {
        case ZXC_BLOCK_GLO:
            return zxc_decode_block_glo_safe(ctx, data, comp_sz, dst, dst_cap);
        case ZXC_BLOCK_GHI:
            return zxc_decode_block_ghi_safe(ctx, data, comp_sz, dst, dst_cap);
        case ZXC_BLOCK_RAW:
            if (UNLIKELY(comp_sz > dst_cap)) return ZXC_ERROR_DST_TOO_SMALL;
            ZXC_MEMCPY(dst, data, comp_sz);
            return (int)comp_sz;
        case ZXC_BLOCK_EOF:
            return ZXC_ERROR_CORRUPT_DATA;
        default:
            return ZXC_ERROR_BAD_BLOCK_TYPE;
    }
}
