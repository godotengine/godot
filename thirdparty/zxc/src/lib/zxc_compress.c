/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_compress.c
 * @brief Block-level compression: LZ77 parsing, GLO / GHI / RAW encoding,
 *        and the chunk-wrapper entry point.
 *
 * Compiled multiple times with different @c ZXC_FUNCTION_SUFFIX values to
 * produce AVX2, AVX-512, NEON, and scalar variants dispatched at runtime
 * by @ref zxc_dispatch.c.
 */

/*
 * Function Multi-Versioning Support
 * If ZXC_FUNCTION_SUFFIX is defined (e.g. _avx2, _neon), rename the public
 * entry point AND the Huffman entry points consumed by this TU. The defines
 * sit before zxc_internal.h so that the prototypes the header declares are
 * also rewritten with the suffix, keeping callers and callees consistent.
 */
#ifdef ZXC_FUNCTION_SUFFIX
#define ZXC_CAT_IMPL(x, y) x##y
#define ZXC_CAT(x, y) ZXC_CAT_IMPL(x, y)
#define zxc_compress_chunk_wrapper ZXC_CAT(zxc_compress_chunk_wrapper, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_build_code_lengths ZXC_CAT(zxc_huf_build_code_lengths, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_encode_section ZXC_CAT(zxc_huf_encode_section, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_encode_section_dict ZXC_CAT(zxc_huf_encode_section_dict, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_unpack_lengths ZXC_CAT(zxc_huf_unpack_lengths, ZXC_FUNCTION_SUFFIX)
#endif

#include "../../include/zxc_error.h"
#include "zxc_internal.h"

/**
 * @brief Computes a hash value for either a 4-byte or 5-byte sequence.
 *
 * @param[in] val The 64-bit integer sequence (e.g., 8 bytes read from input stream).
 * @param[in] use_hash5 Non-zero to use the 5-byte xorshift64* hash (Marsaglia/Vigna), zero for
 * 4-byte Marsaglia hash.
 * @return uint32_t A hash value suitable for indexing the match table.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_hash_func(const uint64_t val, const int use_hash5) {
    if (use_hash5) {
        const uint64_t v5 = val & 0xFFFFFFFFFFULL;
        return (uint32_t)((v5 * ZXC_LZ_HASH_PRIME2) >> (64 - ZXC_LZ_HASH_BITS));
    } else {
        const uint64_t v4 = val ^ (val >> 15);
        return ((uint32_t)v4 * ZXC_LZ_HASH_PRIME1) >> (32 - ZXC_LZ_HASH_BITS);
    }
}

#if defined(ZXC_USE_AVX2)
/**
 * @brief Reduces a 256-bit integer vector to a single scalar by finding the maximum unsigned 32-bit
 * integer element.
 *
 * This function performs a horizontal reduction across the 8 packed 32-bit unsigned integers
 * in the source vector to determine the maximum value.
 *
 * @param[in] v The 256-bit vector containing 8 unsigned 32-bit integers.
 * @return The maximum unsigned 32-bit integer found in the vector.
 */
// codeql[cpp/unused-static-function] : Used conditionally when ZXC_USE_AVX2 is defined
static ZXC_ALWAYS_INLINE uint32_t zxc_mm256_reduce_max_epu32(__m256i v) {
    __m128i vlow = _mm256_castsi256_si128(v);        // Extract the lower 128 bits
    __m128i vhigh = _mm256_extracti128_si256(v, 1);  // Extract the upper 128 bits
    vlow = _mm_max_epu32(vlow, vhigh);               // Element-wise max of lower and upper halves
    __m128i vshuf = _mm_shuffle_epi32(vlow, _MM_SHUFFLE(1, 0, 3, 2));  // Shuffle to swap pairs
    vlow = _mm_max_epu32(vlow, vshuf);  // Max of original and swapped
    vshuf =
        _mm_shuffle_epi32(vlow, _MM_SHUFFLE(2, 3, 0, 1));  // Shuffle to bring remaining candidates
    vlow = _mm_max_epu32(vlow, vshuf);                     // Final max comparison
    return (uint32_t)_mm_cvtsi128_si32(vlow);              // Extract the scalar result
}
#endif

#if defined(ZXC_USE_SSE2)
/**
 * @brief SSE2 emulation of SSE4.1 @c _mm_blendv_epi8.
 *
 * Selects bytes from @p b where the corresponding @p mask byte has its high bit
 * set, else from @p a. In every call site the mask lanes are full-width compare
 * results (all-ones or all-zero per element), so a plain bitwise select is exact.
 *
 * @param[in] a     Lanes chosen where @p mask is clear.
 * @param[in] b     Lanes chosen where @p mask is set.
 * @param[in] mask  Per-byte selector (a full-width compare result).
 * @return The blended 128-bit vector.
 */
// codeql[cpp/unused-static-function] : Used conditionally when ZXC_USE_SSE2 is defined
static ZXC_ALWAYS_INLINE __m128i zxc_mm_blendv_epi8_sse2(__m128i a, __m128i b, __m128i mask) {
    return _mm_or_si128(_mm_and_si128(mask, b), _mm_andnot_si128(mask, a));
}

/**
 * @brief SSE2 emulation of SSE4.1 @c _mm_packus_epi32 (saturating u32 -> u16).
 *
 * SSE2 only has signed @c _mm_packs_epi32 (saturates to int16). Bias each lane
 * by -0x8000 so values in [0, 0xFFFF] land in the signed int16 range with no
 * saturation, pack, then add 0x8000 back per 16-bit lane. Exact for inputs in
 * [0, 0xFFFF] (all call sites pass match lengths < 2^16).
 *
 * @param[in] a  Four u32 lanes forming the low half of the result.
 * @param[in] b  Four u32 lanes forming the high half of the result.
 * @return The eight u16 lanes packed from @p a then @p b.
 */
// codeql[cpp/unused-static-function] : Used conditionally when ZXC_USE_SSE2 is defined
static ZXC_ALWAYS_INLINE __m128i zxc_mm_packus_epi32_sse2(__m128i a, __m128i b) {
    const __m128i bias32 = _mm_set1_epi32(0x8000);
    const __m128i bias16 = _mm_set1_epi16((short)0x8000);
    const __m128i pa = _mm_sub_epi32(a, bias32);
    const __m128i pb = _mm_sub_epi32(b, bias32);
    return _mm_add_epi16(_mm_packs_epi32(pa, pb), bias16);
}
#endif

/**
 * @brief Writes a Prefix Varint encoded value to a buffer.
 *
 * This function encodes a 32-bit unsigned integer using Prefix Varint encoding
 * and writes it to the destination buffer. Unary prefix bits in the first
 * byte determine the total length (1-5 bytes), allowing for branchless or
 * predictable decoding.
 *
 * Format:
 * - 0xxxxxxx (1 byte)
 * - 10xxxxxx ... (2 bytes)
 * - 110xxxxx ... (3 bytes)
 * ...
 *
 * @param[out] dst Pointer to the destination buffer where the encoded value will be written.
 * @param[in] val The 32-bit unsigned integer value to encode.
 * @return The number of bytes written to the destination buffer.
 */
static ZXC_ALWAYS_INLINE size_t zxc_write_varint(uint8_t* RESTRICT dst, const uint32_t val) {
    // Refuse to emit varints above ZXC_MAX_VARINT_VALUE: such values are
    // out-of-spec (block_size_max is 2^21, the largest legitimate varint is
    // strictly less) and would be rejected by the decoder. For valid inputs
    // from the Block API (src_size <= ZXC_BLOCK_SIZE_MAX) this never triggers;
    // it is a defense-in-depth check. Callers must treat a return of 0 as an
    // encoding error.
    if (UNLIKELY(val > ZXC_MAX_VARINT_VALUE)) {
        return 0;
    }

    // 1 byte: 0xxxxxxx (7 bits) = 2^7 = 128
    if (LIKELY(val < (1U << 7))) {
        dst[0] = (uint8_t)val;
        return 1;
    }

    // 2 bytes: 10xxxxxx xxxxxxxx (14 bits) = 2^14 = 16384
    if (LIKELY(val < (1U << 14))) {
        dst[0] = (uint8_t)(0x80 | (val & 0x3F));
        dst[1] = (uint8_t)(val >> 6);
        return 2;
    }

    // 3 bytes: 110xxxxx xxxxxxxx xxxxxxxx (21 bits) -> max emittable value,
    // matching ZXC_MAX_VARINT_VALUE = ZXC_BLOCK_SIZE_MAX - 1.
    dst[0] = (uint8_t)(0xC0 | (val & 0x1F));
    dst[1] = (uint8_t)(val >> 5);
    dst[2] = (uint8_t)(val >> 13);
    return 3;
}

/**
 * @brief Structure representing a match found during compression.
 *
 * This structure holds information about a matching sequence found
 * in the input data during the compression process.
 *
 * @param ref       Pointer to the reference data where the match was found.
 * @param len       Length of the matching sequence in bytes.
 * @param backtrack Distance to backtrack from the current position to find the match.
 */
typedef struct {
    const uint8_t* ref;
    uint32_t len;
    uint32_t backtrack;
} zxc_match_t;

/**
 * @brief Finds the best matching sequence for LZ77 compression
 *
 * Uses a split hash table layout:
 * - hash_table[h]  : uint32_t position + epoch (128 KB for 15-bit hash)
 * - hash_tags[h]   : uint8_t tag for fast rejection (32 KB, L1-resident)
 *
 * @param[in] src Pointer to the start of the source buffer.
 * @param[in] ip Current input position pointer.
 * @param[in] iend Pointer to the end of the input buffer.
 * @param[in] search_limit Pointer to the match finding limit.
 * @param[in] anchor Pointer to the current anchor position.
 * @param[in,out] hash_table Pointer to the position table for match finding.
 * @param[in,out] hash_tags Pointer to the tag table for fast rejection.
 * @param[in,out] chain_table Pointer to the chain table for collision handling.
 * @param[in] epoch_mark Current epoch marker for hash table invalidation.
 * @param[in] offset_mask Mask isolating the position bits in chain/table entries.
 * @param[in] level Compression level (selects search depth and matcher behaviour).
 * @param[in] p LZ77 parameters controlling search depth, lazy matching, and stepping.
 * @param[in] last_off Most recently accepted match offset, used as a repeat-offset
 *            seed probed before the hash-chain walk. Pass 0 to disable (all callers
 *            except the level-6 optimal parser do so).
 * @return zxc_match_t Structure containing the best match information
 *         (reference pointer, length of the match, and backtrack distance).
 */
static ZXC_ALWAYS_INLINE zxc_match_t zxc_lz77_find_best_match(
    const uint8_t* src, const uint8_t* ip, const uint8_t* iend, const uint8_t* search_limit,
    const uint8_t* anchor, uint32_t* RESTRICT hash_table, uint8_t* RESTRICT hash_tags,
    uint16_t* RESTRICT chain_table, const uint32_t epoch_mark, const uint32_t offset_mask,
    const int level, const zxc_lz77_params_t p, const uint32_t last_off) {
    const int use_hash5 = (level >= 3);
    // Track the best match found so far.
    //  ref is the pointer to the start of the match in the history buffer,
    //  len is the match length, and backtrack is the distance from ip to ref.
    //  Start with a sentinel length just below the minimum so any valid match will replace it.
    zxc_match_t best = (zxc_match_t){NULL, ZXC_LZ_MIN_MATCH_LEN - 1, 0};

    // Load the 8-byte sequence at the current position.
    uint64_t cur_val8 = zxc_le64(ip);
    uint32_t cur_val = (uint32_t)cur_val8;
    uint32_t h = zxc_hash_func(cur_val8, use_hash5);

    // 8-bit tag: XOR fold of first 4 bytes for fast rejection
    const uint8_t cur_tag = (uint8_t)(cur_val ^ (cur_val >> 16));

    // Current position in the input buffer expressed as a 32-bit index.
    const uint32_t cur_pos = (uint32_t)(ip - src);

    // Tag-first filter on fast levels.
    const uint8_t stored_tag = hash_tags[h];
    uint32_t match_idx;
    if (level <= ZXC_LEVEL_FAST && stored_tag != cur_tag) {
        match_idx = 0;
    } else {
        const uint32_t raw_head = hash_table[h];
        match_idx = ((raw_head & ~offset_mask) == epoch_mark) ? (raw_head & offset_mask) : 0;
    }

    // skip_head still drives the chain walk on level >= 3 (advances past the
    // mismatched head without comparing). On level <= 2 it is always 0 here:
    // either match_idx == 0 (filter-skip) or stored_tag == cur_tag.
    const int skip_head = (match_idx != 0) & (stored_tag != cur_tag);

    // Split table writes
    hash_table[h] = epoch_mark | cur_pos;
    hash_tags[h] = cur_tag;

    // Branchless chain table update
    const uint32_t dist = cur_pos - match_idx;
    const uint32_t valid_mask = -((int32_t)((match_idx != 0) & (dist < ZXC_LZ_WINDOW_SIZE)));
    chain_table[cur_pos & ZXC_LZ_WINDOW_MASK] = (uint16_t)(dist & valid_mask);

    int attempts = p.search_depth;

    /* Repeat-offset seed (level-6 parser passes last_off; others pass 0).
     * Probing the previous offset first often finds the longest match right
     * away, speeding up the chain walk. It only raises best.len, never lowers
     * it, so the result is unchanged - this is purely a speed optimization. */
    if (last_off != 0U && last_off <= (uint32_t)ZXC_LZ_MAX_DIST && last_off <= cur_pos) {
        const uint8_t* const rep_ref = src + (cur_pos - last_off);
        if (zxc_le32(rep_ref) == cur_val) {
            uint32_t mlen = sizeof(uint32_t);
            const uint8_t* const limit_8 = iend - sizeof(uint64_t);
            while (ip + mlen < limit_8) {
                const uint64_t diff = zxc_le64(ip + mlen) ^ zxc_le64(rep_ref + mlen);
                if (diff == 0) {
                    mlen += sizeof(uint64_t);
                } else {
                    mlen += (uint32_t)(zxc_ctz64(diff) >> 3);
                    goto _rep_done;
                }
            }
            while (ip + mlen < iend && rep_ref[mlen] == ip[mlen]) mlen++;
        _rep_done:;
            best.len = mlen;
            best.ref = rep_ref;
            if (UNLIKELY(best.len >= (uint32_t)p.sufficient_len || ip + best.len >= iend))
                goto _finalize_match;
        }
    }

    if (match_idx == 0) goto _finalize_match;

    // Optimization: If head tag doesn't match, advance immediately without loading the first
    // mismatch.
    if (skip_head) {
        const uint16_t delta = chain_table[match_idx & ZXC_LZ_WINDOW_MASK];
        const uint32_t next_idx = match_idx - delta;
        match_idx = (delta != 0) ? next_idx : 0;
        attempts--;
    }

    while (match_idx > 0) {
        if (UNLIKELY(attempts-- < 0 || cur_pos - match_idx > ZXC_LZ_MAX_DIST)) break;
        const uint8_t* ref = src + match_idx;

        // Load the next chain link early (before the compare) so its address
        // resolves while we prefetch.
        const uint16_t delta = chain_table[match_idx & ZXC_LZ_WINDOW_MASK];
        const uint32_t next_idx = match_idx - delta;
        ZXC_PREFETCH_READ(src + next_idx);

        const uint32_t ref_val = zxc_le32(ref);
        const int tag_match = (ref_val == cur_val);
        // Cheap gate: 4-byte tag match, then check the byte past the current
        // best (the && skips that load unless the tag already matched).
        const int should_compare = tag_match && (ref[best.len] == ip[best.len]);

        if (should_compare) {
            uint32_t mlen = sizeof(uint32_t);  // We already know the first 4 bytes match

            // Fast path: Scalar 64-bit comparison for short matches (=< 64 bytes)
            // Most matches are short, so this avoids SIMD overhead for common cases
            const uint8_t* limit_8 = iend - sizeof(uint64_t);
            const uint8_t* scalar_limit = ip + mlen + 64;
            if (scalar_limit > limit_8) scalar_limit = limit_8;

            while (ip + mlen < scalar_limit) {
                uint64_t diff = zxc_le64(ip + mlen) ^ zxc_le64(ref + mlen);
                if (diff == 0)
                    mlen += sizeof(uint64_t);
                else {
                    mlen += (zxc_ctz64(diff) >> 3);
                    goto _match_len_done;
                }
            }

            // Long match path: Use SIMD for matches exceeding 64 bytes
#if defined(ZXC_USE_AVX512)
            const uint8_t* limit_64 = iend - 64;
            while (ip + mlen < limit_64) {
                const __m512i v_src = _mm512_loadu_si512((const void*)(ip + mlen));
                const __m512i v_ref = _mm512_loadu_si512((const void*)(ref + mlen));
                const __mmask64 mask = _mm512_cmpeq_epi8_mask(v_src, v_ref);
                if (mask == 0xFFFFFFFFFFFFFFFF)
                    mlen += 64;
                else {
                    mlen += (uint32_t)zxc_ctz64(~mask);
                    goto _match_len_done;
                }
            }
#elif defined(ZXC_USE_AVX2)
            const uint8_t* limit_32 = iend - 32;
            while (ip + mlen < limit_32) {
                const __m256i v_src = _mm256_loadu_si256((const __m256i*)(ip + mlen));
                const __m256i v_ref = _mm256_loadu_si256((const __m256i*)(ref + mlen));
                const __m256i v_cmp = _mm256_cmpeq_epi8(v_src, v_ref);
                const uint32_t mask = (uint32_t)_mm256_movemask_epi8(v_cmp);
                if (mask == 0xFFFFFFFF)
                    mlen += 32;
                else {
                    mlen += zxc_ctz32(~mask);
                    goto _match_len_done;
                }
            }
#elif defined(ZXC_USE_SSE2)
            const uint8_t* limit_16 = iend - 16;
            while (ip + mlen < limit_16) {
                const __m128i v_src = _mm_loadu_si128((const __m128i*)(ip + mlen));
                const __m128i v_ref = _mm_loadu_si128((const __m128i*)(ref + mlen));
                const __m128i v_cmp = _mm_cmpeq_epi8(v_src, v_ref);
                const uint32_t mask = (uint32_t)_mm_movemask_epi8(v_cmp);
                if (mask == 0xFFFFU)
                    mlen += 16;
                else {
                    // mask != 0xFFFF => a differing byte exists in bits 0..15,
                    // so the lowest set bit of ~mask lies in that range.
                    mlen += zxc_ctz32(~mask);
                    goto _match_len_done;
                }
            }
#elif defined(ZXC_USE_NEON64)
            {
                const uint8_t* limit_32 = iend - 32;
                while (ip + mlen < limit_32) {
                    const uint8x16_t s0 = vld1q_u8(ip + mlen);
                    const uint8x16_t r0 = vld1q_u8(ref + mlen);
                    const uint8x16_t c0 = vceqq_u8(s0, r0);
                    const uint64_t m0 = vget_lane_u64(
                        vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(c0), 4)), 0);
                    if (UNLIKELY(m0 != ~(uint64_t)0)) {
                        mlen += (uint32_t)(zxc_ctz64(~m0) >> 2);
                        goto _match_len_done;
                    }
                    const uint8x16_t s1 = vld1q_u8(ip + mlen + 16);
                    const uint8x16_t r1 = vld1q_u8(ref + mlen + 16);
                    const uint8x16_t c1 = vceqq_u8(s1, r1);
                    const uint64_t m1 = vget_lane_u64(
                        vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(c1), 4)), 0);
                    if (UNLIKELY(m1 != ~(uint64_t)0)) {
                        mlen += 16 + (uint32_t)(zxc_ctz64(~m1) >> 2);
                        goto _match_len_done;
                    }
                    mlen += 32;
                }
                if (ip + mlen < iend - 16) {
                    const uint8x16_t v_src = vld1q_u8(ip + mlen);
                    const uint8x16_t v_ref = vld1q_u8(ref + mlen);
                    const uint8x16_t v_cmp = vceqq_u8(v_src, v_ref);
                    const uint64_t mask = vget_lane_u64(
                        vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(v_cmp), 4)), 0);
                    if (LIKELY(mask == ~(uint64_t)0))
                        mlen += 16;
                    else {
                        mlen += (uint32_t)(zxc_ctz64(~mask) >> 2);
                        goto _match_len_done;
                    }
                }
            }
#elif defined(ZXC_USE_NEON32)
            {
                const uint8_t* limit_16 = iend - 16;
                while (ip + mlen < limit_16) {
                    const uint8x16_t v_src = vld1q_u8(ip + mlen);
                    const uint8x16_t v_ref = vld1q_u8(ref + mlen);
                    const uint8x16_t v_cmp = vceqq_u8(v_src, v_ref);
                    uint8x8_t p1 = vpmin_u8(vget_low_u8(v_cmp), vget_high_u8(v_cmp));
                    uint8x8_t p2 = vpmin_u8(p1, p1);
                    uint8x8_t p3 = vpmin_u8(p2, p2);
                    uint8x8_t p4 = vpmin_u8(p3, p3);
                    uint8_t min_val = vget_lane_u8(p4, 0);
                    if (min_val == 0xFF)
                        mlen += 16;
                    else {
                        uint8x16_t v_diff = vmvnq_u8(v_cmp);
                        uint64_t lo =
                            (uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 0) |
                            ((uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 1) << 32);
                        if (lo != 0)
                            mlen += (zxc_ctz64(lo) >> 3);
                        else
                            mlen +=
                                8 + (zxc_ctz64(
                                         (uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 2) |
                                         ((uint64_t)vgetq_lane_u32(vreinterpretq_u32_u8(v_diff), 3)
                                          << 32)) >>
                                     3);
                        goto _match_len_done;
                    }
                }
            }
#endif
            while (ip + mlen < limit_8) {
                const uint64_t diff = zxc_le64(ip + mlen) ^ zxc_le64(ref + mlen);
                if (diff == 0)
                    mlen += sizeof(uint64_t);
                else {
                    mlen += (zxc_ctz64(diff) >> 3);
                    goto _match_len_done;
                }
            }
            while (ip + mlen < iend && ref[mlen] == ip[mlen]) mlen++;

        _match_len_done:;
            const int better = (mlen > best.len);
            best.len = better ? mlen : best.len;
            best.ref = better ? ref : best.ref;

            if (UNLIKELY(best.len >= (uint32_t)p.sufficient_len || ip + best.len >= iend)) break;
        }

        match_idx = (delta != 0) ? next_idx : 0;
    }

_finalize_match:
    if (best.ref) {
        // Backtrack to extend match backwards
        const uint8_t* b_ip = ip;
        const uint8_t* b_ref = best.ref;
        while (b_ip > anchor && b_ref > src && b_ip[-1] == b_ref[-1]) {
            b_ip--;
            b_ref--;
            best.len++;
            best.backtrack++;
        }
        best.ref = b_ref;
    }

    if (p.use_lazy && best.ref && best.len < (uint32_t)p.lazy_len_threshold &&
        ip + 1 < search_limit) {
        // --- Lazy evaluation at ip+1 ---
        const uint64_t next_val8 = zxc_le64(ip + 1);
        const uint32_t next_val = (uint32_t)next_val8;
        const uint32_t h2 = zxc_hash_func(next_val8, use_hash5);
        const uint8_t next_stored_tag = hash_tags[h2];
        const uint32_t next_head = hash_table[h2];
        uint32_t next_idx =
            (next_head & ~offset_mask) == epoch_mark ? (next_head & offset_mask) : 0;
        const uint8_t next_tag = (uint8_t)(next_val ^ (next_val >> 16));
        const int skip_lazy_head = (next_idx > 0 && next_stored_tag != next_tag);
        uint32_t max_lazy2 = 0;
        int lazy_att = p.lazy_attempts;
        int is_lazy_first = 1;

        while (next_idx > 0) {
            if (UNLIKELY(lazy_att-- <= 0 || (uint32_t)(ip + 1 - src) - next_idx > ZXC_LZ_MAX_DIST))
                break;
            const uint8_t* ref2 = src + next_idx;

            if ((!is_lazy_first || !skip_lazy_head) && zxc_le32(ref2) == next_val) {
                uint32_t l2 = sizeof(uint32_t);
                const uint8_t* limit = iend - sizeof(uint64_t);

                while (ip + 1 + l2 < limit) {
                    const uint64_t v1 = zxc_le64(ip + 1 + l2);
                    const uint64_t v2 = zxc_le64(ref2 + l2);
                    if (v1 != v2) {
                        l2 += (uint32_t)(zxc_ctz64(v1 ^ v2) >> 3);
                        goto lazy2_done;
                    }
                    l2 += sizeof(uint64_t);
                }
                while (ip + 1 + l2 < iend && ref2[l2] == ip[1 + l2]) l2++;
            lazy2_done:
                max_lazy2 = l2 > max_lazy2 ? l2 : max_lazy2;
            }

            const uint16_t delta = chain_table[next_idx & ZXC_LZ_WINDOW_MASK];
            if (UNLIKELY(delta == 0)) break;
            next_idx -= delta;
            is_lazy_first = 0;
        }

        // --- Lazy evaluation at ip+2 (computed in parallel, no dependency on lazy 1) ---
        uint32_t max_lazy3 = 0;
        if (level >= ZXC_LEVEL_BALANCED && ip + 2 < search_limit) {
            const uint64_t val3_8 = zxc_le64(ip + 2);
            const uint32_t val3 = (uint32_t)val3_8;
            const uint32_t h3 = zxc_hash_func(val3_8, use_hash5);
            const uint8_t tag3 = hash_tags[h3];
            const uint32_t head3 = hash_table[h3];
            uint32_t idx3 = (head3 & ~offset_mask) == epoch_mark ? (head3 & offset_mask) : 0;
            const uint8_t cur_tag3 = (uint8_t)(val3 ^ (val3 >> 16));
            const int skip_head3 = (idx3 > 0 && tag3 != cur_tag3);

            int is_first3 = 1;
            lazy_att = p.lazy_attempts;
            while (idx3 > 0) {
                if (UNLIKELY(lazy_att-- <= 0 || (uint32_t)(ip + 2 - src) - idx3 > ZXC_LZ_MAX_DIST))
                    break;

                const uint8_t* ref3 = src + idx3;
                if ((!is_first3 || !skip_head3) && zxc_le32(ref3) == val3) {
                    uint32_t l3 = sizeof(uint32_t);
                    const uint8_t* limit = iend - sizeof(uint64_t);

                    while (ip + 2 + l3 < limit) {
                        const uint64_t v1 = zxc_le64(ip + 2 + l3);
                        const uint64_t v2 = zxc_le64(ref3 + l3);
                        if (v1 != v2) {
                            l3 += (uint32_t)(zxc_ctz64(v1 ^ v2) >> 3);
                            goto lazy3_done;
                        }
                        l3 += sizeof(uint64_t);
                    }
                    while (ip + 2 + l3 < iend && ref3[l3] == ip[2 + l3]) l3++;
                lazy3_done:
                    max_lazy3 = l3 > max_lazy3 ? l3 : max_lazy3;
                }

                const uint16_t delta = chain_table[idx3 & ZXC_LZ_WINDOW_MASK];
                if (UNLIKELY(delta == 0)) break;
                idx3 -= delta;
                is_first3 = 0;
            }
        }

        // Single decision: invalidate if either lazy position found a better match
        if (max_lazy2 > best.len + 1 || max_lazy3 > best.len + 2) best.ref = NULL;
    }

    return best;
}

/**
 * @brief Update dp[p + L_start .. p + L_end) with a constant transition
 *        cost, in parallel where the target ISA allows.
 *
 * For each L in [L_start, L_end), if @p nxt is strictly less than the
 * current dp[p+L], rewrite dp/parent_len/parent_off in lockstep: same
 * semantics as the scalar update inside ::zxc_lz77_optimal_parse_glo.
 * Caller guarantees @p nxt is independent of L (the cost of the L-th
 * transition does not vary across the requested span).
 *
 * Vectorized prologue per ISA, falling through to a scalar tail:
 *   - AVX-512 BW + VL : 16-wide via vpcmpud + vmask{store,storeu}.
 *                       Falls back to AVX2 if VL is absent.
 *   - AVX2            : 8-wide via biased vpcmpgt + vpblendvb (no 32-bit
 *                       unsigned cmpgt before AVX-512). parent_off is
 *                       updated with a packed 8x16 mask + 128-bit blend.
 *   - NEON64 / NEON32 : 4-wide via vcgtq_u32 + vbslq_u32, with vmovn_u32
 *                       to narrow the mask for the 4x16 parent_off update.
 *
 * @param[in,out] dp         DP cost array; dp[p + L] is relaxed when
 *                           @p nxt < dp[p + L].
 * @param[in,out] parent_len Backtrack length array, written in lockstep
 *                           with @p dp; receives the L of the relaxing
 *                           transition.
 * @param[in,out] parent_off Backtrack offset array, written in lockstep;
 *                           receives @p off_biased on relaxation.
 * @param[in]     p          Source DP position the transitions originate
 *                           from. Indexing into the three arrays is
 *                           `p + L`.
 * @param[in]     L          Initial L value (start of the span, inclusive).
 * @param[in]     L_end      End of the span (exclusive). Must satisfy
 *                           @p L_end <= UINT16_MAX so every written length
 *                           fits in @c parent_len's @c uint16_t cells.
 * @param[in]     nxt        Constant successor cost `dp[p] + transition`,
 *                           shared across the [L, L_end) span.
 * @param[in]     off_biased Match offset minus ::ZXC_LZ_OFFSET_BIAS, the
 *                           value stored when a transition wins.
 * @return The first L value not processed (i.e., @p L_end on success).
 */
// codeql[cpp/unused-static-function]: false positive
static ZXC_ALWAYS_INLINE size_t zxc_opt_dp_update_const_cost(
    uint32_t* RESTRICT dp, uint16_t* RESTRICT parent_len, uint16_t* RESTRICT parent_off,
    const size_t p, size_t L, const size_t L_end, const uint32_t nxt, const uint16_t off_biased) {
#if defined(ZXC_USE_AVX512) && defined(__AVX512VL__)
    if (L + 16 <= L_end) {
        const __m512i v_inc =
            _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const __m512i v_nxt = _mm512_set1_epi32((int)nxt);
        const __m256i v_off = _mm256_set1_epi16((int16_t)off_biased);
        for (; L + 16 <= L_end; L += 16) {
            const __m512i v_L_lanes = _mm512_add_epi32(v_inc, _mm512_set1_epi32((int)L));
            const __m512i v_dp = _mm512_loadu_si512((const void*)&dp[p + L]);
            const __mmask16 m = _mm512_cmplt_epu32_mask(v_nxt, v_dp);
            _mm512_mask_storeu_epi32(&dp[p + L], m, v_nxt);
            const __m256i v_L_u16 = _mm512_cvtusepi32_epi16(v_L_lanes);
            _mm256_mask_storeu_epi16((void*)&parent_len[p + L], m, v_L_u16);
            _mm256_mask_storeu_epi16((void*)&parent_off[p + L], m, v_off);
        }
    }
#elif defined(ZXC_USE_AVX2)
    if (L + 8 <= L_end) {
        const __m256i v_inc = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i v_nxt = _mm256_set1_epi32((int)nxt);
        const __m256i v_bias = _mm256_set1_epi32((int)0x80000000);
        const __m256i v_nxt_b = _mm256_xor_si256(v_nxt, v_bias);
        const __m128i v_off = _mm_set1_epi16((int16_t)off_biased);
        for (; L + 8 <= L_end; L += 8) {
            const __m256i v_L_lanes = _mm256_add_epi32(v_inc, _mm256_set1_epi32((int)L));
            const __m256i v_dp = _mm256_loadu_si256((const __m256i*)&dp[p + L]);
            /* Unsigned-compare-via-bias trick:
             *   (dp ^ 0x80000000) > (nxt ^ 0x80000000)  iff  dp > nxt
             * because XOR with the sign bit maps unsigned ordering to
             * signed ordering. AVX2 only has signed cmpgt for 32-bit. */
            const __m256i v_dp_b = _mm256_xor_si256(v_dp, v_bias);
            const __m256i v_mask = _mm256_cmpgt_epi32(v_dp_b, v_nxt_b);
            const __m256i v_dp_new = _mm256_blendv_epi8(v_dp, v_nxt, v_mask);
            _mm256_storeu_si256((__m256i*)&dp[p + L], v_dp_new);
            /* Pack 8x int32 mask -> 8x int16 mask with signed saturation:
             * 0xFFFFFFFF -> 0xFFFF, 0x00000000 -> 0x0000. */
            const __m128i v_mask16 = _mm_packs_epi32(_mm256_castsi256_si128(v_mask),
                                                     _mm256_extracti128_si256(v_mask, 1));
            const __m128i v_L_u16 = _mm_packus_epi32(_mm256_castsi256_si128(v_L_lanes),
                                                     _mm256_extracti128_si256(v_L_lanes, 1));
            const __m128i v_pl = _mm_loadu_si128((const __m128i*)&parent_len[p + L]);
            const __m128i v_pl_new = _mm_blendv_epi8(v_pl, v_L_u16, v_mask16);
            _mm_storeu_si128((__m128i*)&parent_len[p + L], v_pl_new);
            const __m128i v_po = _mm_loadu_si128((const __m128i*)&parent_off[p + L]);
            const __m128i v_po_new = _mm_blendv_epi8(v_po, v_off, v_mask16);
            _mm_storeu_si128((__m128i*)&parent_off[p + L], v_po_new);
        }
    }
#elif defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32)
    if (L + 4 <= L_end) {
        static const uint32_t k_inc_array[4] = {0, 1, 2, 3};
        const uint32x4_t v_inc = vld1q_u32(k_inc_array);
        const uint32x4_t v_nxt = vdupq_n_u32(nxt);
        const uint16x4_t v_off = vdup_n_u16(off_biased);
        for (; L + 4 <= L_end; L += 4) {
            const uint32x4_t v_L_lanes = vaddq_u32(v_inc, vdupq_n_u32((uint32_t)L));
            const uint32x4_t v_dp = vld1q_u32(&dp[p + L]);
            const uint32x4_t v_mask = vcgtq_u32(v_dp, v_nxt);
            vst1q_u32(&dp[p + L], vbslq_u32(v_mask, v_nxt, v_dp));
            const uint16x4_t v_mask16 = vmovn_u32(v_mask);
            const uint16x4_t v_L_u16 = vqmovn_u32(v_L_lanes);
            const uint16x4_t v_pl = vld1_u16(&parent_len[p + L]);
            vst1_u16(&parent_len[p + L], vbsl_u16(v_mask16, v_L_u16, v_pl));
            const uint16x4_t v_po = vld1_u16(&parent_off[p + L]);
            vst1_u16(&parent_off[p + L], vbsl_u16(v_mask16, v_off, v_po));
        }
    }
#elif defined(ZXC_USE_SSE2)
    if (L + 4 <= L_end) {
        const __m128i v_inc = _mm_setr_epi32(0, 1, 2, 3);
        const __m128i v_nxt = _mm_set1_epi32((int)nxt);
        const __m128i v_bias = _mm_set1_epi32((int)0x80000000);
        const __m128i v_nxt_b = _mm_xor_si128(v_nxt, v_bias);
        const __m128i v_off = _mm_set1_epi16((short)off_biased);
        for (; L + 4 <= L_end; L += 4) {
            const __m128i v_L_lanes = _mm_add_epi32(v_inc, _mm_set1_epi32((int)L));
            const __m128i v_dp = _mm_loadu_si128((const __m128i*)&dp[p + L]);
            /* Unsigned compare via sign-bit bias (SSE2 cmpgt is signed only):
             *   (dp ^ 0x80000000) > (nxt ^ 0x80000000)  iff  dp > nxt. */
            const __m128i v_dp_b = _mm_xor_si128(v_dp, v_bias);
            const __m128i v_mask = _mm_cmpgt_epi32(v_dp_b, v_nxt_b);
            const __m128i v_dp_new = zxc_mm_blendv_epi8_sse2(v_dp, v_nxt, v_mask);
            _mm_storeu_si128((__m128i*)&dp[p + L], v_dp_new);
            /* Narrow the 4x int32 mask / length lanes to 4x int16 (low 64 bits).
             * packs: 0xFFFFFFFF -> 0xFFFF, 0 -> 0; packus (SSE2-emulated): u32->u16. */
            const __m128i v_mask16 = _mm_packs_epi32(v_mask, v_mask);
            const __m128i v_L_u16 = zxc_mm_packus_epi32_sse2(v_L_lanes, v_L_lanes);
            __m128i v_pl = _mm_loadl_epi64((const __m128i*)&parent_len[p + L]);
            v_pl = zxc_mm_blendv_epi8_sse2(v_pl, v_L_u16, v_mask16);
            _mm_storel_epi64((__m128i*)&parent_len[p + L], v_pl);
            __m128i v_po = _mm_loadl_epi64((const __m128i*)&parent_off[p + L]);
            v_po = zxc_mm_blendv_epi8_sse2(v_po, v_off, v_mask16);
            _mm_storel_epi64((__m128i*)&parent_off[p + L], v_po);
        }
    }
#endif
    /* Scalar tail (and full path on archs without SIMD).
     * L < L_end <= UINT16_MAX (caller precondition), so the cast is lossless. */
    for (; L < L_end; L++) {
        if (nxt < dp[p + L]) {
            dp[p + L] = nxt;
            parent_len[p + L] = (uint16_t)L;
            parent_off[p + L] = off_biased;
        }
    }
    return L;
}

/**
 * @brief Estimate per-block literal cost from a sampled histogram passed
 *        through the actual length-limited Huffman builder.
 *
 * Strategy: build a strided sample of @p src (4096 entries), run the same
 * length-limited Huffman code construction the encoder uses, and report the
 * sample-weighted average code length. This is the predicted bits/byte
 * for Huffman-encoded literals on this distribution: no calibration
 * constants, no per-corpus tuning. The cap at 8 reflects that RAW is
 * always available at exactly that cost; if Huffman doesn't beat 8 on the
 * sample, the encoder will pick RAW and 8 is the right price.
 *
 * @param[in] src     Source buffer for the block.
 * @param[in] src_sz  Length of @p src in bytes.
 * @param[in] scratch Package-merge scratch (pre-allocated in the cctx for
 *                    level >= 6). May be `NULL`, in which case the builder
 *                    allocates its own working memory.
 * @return Estimated literal cost in bits, in `[1, 8]`.
 */
// codeql[cpp/unused-static-function]: false positive
static uint32_t zxc_opt_estimate_lit_bits(const uint8_t* RESTRICT src, const size_t src_sz,
                                          void* RESTRICT scratch) {
    if (UNLIKELY(src_sz < ZXC_OPT_LIT_SAMPLE_MIN)) return CHAR_BIT;

    uint32_t hist[ZXC_HUF_NUM_SYMBOLS] = {0};
    const size_t step = (src_sz > 4096) ? (src_sz >> 12) : 1U;
    size_t sampled = 0;
    for (size_t i = 0; i < src_sz; i += step) {
        hist[src[i]]++;
        sampled++;
    }

    uint8_t code_len[ZXC_HUF_NUM_SYMBOLS];
    if (UNLIKELY(zxc_huf_build_code_lengths(hist, code_len, scratch) != ZXC_OK)) return CHAR_BIT;

    /* Sample-weighted sum of code lengths == predicted total Huffman bits
     * for the sample. Divide by sample count for bits/byte, rounded up
     * (DP works in integer bits; rounding up errs on the conservative
     * side, slightly favoring matches over fractional-cost literals). */
    uint64_t total_bits = 0;
    for (int k = 0; k < ZXC_HUF_NUM_SYMBOLS; k++) {
        total_bits += (uint64_t)hist[k] * (uint64_t)code_len[k];
    }
    const uint32_t avg = (uint32_t)((total_bits + sampled - 1) / sampled);

    /* Cap at RAW cost: if Huffman can't beat 8 bits/byte on the sample,
     * the encoder will pick RAW anyway and 8 is the actual literal cost. */
    return (avg < CHAR_BIT) ? avg : CHAR_BIT;
}

/**
 * @brief Static price-based optimal LZ77 parser for level 6.
 *
 * Forward DP over the block's positions: `dp[p]` = min bit cost to encode
 * `src[0..p)`. Per-position transitions are
 *   - literal: `dp[p+1] = min(dp[p+1], dp[p] + lit_cost`
 *   - match  : `dp[p+L] = min(dp[p+L], dp[p] + match_cost(L))` for L in
 *              `[MIN_MATCH, max_L]`
 * where `max_L` is the longest match found by ::zxc_lz77_find_best_match at
 * `p` (with lazy disabled, the DP itself handles position-based
 * optimization). Backtracking from `dp[src_sz]` reconstructs the
 * optimal token sequence.
 *
 * Complexity guard: ::ZXC_OPT_LONG_MATCH_SKIP causes ::zxc_lz77_find_best_match
 * to be skipped at positions strictly inside a long match, without this
 * guard, highly repetitive data (e.g. Lorem-loop with multi-MB matches at
 * every offset) makes the parser quadratic and unit tests run for minutes.
 * The inner sub-length update loop visits every L from `MIN_MATCH` to
 * `max_L`; the skip threshold means each long-match region only pays its
 * O(L) cost once at the starting position, keeping total work O(N).
 *
 * @param[in,out] ctx           Compression context. The lazy-allocated
 *                              `opt_scratch` field provides the DP arrays;
 *                              it is grown on first use and reused on
 *                              subsequent blocks.
 * @param[in]  src              Source buffer to parse.
 * @param[in]  src_sz           Length of @p src in bytes.
 * @param[in,out] hash_table    LZ77 hash table (epoch | position entries).
 * @param[in,out] hash_tags     8-bit fast-rejection tags paired with @p hash_table.
 * @param[in,out] chain_table   Hash-chain link table (ring buffer).
 * @param[in]  epoch_mark       Current epoch shifted into the high bits.
 * @param[in]  offset_mask      Mask isolating the position bits in chain entries.
 * @param[in]  level            Compression level (used to size the matcher).
 * @param[out] literals         Buffer receiving the gathered literal bytes.
 * @param[out] buf_tokens       Buffer receiving the per-sequence token bytes.
 * @param[out] buf_offsets      Buffer receiving the per-sequence offsets.
 * @param[out] buf_extras       Buffer receiving variable-length overflow data.
 * @param[out] seq_c_out        Number of emitted sequences.
 * @param[out] lit_c_out        Number of literal bytes written into @p literals.
 * @param[out] extras_sz_out    Number of bytes written into @p buf_extras.
 * @param[out] max_offset_out   Largest biased offset emitted (used by the caller
 *                              to choose 1-byte vs 2-byte offset encoding).
 * @return @c ZXC_OK on success, or a negative @ref zxc_error_t.
 */
static int zxc_lz77_optimal_parse_glo(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint32_t* RESTRICT hash_table,
                                      uint8_t* RESTRICT hash_tags, uint16_t* RESTRICT chain_table,
                                      const uint32_t epoch_mark, const uint32_t offset_mask,
                                      const int level, uint8_t* RESTRICT literals,
                                      uint8_t* RESTRICT buf_tokens, uint16_t* RESTRICT buf_offsets,
                                      uint8_t* RESTRICT buf_extras, uint32_t* RESTRICT seq_c_out,
                                      size_t* RESTRICT lit_c_out, size_t* RESTRICT extras_sz_out,
                                      uint16_t* RESTRICT max_offset_out) {
    zxc_lz77_params_t lzp_opt = zxc_get_lz77_params(level);
    lzp_opt.use_lazy = 0;  // guard

    /* When a dictionary is active, src = [dict | block_data]. DP arrays are
     * indexed relative to the block start (position dict_sz in src). The
     * variable src_base points to the first block byte for literal copies,
     * while src remains the base for the match finder (absolute positions). */
    const size_t dict_sz = ctx->dict_size;
    const size_t block_sz = src_sz - dict_sz;
    const uint8_t* const src_base = src + dict_sz;
    const uint8_t* const iend = src + src_sz;

    /* Block too small for any match: emit all as literals. */
    if (UNLIKELY(block_sz < ZXC_LZ_SEARCH_MARGIN + 1)) {
        if (block_sz > 0) ZXC_MEMCPY(literals, src_base, block_sz);
        *lit_c_out = block_sz;
        *seq_c_out = 0;
        *extras_sz_out = 0;
        *max_offset_out = 0;
        return 0;
    }

    const size_t search_limit_pos = block_sz - ZXC_LZ_SEARCH_MARGIN;
    const uint8_t* const search_limit = src + search_limit_pos;

    /* DP arrays carved from ctx->opt_scratch: a single allocation lazy-
     * grown on the first level-6 call and reused across blocks. Each
     * sub-buffer is cache-line padded so the next one starts on a 64 B
     * boundary. The total `needed` matches zxc_estimate_cctx_size() keep
     * the formula in sync.
     *
     *   dp             : (chunk+1) x uint32_t: min cost to reach position p.
     *   parent_len     : (chunk+1) x uint16_t: 0 = literal, >= MIN_MATCH = match.
     *   parent_off     : (chunk+1) x uint16_t: biased match offset (distance-1).
     *   match_end_bits : ceil((chunk+1)/64) x uint64_t: 1 bit per position,
     *                                                  set when that position
     *                                                  is the end of a match
     *                                                  on the chosen DP path.
     *                                                  Replaces a forward-order
     *                                                  actions[] stack at 1/64
     *                                                  the cost.
     *
     * The same buffer is reused as transient scratch for the length-limited
     * Huffman code-length builder (see zxc_opt_estimate_lit_bits below and
     * the Huffman selection in zxc_encode_block_glo): the package-merge
     * scratch is needed before the DP runs and again after the parse has
     * been read out, so the lifetimes never overlap. The capacity is the
     * larger of the two demands. */
    const size_t chunk = ctx->chunk_size;
    const size_t sz_dp = ZXC_ALIGN_CL((chunk + 1) * sizeof(uint32_t));
    const size_t sz_pl = ZXC_ALIGN_CL((chunk + 1) * sizeof(uint16_t));
    const size_t sz_po = ZXC_ALIGN_CL((chunk + 1) * sizeof(uint16_t));
    const size_t n_bm_words = ZXC_BITMAP_WORDS(chunk + 1);
    const size_t sz_bm = ZXC_ALIGN_CL(n_bm_words * sizeof(uint64_t));
    const size_t dp_needed = sz_dp + sz_pl + sz_po + sz_bm;
    const size_t needed =
        (dp_needed > ZXC_HUF_BUILD_SCRATCH_SIZE) ? dp_needed : ZXC_HUF_BUILD_SCRATCH_SIZE;

    /* opt_scratch is now pre-allocated inside ctx->memory_block by
     * zxc_cctx_init when level >= ZXC_LEVEL_DENSITY. The formula above must
     * stay byte-for-byte in sync with the one in zxc_cctx_init() and
     * zxc_estimate_cctx_size(). */
    (void)needed;

    /* Per-block literal cost (sample only block data, not dict prefix): */
    const uint32_t lit_cost = zxc_opt_estimate_lit_bits(src_base, block_sz, ctx->opt_scratch);

    uint32_t* const dp = (uint32_t*)ctx->opt_scratch;
    uint16_t* const parent_len = (uint16_t*)(ctx->opt_scratch + sz_dp);
    uint16_t* const parent_off = (uint16_t*)(ctx->opt_scratch + sz_dp + sz_pl);
    uint64_t* const match_end_bits = (uint64_t*)(ctx->opt_scratch + sz_dp + sz_pl + sz_po);

    dp[0] = 0;
    ZXC_MEMSET(dp + 1, 0xFF, block_sz * sizeof(uint32_t));
    ZXC_MEMSET(parent_len, 0, sz_pl + sz_po + sz_bm);

    /* Forward DP: visit every position, update reachable successors.
     * `skip_until` skips find_best_match at positions strictly inside the
     * last long match, the DP transition from the start of the match
     * already covers dp[p+1..p+L], and re-searching at every intra-match
     * position is what makes the parser quadratic on repetitive inputs. */
    size_t skip_until = 0;
    /* Rolling repeat-offset seed for find_best_match */
    uint32_t last_off = 0;
    for (size_t p = 0; p < search_limit_pos; p++) {
        if (UNLIKELY(dp[p] == UINT32_MAX)) continue;

        /* Literal transition. */
        const uint32_t lit_next = dp[p] + lit_cost;
        if (lit_next < dp[p + 1]) {
            dp[p + 1] = lit_next;
            parent_len[p + 1] = 0;
        }

        if (p < skip_until) continue;

        /* Match transition: call find_best_match (no lazy, no backtrack via
         * anchor=ip). Iterate sub-lengths since any L <= max_L matches at the
         * same offset and may end at a more useful DP position.
         * ip uses absolute position (src + dict_sz + p) so match finder
         * resolves dict references correctly via src as base. */
        const uint8_t* ip = src_base + p;
        const zxc_match_t m = zxc_lz77_find_best_match(
            src, ip, iend, search_limit, /*anchor=*/ip, hash_table, hash_tags, chain_table,
            epoch_mark, offset_mask, level, lzp_opt, last_off);

        if (m.ref) {
            const uint32_t off = (uint32_t)(ip - m.ref);
            if (off > 0 && off <= ZXC_LZ_WINDOW_SIZE) {
                last_off = off;
                const size_t L_max_raw = (m.len > block_sz - p) ? (block_sz - p) : (size_t)m.len;
                const size_t L_max = (L_max_raw > UINT16_MAX) ? UINT16_MAX : L_max_raw;

                /* The L-iteration cost function is piecewise constant in
                 * varint segments. Split the [MIN_MATCH, L_max] span into:
                 *   1. cheap   : v < ML_MASK            -> cost = base
                 *   2. varint1 : v in [ML_MASK, ML_MASK + 128) -> cost = base + 8
                 *   3. varint2+: v >= ML_MASK + 128     -> cost = base + 16, +24, ...
                 *
                 * Steps 1 and 2 use constant nxt and are vectorized via
                 * the helper. Step 3 is rare (typical matches are short)
                 * and stays scalar. */
                const uint16_t off_biased = (uint16_t)(off - ZXC_LZ_OFFSET_BIAS);
                const size_t L_max_plus = L_max + 1;
                size_t L = ZXC_LZ_MIN_MATCH_LEN;

                /* 1. Cheap range. */
                {
                    const size_t L_cheap_end = ZXC_LZ_MIN_MATCH_LEN + ZXC_TOKEN_ML_MASK;
                    const size_t L_end = (L_max_plus < L_cheap_end) ? L_max_plus : L_cheap_end;
                    const uint32_t nxt = dp[p] + ZXC_OPT_MATCH_COST_BASE;
                    L = zxc_opt_dp_update_const_cost(dp, parent_len, parent_off, p, L, L_end, nxt,
                                                     off_biased);
                }

                /* 2. First varint level (1-byte extension). */
                if (L < L_max_plus) {
                    const size_t L_v1_end = ZXC_LZ_MIN_MATCH_LEN + ZXC_TOKEN_ML_MASK + 128;
                    const size_t L_end = (L_max_plus < L_v1_end) ? L_max_plus : L_v1_end;
                    const uint32_t nxt = dp[p] + ZXC_OPT_MATCH_COST_BASE + CHAR_BIT;
                    L = zxc_opt_dp_update_const_cost(dp, parent_len, parent_off, p, L, L_end, nxt,
                                                     off_biased);
                }

                /* 3. Higher varint levels: variable cost, kept scalar.
                 * Reached only by L >= ML_MASK + 128 + MIN_MATCH, so the
                 * v >= ML_MASK guard from the original loop is implied. */
                for (; L < L_max_plus; L++) {
                    uint32_t cost = ZXC_OPT_MATCH_COST_BASE;
                    uint32_t v = (uint32_t)(L - ZXC_LZ_MIN_MATCH_LEN) - ZXC_TOKEN_ML_MASK;
                    cost += CHAR_BIT;
                    while (v >= 128) {
                        v >>= 7;
                        cost += CHAR_BIT;
                    }
                    const uint32_t nxt = dp[p] + cost;
                    if (nxt < dp[p + L]) {
                        dp[p + L] = nxt;
                        parent_len[p + L] = (uint16_t)L;
                        parent_off[p + L] = off_biased;
                    }
                }
                if (UNLIKELY(L_max >= ZXC_OPT_LONG_MATCH_SKIP)) skip_until = p + L_max - 1;
            }
        }
    }

    /* Tail (last ZXC_LZ_SEARCH_MARGIN bytes) can only be literals: the match finder
     * stops at search_limit so its 8-byte probe reads stay in bounds. */
    for (size_t p = search_limit_pos; p < block_sz; p++) {
        if (UNLIKELY(dp[p] == UINT32_MAX)) continue;
        const uint32_t lit_next = dp[p] + lit_cost;
        if (lit_next < dp[p + 1]) {
            dp[p + 1] = lit_next;
            parent_len[p + 1] = 0;
        }
    }

    /* Backtrack from src_sz to 0: only match endpoints are recorded (one bit
     * per position in match_end_bits). Literals between matches are implicit
     * runs of unmarked positions and are reconstructed during forward emission
     * via lit_start tracking, so they need no backtrack storage. */
    {
        size_t pos = block_sz;
        while (pos > 0) {
            const uint32_t L = parent_len[pos];
            if (L == 0) {
                pos -= 1;
            } else {
                match_end_bits[pos >> 6] |= (uint64_t)1 << (pos & 63);
                pos -= L;
            }
        }
    }

    /* Forward emission: walk match_end_bits word-by-word, peeling set bits
     * with ctzll. Each set bit gives a match endpoint; parent_len/parent_off
     * at that position recover (length, offset). */
    uint32_t seq_c = 0;
    size_t lit_c = 0;
    size_t extras_sz = 0;
    uint16_t max_offset = 0;
    size_t lit_start = 0;

    for (size_t word_idx = 0; word_idx < n_bm_words; word_idx++) {
        uint64_t w = match_end_bits[word_idx];
        while (w) {
            const size_t pos = (word_idx << 6) + (size_t)zxc_ctz64(w);
            w &= w - 1;
            const uint32_t L = parent_len[pos];
            const uint16_t off_biased = parent_off[pos];
            const size_t match_start = pos - L;

            const size_t LL = match_start - lit_start;
            if (LL > 0) {
                ZXC_MEMCPY(literals + lit_c, src_base + lit_start, LL);
                lit_c += LL;
            }
            const uint32_t ll = (uint32_t)LL;
            const uint32_t ml = L - ZXC_LZ_MIN_MATCH_LEN;
            const uint8_t ll_code = (ll >= ZXC_TOKEN_LL_MASK) ? ZXC_TOKEN_LL_MASK : (uint8_t)ll;
            const uint8_t ml_code = (ml >= ZXC_TOKEN_ML_MASK) ? ZXC_TOKEN_ML_MASK : (uint8_t)ml;
            buf_tokens[seq_c] = (ll_code << ZXC_TOKEN_LIT_BITS) | ml_code;
            buf_offsets[seq_c] = off_biased;
            if (off_biased > max_offset) max_offset = off_biased;

            if (UNLIKELY(ll >= ZXC_TOKEN_LL_MASK)) {
                const size_t n = zxc_write_varint(buf_extras + extras_sz, ll - ZXC_TOKEN_LL_MASK);
                if (UNLIKELY(n == 0)) return ZXC_ERROR_OVERFLOW;
                extras_sz += n;
            }
            if (UNLIKELY(ml >= ZXC_TOKEN_ML_MASK)) {
                const size_t n = zxc_write_varint(buf_extras + extras_sz, ml - ZXC_TOKEN_ML_MASK);
                if (UNLIKELY(n == 0)) return ZXC_ERROR_OVERFLOW;
                extras_sz += n;
            }

            seq_c++;
            lit_start = pos;
        }
    }

    /* Tail literals after the last match (or all literals if no match). */
    if (lit_start < block_sz) {
        const size_t tail = block_sz - lit_start;
        ZXC_MEMCPY(literals + lit_c, src_base + lit_start, tail);
        lit_c += tail;
    }

    *seq_c_out = seq_c;
    *lit_c_out = lit_c;
    *extras_sz_out = extras_sz;
    *max_offset_out = max_offset;
    return 0;
}

/**
 * @brief Seeds the hash/chain tables from dictionary content prepended to @p src.
 *
 * When a dictionary is active, @p src is laid out as [dict_content | block_data].
 * This function inserts hash entries for dictionary positions [0, dict_size) so
 * the match finder can reference them during block encoding.
 *
 * @param[in]     src         Source buffer starting with dictionary content.
 * @param[in]     dict_size   Size of the dictionary prefix in bytes.
 * @param[in,out] hash_table  Hash table to seed with dictionary positions.
 * @param[in,out] hash_tags   Tag table for fast match rejection.
 * @param[in,out] chain_table Chain table for collision resolution.
 * @param[in]     epoch_mark  Current epoch marker for hash table entries.
 * @param[in]     offset_mask Position mask for epoch/offset encoding.
 * @param[in]     level       Compression level (controls hash function variant).
 */
static void zxc_lz_seed_dict(const uint8_t* RESTRICT src, const size_t dict_size,
                             uint32_t* RESTRICT hash_table, uint8_t* RESTRICT hash_tags,
                             uint16_t* RESTRICT chain_table, const uint32_t epoch_mark,
                             const uint32_t offset_mask, const int level) {
    if (UNLIKELY(dict_size < ZXC_LZ_MIN_MATCH_LEN)) return;

    const int use_hash5 = (level >= 3);
    const size_t limit = dict_size - (ZXC_LZ_MIN_MATCH_LEN - 1);

    /* Sparse seeding for the first half, dense for the second half.
     * Positions near the end of the dict produce shorter offsets and are
     * more likely to yield matches, so they deserve full coverage. */
    const size_t half = limit / 2;
    for (size_t i = 0; i < half; i += 4) {
        const uint64_t val8 = zxc_le64(src + i);
        const uint32_t h = zxc_hash_func(val8, use_hash5);
        const uint32_t cur_pos = (uint32_t)i;
        const uint8_t tag = (uint8_t)((uint32_t)val8 ^ ((uint32_t)val8 >> 16));

        hash_table[h] = epoch_mark | cur_pos;
        hash_tags[h] = tag;
        chain_table[cur_pos & ZXC_LZ_WINDOW_MASK] = 0;
    }
    for (size_t i = half; i < limit; i++) {
        const uint64_t val8 = zxc_le64(src + i);
        const uint32_t h = zxc_hash_func(val8, use_hash5);
        const uint32_t cur_pos = (uint32_t)i;
        const uint8_t tag = (uint8_t)((uint32_t)val8 ^ ((uint32_t)val8 >> 16));

        const uint32_t raw_head = hash_table[h];
        const uint32_t prev_idx =
            ((raw_head & ~offset_mask) == epoch_mark) ? (raw_head & offset_mask) : 0;

        hash_table[h] = epoch_mark | cur_pos;
        hash_tags[h] = tag;

        const uint32_t dist = cur_pos - prev_idx;
        const uint32_t valid = -((int32_t)((prev_idx != 0) & (dist < ZXC_LZ_WINDOW_SIZE)));
        chain_table[cur_pos & ZXC_LZ_WINDOW_MASK] = (uint16_t)(dist & valid);
    }
}

/**
 * @brief Encodes a data block using the General (GLO) compression format.
 *
 * This function implements the core LZ77 compression logic. It dynamically
 * adjusts compression parameters (search depth, lazy matching strategy, and
 * step skipping) based on the compression level configured in the context.
 *
 * **LZ77 Implementation Details:**
 * 1. **Hash Chain:** Uses a hash table (`ctx->hash_table`) to find potential
 * match positions. Collisions are handled via a `chain_table`, allowing us to
 * search deeper into the history for a better match.
 * 2. **Lazy Matching:** If a match is found, we check the *next* byte to see if
 *    it produces a longer match. If so, we output a literal and take the better
 * match. This is enabled for levels >= 3.
 * 3. **Step Skipping:** For lower levels (1-3), we skip bytes when updating the
 *    hash table to increase speed (`step > 1`). For levels 4+, we process every
 * byte to maximize compression ratio.
 * 4. **SIMD Match Finding:** Uses AVX2/AVX512/NEON to compare 32/64 bytes at a
 * time during match length calculation, significantly speeding up long match
 * verification.
 * 5. **RLE Detection:** Analyzes literals to see if Run-Length Encoding would
 * be beneficial (saving > 10% space).
 *
 * The encoding process consists of:
 * 1. **LZ77 Parsing**: The function iterates through the source data,
 * maintaining a hash chain to find repeated patterns (matches). It supports
 * "Lazy Matching" for higher compression levels to optimize match selection.
 * 2. **Sequence Storage**: Matches are converted into sequences consisting of
 *    literal lengths, match lengths, and offsets.
 * 3. **Bitpacking & Serialization**: The sequences are analyzed to determine
 * optimal bit-widths. The function then writes the block header, encodes
 * literals (using Raw or RLE encoding), and bit-packs the sequence streams into
 * the destination buffer.
 *
 * @param[in,out] ctx       Pointer to the compression context containing hash tables
 * and configuration.
 * @param[in] src       Pointer to the input source data.
 * @param[in] src_sz  Size of the input data in bytes.
 * @param[out] dst       Pointer to the destination buffer where compressed data will
 * be written.
 * @param[in] dst_cap   Maximum capacity of the destination buffer.
 * @param[out] out_sz    [Out] Pointer to a variable that will receive the total size
 * of the compressed output.
 *
 * @return ZXC_OK on success, or a negative zxc_error_t code (e.g., ZXC_ERROR_DST_TOO_SMALL) if an
 * error occurs (e.g., buffer overflow).
 */
static int zxc_encode_block_glo(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                const size_t src_sz, uint8_t* RESTRICT dst, size_t dst_cap,
                                size_t* RESTRICT out_sz) {
    const int level = ctx->compression_level;
    const size_t dict_sz = ctx->dict_size;

    const zxc_lz77_params_t lzp = zxc_get_lz77_params(level);

    ctx->epoch++;
    if (UNLIKELY(ctx->epoch >= ctx->max_epoch)) {
        ZXC_MEMSET(ctx->hash_table, 0, ZXC_LZ_HASH_SIZE * sizeof(uint32_t));
        ZXC_MEMSET(ctx->hash_tags, 0, ZXC_LZ_HASH_SIZE * sizeof(uint8_t));
        ctx->epoch = 1;
    }
    const uint32_t offset_bits = ctx->offset_bits;
    const uint32_t offset_mask = ctx->offset_mask;
    const uint32_t epoch_mark = ctx->epoch << offset_bits;

    if (dict_sz > 0)
        zxc_lz_seed_dict(src, dict_sz, ctx->hash_table, ctx->hash_tags, ctx->chain_table,
                         epoch_mark, offset_mask, level);

    const uint8_t* ip = src + dict_sz;
    const uint8_t* iend = src + src_sz;
    const uint8_t* anchor = ip;
    const uint8_t* search_limit = iend - ZXC_LZ_SEARCH_MARGIN;

    uint32_t* const hash_table = ctx->hash_table;
    uint8_t* const hash_tags = ctx->hash_tags;
    uint16_t* const chain_table = ctx->chain_table;
    uint8_t* const literals = ctx->literals;
    uint8_t* const buf_tokens = ctx->buf_tokens;
    uint16_t* const buf_offsets = ctx->buf_offsets;
    uint8_t* const buf_extras = ctx->buf_extras;

    uint32_t seq_c = 0;
    size_t lit_c = 0;
    size_t extras_sz = 0;
    uint16_t max_offset = 0;  // Track max offset for 1-byte/2-byte mode decision

    /* Level 6+: price-based optimal parser (fills outputs and skips the
     * lazy loop + last_lits handling below via `goto parse_done`). */
    if (level >= ZXC_LEVEL_DENSITY) {
        const int rc = zxc_lz77_optimal_parse_glo(
            ctx, src, src_sz, hash_table, hash_tags, chain_table, epoch_mark, offset_mask, level,
            literals, buf_tokens, buf_offsets, buf_extras, &seq_c, &lit_c, &extras_sz, &max_offset);
        if (UNLIKELY(rc != 0)) return rc;
        goto parse_done;
    }

    while (LIKELY(ip < search_limit)) {
        const size_t dist = (size_t)(ip - anchor);
        size_t step = lzp.step_base + (dist >> lzp.step_shift);
        if (UNLIKELY(ip + step >= search_limit)) step = 1;

        if (LIKELY(ip + step + sizeof(uint64_t) <= iend)) {
            const uint64_t v_next = zxc_le64(ip + step);
            // cppcheck-suppress unreadVariable
            const uint32_t h_next = zxc_hash_func(v_next, 1);
            ZXC_PREFETCH_READ(&hash_tags[h_next]);
            ZXC_PREFETCH_READ(&hash_table[h_next]);
        }

        const zxc_match_t m =
            zxc_lz77_find_best_match(src, ip, iend, search_limit, anchor, hash_table, hash_tags,
                                     chain_table, epoch_mark, offset_mask, level, lzp,
                                     /*last_off=*/0U);

        if (m.ref) {
            ip -= m.backtrack;
            const uint32_t ll = (uint32_t)(ip - anchor);
            const uint32_t ml = m.len - ZXC_LZ_MIN_MATCH_LEN;
            const uint32_t off = (uint32_t)(ip - m.ref);

            if (ll > 0) {
                if (LIKELY(anchor + ZXC_PAD_SIZE <= iend)) {
                    zxc_copy32(literals + lit_c, anchor);
                    if (UNLIKELY(ll > ZXC_PAD_SIZE)) {
                        ZXC_MEMCPY(literals + lit_c + ZXC_PAD_SIZE, anchor + ZXC_PAD_SIZE,
                                   ll - ZXC_PAD_SIZE);
                    }
                } else {
                    ZXC_MEMCPY(literals + lit_c, anchor, ll);
                }
                lit_c += ll;
            }

            const uint8_t ll_code = (ll >= ZXC_TOKEN_LL_MASK) ? ZXC_TOKEN_LL_MASK : (uint8_t)ll;
            const uint8_t ml_code = (ml >= ZXC_TOKEN_ML_MASK) ? ZXC_TOKEN_ML_MASK : (uint8_t)ml;
            buf_tokens[seq_c] = (ll_code << ZXC_TOKEN_LIT_BITS) | ml_code;
            buf_offsets[seq_c] = (uint16_t)(off - ZXC_LZ_OFFSET_BIAS);
            if ((off - ZXC_LZ_OFFSET_BIAS) > max_offset)
                max_offset = (uint16_t)(off - ZXC_LZ_OFFSET_BIAS);

            if (ll >= ZXC_TOKEN_LL_MASK) {
                const size_t n = zxc_write_varint(buf_extras + extras_sz, ll - ZXC_TOKEN_LL_MASK);
                if (UNLIKELY(n == 0)) return ZXC_ERROR_OVERFLOW;
                extras_sz += n;
            }
            if (ml >= ZXC_TOKEN_ML_MASK) {
                const size_t n = zxc_write_varint(buf_extras + extras_sz, ml - ZXC_TOKEN_ML_MASK);
                if (UNLIKELY(n == 0)) return ZXC_ERROR_OVERFLOW;
                extras_sz += n;
            }

            seq_c++;

            if (m.len > 2 && level > ZXC_LEVEL_BALANCED) {
                const uint8_t* match_end = ip + m.len;
                if (match_end < iend - 7) {
                    const uint32_t pos_u = (uint32_t)((match_end - 2) - src);
                    const uint64_t val_u8 = zxc_le64(match_end - 2);
                    const uint32_t val_u = (uint32_t)val_u8;
                    const uint32_t h_u = zxc_hash_func(val_u8, 1);
                    const uint32_t prev_head = hash_table[h_u];
                    const uint32_t prev_idx =
                        (prev_head & ~offset_mask) == epoch_mark ? (prev_head & offset_mask) : 0;
                    hash_table[h_u] = epoch_mark | pos_u;
                    hash_tags[h_u] = (uint8_t)(val_u ^ (val_u >> 16));
                    chain_table[pos_u & ZXC_LZ_WINDOW_MASK] =
                        (prev_idx > 0 && (pos_u - prev_idx) < ZXC_LZ_WINDOW_SIZE)
                            ? (uint16_t)(pos_u - prev_idx)
                            : 0;
                }
            }

            ip += m.len;
            anchor = ip;
        } else {
            ip += step;
        }
    }

    const size_t last_lits = iend - anchor;
    if (last_lits > 0) {
        ZXC_MEMCPY(literals + lit_c, anchor, last_lits);
        lit_c += last_lits;
    }

parse_done:;
    /* Dictionary-table trainer hook: accumulate the REAL post-LZ literal
     * frequencies (see zxc_train_dict_huf). Cold path, NULL outside training. */
    if (UNLIKELY(ctx->lit_freq_acc != NULL)) {
        for (size_t i = 0; i < lit_c; i++) ctx->lit_freq_acc[literals[i]]++;
    }

    // --- RLE ANALYSIS ---
    size_t rle_size = 0;
    int enc_lit = ZXC_SECTION_ENCODING_RAW;

    if (lit_c > 0) {
        const uint8_t* p = literals;
        const uint8_t* const p_end = literals + lit_c;
        const uint8_t* const p_end_4 = p_end - 3;  // Safe limit for 4-byte lookahead

        while (LIKELY(p < p_end)) {
            const uint8_t b = *p;
            const uint8_t* run_start = p++;

            // Fast run counting with early SIMD exit
#if defined(ZXC_USE_AVX512)
            const __m512i vb = _mm512_set1_epi8((char)b);
            while (p <= p_end - 64) {
                const __m512i v = _mm512_loadu_si512((const void*)p);
                const __mmask64 mask = _mm512_cmpeq_epi8_mask(v, vb);
                if (mask != 0xFFFFFFFFFFFFFFFFULL) {
                    p += (size_t)zxc_ctz64(~mask);
                    goto _run_done;
                }
                p += 64;
            }
#elif defined(ZXC_USE_AVX2)
            const __m256i vb = _mm256_set1_epi8((char)b);
            while (p <= p_end - 32) {
                const __m256i v = _mm256_loadu_si256((const __m256i*)p);
                const uint32_t mask = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vb));
                if (mask != 0xFFFFFFFF) {
                    p += zxc_ctz32(~mask);
                    goto _run_done;
                }
                p += 32;
            }
#elif defined(ZXC_USE_SSE2)
            const __m128i vb = _mm_set1_epi8((char)b);
            while (p <= p_end - 16) {
                const __m128i v = _mm_loadu_si128((const __m128i*)p);
                const uint32_t mask = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(v, vb));
                if (mask != 0xFFFFU) {
                    p += zxc_ctz32(~mask);
                    goto _run_done;
                }
                p += 16;
            }
#elif defined(ZXC_USE_NEON64)
            const uint8x16_t vb = vdupq_n_u8(b);
            while (p <= p_end - 16) {
                const uint8x16_t v = vld1q_u8(p);
                const uint8x16_t eq = vceqq_u8(v, vb);
                /* SHRN nibble-mask: see find_best_match above for rationale. */
                const uint64_t mask =
                    vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(eq), 4)), 0);
                if (LIKELY(mask == ~(uint64_t)0)) {
                    p += 16;
                } else {
                    p += (size_t)(zxc_ctz64(~mask) >> 2);
                    goto _run_done;
                }
            }
#elif defined(ZXC_USE_NEON32)
            uint8x16_t vb = vdupq_n_u8(b);
            while (p <= p_end - 16) {
                uint8x16_t v = vld1q_u8(p);
                uint8x16_t eq = vceqq_u8(v, vb);
                uint8x16_t not_eq = vmvnq_u8(eq);

                // 32-bit ARM NEON doesn't always support vgetq_lane_u64 / vreinterpretq_u64_u8 so
                // we treat the 128-bit vector as 4 x 32-bit lanes */
                const uint32x4_t neq32 = vreinterpretq_u32_u8(not_eq);
                const uint32_t l0 = vgetq_lane_u32(neq32, 0);
                const uint32_t l1 = vgetq_lane_u32(neq32, 1);

                const uint64_t lo = ((uint64_t)l1 << 32) | l0;
                if (lo != 0) {
                    p += (size_t)(zxc_ctz64(lo) >> 3);
                    goto _run_done;
                }

                const uint32_t h0 = vgetq_lane_u32(neq32, 2);
                const uint32_t h1 = vgetq_lane_u32(neq32, 3);
                const uint64_t hi = ((uint64_t)h1 << 32) | h0;

                if (hi != 0) {
                    p += 8 + (zxc_ctz64(hi) >> 3);
                    goto _run_done;
                }
                p += 16;
            }
#endif
            while (p < p_end && *p == b) p++;

#if defined(ZXC_USE_AVX512) || defined(ZXC_USE_AVX2) || defined(ZXC_USE_NEON64) || \
    defined(ZXC_USE_NEON32) || defined(ZXC_USE_SSE2)
        _run_done:;
#endif
            const size_t run = (size_t)(p - run_start);

            if (run >= 4) {
                // RLE run: 2 bytes per 131 values, then remainder
                // Branchless: full_chunks * 2 + remainder handling
                const size_t full_chunks = run / 131;
                const size_t rem = run - full_chunks * 131;  // Avoid modulo
                rle_size += full_chunks * 2;
                // Remainder: if >= 4 -> 2 bytes (RLE), else 1 + rem (literal)
                if (rem >= 4)
                    rle_size += 2;
                else if (rem > 0)
                    rle_size += 1 + rem;
            } else {
                // Literal run: scan ahead with fast SIMD lookahead
                const uint8_t* lit_start = run_start;

#if defined(ZXC_USE_AVX512)
                while (p <= p_end_4 - 64) {
                    const __m512i v0 = _mm512_loadu_si512((const void*)p);
                    const __m512i v1 = _mm512_loadu_si512((const void*)(p + 1));
                    const __m512i v2 = _mm512_loadu_si512((const void*)(p + 2));
                    const __m512i v3 = _mm512_loadu_si512((const void*)(p + 3));
                    const __mmask64 mask = _mm512_cmpeq_epi8_mask(v0, v1) &
                                           _mm512_cmpeq_epi8_mask(v1, v2) &
                                           _mm512_cmpeq_epi8_mask(v2, v3);
                    if (mask != 0) {
                        p += (size_t)zxc_ctz64(mask);
                        goto _lit_done;
                    }
                    p += 64;
                }
#elif defined(ZXC_USE_AVX2)
                while (p <= p_end_4 - 32) {
                    __m256i v0 = _mm256_loadu_si256((const __m256i*)p);
                    __m256i v1 = _mm256_loadu_si256((const __m256i*)(p + 1));
                    __m256i v2 = _mm256_loadu_si256((const __m256i*)(p + 2));
                    __m256i v3 = _mm256_loadu_si256((const __m256i*)(p + 3));
                    __m256i vend = _mm256_and_si256(
                        _mm256_cmpeq_epi8(v0, v1),
                        _mm256_and_si256(_mm256_cmpeq_epi8(v1, v2), _mm256_cmpeq_epi8(v2, v3)));
                    uint32_t mask = (uint32_t)_mm256_movemask_epi8(vend);
                    if (mask != 0) {
                        p += zxc_ctz32(mask);
                        goto _lit_done;
                    }
                    p += 32;
                }
#elif defined(ZXC_USE_SSE2)
                while (p <= p_end_4 - 16) {
                    __m128i v0 = _mm_loadu_si128((const __m128i*)p);
                    __m128i v1 = _mm_loadu_si128((const __m128i*)(p + 1));
                    __m128i v2 = _mm_loadu_si128((const __m128i*)(p + 2));
                    __m128i v3 = _mm_loadu_si128((const __m128i*)(p + 3));
                    __m128i vend = _mm_and_si128(
                        _mm_cmpeq_epi8(v0, v1),
                        _mm_and_si128(_mm_cmpeq_epi8(v1, v2), _mm_cmpeq_epi8(v2, v3)));
                    uint32_t mask = (uint32_t)_mm_movemask_epi8(vend);
                    if (mask != 0) {
                        p += zxc_ctz32(mask);
                        goto _lit_done;
                    }
                    p += 16;
                }
#elif defined(ZXC_USE_NEON64)
                while (p <= p_end_4 - 16) {
                    uint8x16_t v0 = vld1q_u8(p);
                    uint8x16_t v1 = vld1q_u8(p + 1);
                    uint8x16_t v2 = vld1q_u8(p + 2);
                    uint8x16_t v3 = vld1q_u8(p + 3);
                    uint8x16_t eq =
                        vandq_u8(vceqq_u8(v0, v1), vandq_u8(vceqq_u8(v1, v2), vceqq_u8(v2, v3)));
                    /* Dual of the run scan: searching for the FIRST set
                     * nibble (a position where 4 consecutive bytes match).
                     * mask == 0 means no break found in this 16-byte
                     * window. Same SHRN compression as elsewhere. */
                    const uint64_t mask = vget_lane_u64(
                        vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(eq), 4)), 0);
                    if (LIKELY(mask == 0)) {
                        p += 16;
                    } else {
                        p += (size_t)(zxc_ctz64(mask) >> 2);
                        goto _lit_done;
                    }
                }
#elif defined(ZXC_USE_NEON32)
                while (p <= p_end_4 - 16) {
                    uint8x16_t v0 = vld1q_u8(p);
                    uint8x16_t v1 = vld1q_u8(p + 1);
                    uint8x16_t v2 = vld1q_u8(p + 2);
                    uint8x16_t v3 = vld1q_u8(p + 3);
                    uint8x16_t eq =
                        vandq_u8(vceqq_u8(v0, v1), vandq_u8(vceqq_u8(v1, v2), vceqq_u8(v2, v3)));

                    uint32x4_t eq32 = vreinterpretq_u32_u8(eq);
                    uint32_t l0 = vgetq_lane_u32(eq32, 0);
                    uint32_t l1 = vgetq_lane_u32(eq32, 1);
                    uint64_t lo = ((uint64_t)l1 << 32) | l0;

                    if (lo != 0) {
                        p += (zxc_ctz64(lo) >> 3);
                        goto _lit_done;
                    }

                    uint32_t h0 = vgetq_lane_u32(eq32, 2);
                    uint32_t h1 = vgetq_lane_u32(eq32, 3);
                    uint64_t hi = ((uint64_t)h1 << 32) | h0;

                    if (hi != 0) {
                        p += 8 + (zxc_ctz64(hi) >> 3);
                        goto _lit_done;
                    }
                    p += 16;
                }
#endif
                while (p < p_end_4) {
                    // Check for RLE opportunity (4 identical bytes)
                    if (UNLIKELY(p[0] == p[1] && p[1] == p[2] && p[2] == p[3])) break;
                    p++;
                }
                // Handle remaining bytes near end
                while (p < p_end) {
                    if (UNLIKELY(p + 3 < p_end && p[0] == p[1] && p[1] == p[2] && p[2] == p[3]))
                        break;
                    p++;
                }

#if defined(ZXC_USE_AVX512) || defined(ZXC_USE_AVX2) || defined(ZXC_USE_NEON64) || \
    defined(ZXC_USE_NEON32) || defined(ZXC_USE_SSE2)
            _lit_done:;
#endif
                const size_t lit_run = (size_t)(p - lit_start);
                // 1 header per 128 bytes + all data bytes
                // lit_run + ceil(lit_run / 128)
                rle_size += lit_run + ((lit_run + 127) >> 7);
            }
        }

        // Threshold: ~3% savings using integer math (97% ~= 1 - 1/32)
        if (rle_size < lit_c - (lit_c >> 5)) enc_lit = ZXC_SECTION_ENCODING_RLE;
    }

    /* Level >= 6: also evaluate Huffman as a 3rd literal-encoding candidate.
     * Build a histogram and length-limited canonical code lengths, compute the
     * exact byte size of the 4-way interleaved bitstream + 134-byte header,
     * and switch to HUFFMAN if it beats the current choice by >= 3%. */
    uint8_t huf_code_len[ZXC_HUF_NUM_SYMBOLS];
    size_t huf_total_size = SIZE_MAX;
    if (level >= ZXC_LEVEL_DENSITY && lit_c >= ZXC_HUF_MIN_LITERALS) {
        uint32_t freq0[ZXC_HUF_NUM_SYMBOLS] = {0};
        uint32_t freq1[ZXC_HUF_NUM_SYMBOLS] = {0};
        uint32_t freq2[ZXC_HUF_NUM_SYMBOLS] = {0};
        uint32_t freq3[ZXC_HUF_NUM_SYMBOLS] = {0};
        {
            size_t i = 0;
            for (; i + 4 <= lit_c; i += 4) {
                freq0[literals[i + 0]]++;
                freq1[literals[i + 1]]++;
                freq2[literals[i + 2]]++;
                freq3[literals[i + 3]]++;
            }
            for (; i < lit_c; i++) freq0[literals[i]]++;
        }
        uint32_t freq[ZXC_HUF_NUM_SYMBOLS];
        for (int k = 0; k < ZXC_HUF_NUM_SYMBOLS; k++) {
            freq[k] = freq0[k] + freq1[k] + freq2[k] + freq3[k];
        }

        if (zxc_huf_build_code_lengths(freq, huf_code_len, ctx->opt_scratch) == ZXC_OK) {
            const size_t Q = (lit_c + ZXC_HUF_NUM_STREAMS - 1) / ZXC_HUF_NUM_STREAMS;
            size_t streams_bytes = 0;
            for (int s = 0; s < ZXC_HUF_NUM_STREAMS; s++) {
                size_t start = (size_t)s * Q;
                size_t stop = start + Q;
                if (start > lit_c) start = lit_c;
                if (stop > lit_c) stop = lit_c;
                uint64_t b0 = 0;
                uint64_t b1 = 0;
                uint64_t b2 = 0;
                uint64_t b3 = 0;
                size_t i = start;

                for (; i + 4 <= stop; i += 4) {
                    b0 += huf_code_len[literals[i + 0]];
                    b1 += huf_code_len[literals[i + 1]];
                    b2 += huf_code_len[literals[i + 2]];
                    b3 += huf_code_len[literals[i + 3]];
                }
                uint64_t bits = b0 + b1 + b2 + b3;
                for (; i < stop; i++) bits += huf_code_len[literals[i]];
                streams_bytes += (size_t)((bits + 7) / 8);
            }
            huf_total_size = ZXC_HUF_HEADER_SIZE + streams_bytes;
            const size_t baseline = (enc_lit == ZXC_SECTION_ENCODING_RLE) ? rle_size : lit_c;
            /* Threshold: 3% savings (1/32) over the chosen RAW/RLE baseline.
             * Same heuristic as the RAW/RLE switch above. */
            if (huf_total_size < baseline - (baseline >> 5)) {
                enc_lit = ZXC_SECTION_ENCODING_HUFFMAN;
            }
        }
    }

    /* Shared dictionary table candidate (level >= 6, dict with table attached):
     * same bitstream as HUFFMAN but no 128-byte lengths header, so it stays
     * viable on literal sections far below ZXC_HUF_MIN_LITERALS. Exact size
     * accounting; invalid (a literal byte without a code) drops the candidate. */
    size_t huf_dict_total_size = SIZE_MAX;
    uint8_t dict_code_len[ZXC_HUF_NUM_SYMBOLS];
    if (level >= ZXC_LEVEL_DENSITY && ctx->dict_huf_lengths != NULL && lit_c > 0 &&
        zxc_huf_unpack_lengths(ctx->dict_huf_lengths, dict_code_len) == ZXC_OK) {
        const size_t Q = (lit_c + ZXC_HUF_NUM_STREAMS - 1) / ZXC_HUF_NUM_STREAMS;
        size_t streams_bytes = 0;
        int valid = 1;
        for (int s = 0; s < ZXC_HUF_NUM_STREAMS && valid; s++) {
            size_t start = (size_t)s * Q;
            size_t stop = start + Q;
            if (start > lit_c) start = lit_c;
            if (stop > lit_c) stop = lit_c;
            uint64_t bits = 0;
            for (size_t i = start; i < stop; i++) {
                const int len = dict_code_len[literals[i]];
                if (UNLIKELY(len == 0)) {
                    valid = 0;
                    break;
                }
                bits += (uint64_t)len;
            }
            streams_bytes += (size_t)((bits + 7) / 8);
        }
        if (valid) {
            huf_dict_total_size = (size_t)ZXC_HUF_STREAM_SIZES_HEADER_SIZE + streams_bytes;
            if (enc_lit == ZXC_SECTION_ENCODING_HUFFMAN) {
                /* Both candidates are Huffman bitstreams: pick the smaller. */
                if (huf_dict_total_size < huf_total_size)
                    enc_lit = ZXC_SECTION_ENCODING_HUFFMAN_DICT;
            } else {
                const size_t baseline = (enc_lit == ZXC_SECTION_ENCODING_RLE) ? rle_size : lit_c;
                /* Same 3% (1/32) margin as the other encoding switches. */
                if (huf_dict_total_size < baseline - (baseline >> 5))
                    enc_lit = ZXC_SECTION_ENCODING_HUFFMAN_DICT;
            }
        }
    }

    zxc_block_header_t bh = {.block_type = ZXC_BLOCK_GLO};
    uint8_t* const p = dst + ZXC_BLOCK_HEADER_SIZE;
    size_t rem = dst_cap - ZXC_BLOCK_HEADER_SIZE;

    // Decide offset encoding mode: 1-byte if all offsets <= 255
    const int use_8bit_off = (max_offset <= 255) ? 1 : 0;
    const size_t off_stream_size = use_8bit_off ? seq_c : (seq_c * 2);

    const zxc_gnr_header_t gh = {.n_sequences = seq_c,
                                 .n_literals = (uint32_t)lit_c,
                                 .enc_lit = enc_lit,
                                 .enc_litlen = 0,
                                 .enc_mlen = 0,
                                 .enc_off = (uint8_t)use_8bit_off};

    zxc_section_desc_t desc[ZXC_GLO_SECTIONS] = {0};
    const size_t lit_section_size = (enc_lit == ZXC_SECTION_ENCODING_RLE)       ? rle_size
                                    : (enc_lit == ZXC_SECTION_ENCODING_HUFFMAN) ? huf_total_size
                                    : (enc_lit == ZXC_SECTION_ENCODING_HUFFMAN_DICT)
                                        ? huf_dict_total_size
                                        : lit_c;
    desc[0].sizes = (uint64_t)lit_section_size | ((uint64_t)lit_c << 32);
    desc[1].sizes = (uint64_t)seq_c | ((uint64_t)seq_c << 32);
    desc[2].sizes = (uint64_t)off_stream_size | ((uint64_t)off_stream_size << 32);
    desc[3].sizes = (uint64_t)extras_sz | ((uint64_t)extras_sz << 32);

    const int ghs = zxc_write_glo_header_and_desc(p, rem, &gh, desc);
    if (UNLIKELY(ghs < 0)) return ghs;

    uint8_t* p_curr = p + ghs;
    rem -= ghs;

    // Extract stream sizes once
    const size_t sz_lit = (size_t)(desc[0].sizes & ZXC_SECTION_SIZE_MASK);
    const size_t sz_tok = (size_t)(desc[1].sizes & ZXC_SECTION_SIZE_MASK);
    const size_t sz_off = (size_t)(desc[2].sizes & ZXC_SECTION_SIZE_MASK);
    const size_t sz_ext = (size_t)(desc[3].sizes & ZXC_SECTION_SIZE_MASK);

    if (UNLIKELY(rem < sz_lit)) return ZXC_ERROR_DST_TOO_SMALL;

    if (enc_lit == ZXC_SECTION_ENCODING_HUFFMAN) {
        const int written = zxc_huf_encode_section(literals, lit_c, huf_code_len, p_curr, rem);
        if (UNLIKELY(written < 0)) return written;
        if (UNLIKELY((size_t)written != huf_total_size)) return ZXC_ERROR_DST_TOO_SMALL;
        p_curr += written;
    } else if (enc_lit == ZXC_SECTION_ENCODING_HUFFMAN_DICT) {
        const int written =
            zxc_huf_encode_section_dict(literals, lit_c, dict_code_len, p_curr, rem);
        if (UNLIKELY(written < 0)) return written;
        if (UNLIKELY((size_t)written != huf_dict_total_size)) return ZXC_ERROR_DST_TOO_SMALL;
        p_curr += written;
    } else if (enc_lit == ZXC_SECTION_ENCODING_RLE) {
        // Write RLE - optimized single-pass encoding
        const uint8_t* lit_ptr = literals;
        const uint8_t* const lit_end = literals + lit_c;

        while (lit_ptr < lit_end) {
            uint8_t b = *lit_ptr;
            const uint8_t* run_start = lit_ptr++;

            // Count run length
            while (lit_ptr < lit_end && *lit_ptr == b) lit_ptr++;
            size_t run = (size_t)(lit_ptr - run_start);

            if (run >= 4) {
                // RLE runs: emit 2-byte tokens (header + value)
                while (run >= 4) {
                    size_t chunk = (run > 131) ? 131 : run;
                    *p_curr++ = (uint8_t)(ZXC_LIT_RLE_FLAG | (chunk - 4));
                    *p_curr++ = b;
                    run -= chunk;
                }
                // Leftover < 4 bytes: emit as literal
                if (run > 0) {
                    *p_curr++ = (uint8_t)(run - 1);
                    ZXC_MEMCPY(p_curr, lit_ptr - run, run);
                    p_curr += run;
                }
            } else {
                // Literal run: scan ahead to find next RLE opportunity
                const uint8_t* lit_run_start = run_start;

                while (lit_ptr < lit_end) {
                    // Quick check: need 4 identical bytes to break
                    if (UNLIKELY(lit_ptr + 3 < lit_end && lit_ptr[0] == lit_ptr[1] &&
                                 lit_ptr[1] == lit_ptr[2] && lit_ptr[2] == lit_ptr[3])) {
                        break;
                    }
                    lit_ptr++;
                }

                size_t lit_run = (size_t)(lit_ptr - lit_run_start);
                const uint8_t* src_ptr = lit_run_start;

                // Emit literal chunks (max 128 bytes each)
                while (lit_run > 0) {
                    size_t chunk = (lit_run > 128) ? 128 : lit_run;
                    *p_curr++ = (uint8_t)(chunk - 1);
                    ZXC_MEMCPY(p_curr, src_ptr, chunk);
                    p_curr += chunk;
                    src_ptr += chunk;
                    lit_run -= chunk;
                }
            }
        }
    } else {
        ZXC_MEMCPY(p_curr, literals, lit_c);
        p_curr += lit_c;
    }
    rem -= sz_lit;

    if (UNLIKELY(rem < sz_tok)) return ZXC_ERROR_DST_TOO_SMALL;

    ZXC_MEMCPY(p_curr, buf_tokens, seq_c);
    p_curr += seq_c;
    rem -= sz_tok;

    if (UNLIKELY(rem < sz_off)) return ZXC_ERROR_DST_TOO_SMALL;

    if (use_8bit_off) {
        // Write 1-byte offsets - unroll for better throughput
        uint32_t i = 0;
        for (; i + 8 <= seq_c; i += 8) {
            p_curr[0] = (uint8_t)buf_offsets[i + 0];
            p_curr[1] = (uint8_t)buf_offsets[i + 1];
            p_curr[2] = (uint8_t)buf_offsets[i + 2];
            p_curr[3] = (uint8_t)buf_offsets[i + 3];
            p_curr[4] = (uint8_t)buf_offsets[i + 4];
            p_curr[5] = (uint8_t)buf_offsets[i + 5];
            p_curr[6] = (uint8_t)buf_offsets[i + 6];
            p_curr[7] = (uint8_t)buf_offsets[i + 7];
            p_curr += 8;
        }
        for (; i < seq_c; i++) {
            *p_curr++ = (uint8_t)buf_offsets[i];
        }
    } else {
        // Write 2-byte offsets in little-endian order
#ifdef ZXC_BIG_ENDIAN
        for (uint32_t i = 0; i < seq_c; i++) {
            zxc_store_le16(p_curr, buf_offsets[i]);
            p_curr += sizeof(uint16_t);
        }
#else
        ZXC_MEMCPY(p_curr, buf_offsets, seq_c * sizeof(uint16_t));
        p_curr += seq_c * sizeof(uint16_t);
#endif
    }
    rem -= sz_off;

    if (UNLIKELY(rem < sz_ext)) return ZXC_ERROR_DST_TOO_SMALL;

    ZXC_MEMCPY(p_curr, buf_extras, extras_sz);
    p_curr += extras_sz;

    bh.comp_size = (uint32_t)(p_curr - (dst + ZXC_BLOCK_HEADER_SIZE));
    const int hw = zxc_write_block_header(dst, dst_cap, &bh);
    if (UNLIKELY(hw < 0)) return hw;

    // Checksum will be appended by the wrapper
    *out_sz = ZXC_BLOCK_HEADER_SIZE + bh.comp_size;
    return ZXC_OK;
}

/**
 * @brief Encodes a data block using the General High Velocity (GHI) compression format.
 *
 * 1. Compression Strategy
 * It uses an LZ77-based algorithm with a sliding window (64KB) and a hash table/chain table
 * mechanism.
 *
 * 2. Token Format (Fixed-Width)
 * Unlike the standard GLO block which uses 1-byte tokens (4-bit literal length / 4-bit match
 * length), GHI uses 4-byte (32-bit) sequence records for better performance on long runs:
 * Literal Length (LL): 8 bits (stores 0-254; 255 indicates overflow).
 * Match Length (ML): 8 bits (stores 0-254; 255 indicates overflow).
 * Offset: 16 bits (supports the full 64KB window).
 * This format minimizes the number of expensive VByte reads during decompression for common
 * sequences where lengths are between 16 and 255.
 *
 * @param[in,out] ctx   Pointer to the compression context containing hash tables
 * and configuration.
 * @param[in] src       Pointer to the input source data.
 * @param[in] src_sz    Size of the input data in bytes.
 * @param[out] dst      Pointer to the destination buffer where compressed data will
 * be written.
 * @param[in] dst_cap   Maximum capacity of the destination buffer.
 * @param[out] out_sz   Pointer to a variable that will receive the total size
 * of the compressed output.
 *
 * @return ZXC_OK on success, or a negative zxc_error_t code (e.g., ZXC_ERROR_DST_TOO_SMALL) if an
 * error occurs (e.g., buffer overflow).
 */
static int zxc_encode_block_ghi(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap,
                                size_t* RESTRICT const out_sz) {
    const int level = ctx->compression_level;
    const size_t dict_sz = ctx->dict_size;

    const zxc_lz77_params_t lzp = zxc_get_lz77_params(level);

    ctx->epoch++;
    if (UNLIKELY(ctx->epoch >= ctx->max_epoch)) {
        ZXC_MEMSET(ctx->hash_table, 0, ZXC_LZ_HASH_SIZE * sizeof(uint32_t));
        ZXC_MEMSET(ctx->hash_tags, 0, ZXC_LZ_HASH_SIZE * sizeof(uint8_t));
        ctx->epoch = 1;
    }
    const uint32_t offset_bits = ctx->offset_bits;
    const uint32_t offset_mask = ctx->offset_mask;
    const uint32_t epoch_mark = ctx->epoch << offset_bits;

    if (dict_sz > 0)
        zxc_lz_seed_dict(src, dict_sz, ctx->hash_table, ctx->hash_tags, ctx->chain_table,
                         epoch_mark, offset_mask, level);

    const uint8_t* ip = src + dict_sz;
    const uint8_t* iend = src + src_sz;
    const uint8_t* anchor = ip;
    const uint8_t* search_limit = iend - ZXC_LZ_SEARCH_MARGIN;

    uint32_t* const hash_table = ctx->hash_table;
    uint8_t* const hash_tags = ctx->hash_tags;
    uint8_t* const buf_extras = ctx->buf_extras;
    uint16_t* const chain_table = ctx->chain_table;
    uint8_t* const literals = ctx->literals;
    uint32_t* const buf_sequences = ctx->buf_sequences;

    uint32_t seq_c = 0;
    size_t extras_c = 0;
    size_t lit_c = 0;
    uint16_t max_offset = 0;

    while (LIKELY(ip < search_limit)) {
        size_t dist = (size_t)(ip - anchor);
        size_t step = lzp.step_base + (dist >> lzp.step_shift);
        if (UNLIKELY(ip + step >= search_limit)) step = 1;

        ZXC_PREFETCH_READ(ip + step * 4 + ZXC_CACHE_LINE_SIZE);

        if (LIKELY(ip + step + sizeof(uint64_t) <= iend)) {
            const uint64_t v_next = zxc_le64(ip + step);
            // cppcheck-suppress unreadVariable
            const uint32_t h_next = zxc_hash_func(v_next, 0);
            ZXC_PREFETCH_READ(&hash_tags[h_next]);
            ZXC_PREFETCH_READ(&hash_table[h_next]);
        }

        const zxc_match_t m =
            zxc_lz77_find_best_match(src, ip, iend, search_limit, anchor, hash_table, hash_tags,
                                     chain_table, epoch_mark, offset_mask, level, lzp,
                                     /*last_off=*/0U);

        if (m.ref) {
            ip -= m.backtrack;
            const uint32_t ll = (uint32_t)(ip - anchor);
            const uint32_t ml = m.len - ZXC_LZ_MIN_MATCH_LEN;
            const uint32_t off = (uint32_t)(ip - m.ref);

            if (ll > 0) {
                if (LIKELY(anchor + ZXC_PAD_SIZE <= iend)) {
                    zxc_copy32(literals + lit_c, anchor);
                    if (UNLIKELY(ll > ZXC_PAD_SIZE)) {
                        ZXC_MEMCPY(literals + lit_c + ZXC_PAD_SIZE, anchor + ZXC_PAD_SIZE,
                                   ll - ZXC_PAD_SIZE);
                    }
                } else {
                    ZXC_MEMCPY(literals + lit_c, anchor, ll);
                }
                lit_c += ll;
            }

            const uint32_t ll_write = (ll >= ZXC_SEQ_LL_MASK) ? 255U : ll;
            const uint32_t ml_write = (ml >= ZXC_SEQ_ML_MASK) ? 255U : ml;
            const uint32_t seq_val = (ll_write << (ZXC_SEQ_ML_BITS + ZXC_SEQ_OFF_BITS)) |
                                     (ml_write << ZXC_SEQ_OFF_BITS) |
                                     ((off - ZXC_LZ_OFFSET_BIAS) & ZXC_SEQ_OFF_MASK);
            if ((off - ZXC_LZ_OFFSET_BIAS) > max_offset)
                max_offset = (uint16_t)(off - ZXC_LZ_OFFSET_BIAS);
            buf_sequences[seq_c] = seq_val;
            seq_c++;

            if (ll >= ZXC_SEQ_LL_MASK) {
                const size_t n = zxc_write_varint(buf_extras + extras_c, ll - ZXC_SEQ_LL_MASK);
                if (UNLIKELY(n == 0)) return ZXC_ERROR_OVERFLOW;
                extras_c += n;
            }
            if (ml >= ZXC_SEQ_ML_MASK) {
                const size_t n = zxc_write_varint(buf_extras + extras_c, ml - ZXC_SEQ_ML_MASK);
                if (UNLIKELY(n == 0)) return ZXC_ERROR_OVERFLOW;
                extras_c += n;
            }

            ip += m.len;
            anchor = ip;
        } else {
            ip += step;
        }
    }

    const size_t last_lits = iend - anchor;
    if (last_lits > 0) {
        ZXC_MEMCPY(literals + lit_c, anchor, last_lits);
        lit_c += last_lits;
    }

    zxc_block_header_t bh = {.block_type = ZXC_BLOCK_GHI};
    uint8_t* const p = dst + ZXC_BLOCK_HEADER_SIZE;
    size_t rem = dst_cap - ZXC_BLOCK_HEADER_SIZE;

    // Decide offset encoding mode
    const zxc_gnr_header_t gh = {.n_sequences = seq_c,
                                 .n_literals = (uint32_t)lit_c,
                                 .enc_lit = ZXC_SECTION_ENCODING_RAW,
                                 .enc_litlen = 0,
                                 .enc_mlen = 0,
                                 .enc_off = (uint8_t)(max_offset <= 255) ? 1 : 0};

    zxc_section_desc_t desc[ZXC_GHI_SECTIONS] = {0};
    desc[0].sizes = (uint64_t)lit_c | ((uint64_t)lit_c << 32);
    size_t sz_seqs = seq_c * sizeof(uint32_t);
    desc[1].sizes = (uint64_t)sz_seqs | ((uint64_t)sz_seqs << 32);
    desc[2].sizes = (uint64_t)extras_c | ((uint64_t)extras_c << 32);

    const int ghs = zxc_write_ghi_header_and_desc(p, rem, &gh, desc);
    if (UNLIKELY(ghs < 0)) return ghs;

    uint8_t* p_curr = p + ghs;
    rem -= ghs;

    // Extract stream sizes once
    const size_t sz_lit = (size_t)(desc[0].sizes & ZXC_SECTION_SIZE_MASK);
    const size_t sz_seq = (size_t)(desc[1].sizes & ZXC_SECTION_SIZE_MASK);
    const size_t sz_ext = (size_t)(desc[2].sizes & ZXC_SECTION_SIZE_MASK);

    if (UNLIKELY(rem < sz_lit + sz_seq + sz_ext)) return ZXC_ERROR_DST_TOO_SMALL;

    ZXC_MEMCPY(p_curr, literals, lit_c);
    p_curr += lit_c;
    rem -= sz_lit;

    if (UNLIKELY(rem < sz_seq)) return ZXC_ERROR_DST_TOO_SMALL;
    // Write sequences in little-endian order
#ifdef ZXC_BIG_ENDIAN
    for (uint32_t i = 0; i < seq_c; i++) {
        zxc_store_le32(p_curr, buf_sequences[i]);
        p_curr += sizeof(uint32_t);
    }
#else
    ZXC_MEMCPY(p_curr, buf_sequences, sz_seq);
    p_curr += sz_seq;
#endif

    // --- WRITE EXTRAS ---
    ZXC_MEMCPY(p_curr, buf_extras, sz_ext);
    p_curr += sz_ext;

    bh.comp_size = (uint32_t)(p_curr - (dst + ZXC_BLOCK_HEADER_SIZE));
    const int hw = zxc_write_block_header(dst, dst_cap, &bh);
    if (UNLIKELY(hw < 0)) return hw;

    // Checksum will be appended by the wrapper
    *out_sz = ZXC_BLOCK_HEADER_SIZE + bh.comp_size;
    return ZXC_OK;
}

/**
 * @brief Encodes a raw data block (uncompressed).
 *
 * This function prepares and writes a "RAW" type block into the destination
 * buffer. It handles the block header and copying of source data; any checksum
 * is appended separately by the wrapper.
 *
 * @param[in] src Pointer to the source data to encode.
 * @param[in] src_sz Size of the source data in bytes.
 * @param[out] dst Pointer to the destination buffer.
 * @param[in] dst_cap Maximum capacity of the destination buffer.
 * @param[out] out_sz Pointer to a variable receiving the total written size
 * (header + data).
 *
 * @return ZXC_OK on success, or a negative zxc_error_t code (e.g., ZXC_ERROR_DST_TOO_SMALL) if the
 * destination buffer capacity is insufficient.
 */
static int zxc_encode_block_raw(const uint8_t* RESTRICT src, const size_t src_sz,
                                uint8_t* RESTRICT const dst, const size_t dst_cap,
                                size_t* RESTRICT const out_sz) {
    if (UNLIKELY(dst_cap < ZXC_BLOCK_HEADER_SIZE + src_sz)) return ZXC_ERROR_DST_TOO_SMALL;

    // Compute block RAW
    zxc_block_header_t bh;
    bh.block_type = ZXC_BLOCK_RAW;
    bh.block_flags = 0;  // Checksum flag moved to file header
    bh.reserved = 0;
    bh.comp_size = (uint32_t)src_sz;

    const int hw = zxc_write_block_header(dst, dst_cap, &bh);
    if (UNLIKELY(hw < 0)) return hw;

    ZXC_MEMCPY(dst + ZXC_BLOCK_HEADER_SIZE, src, src_sz);

    // Checksum will be appended by the wrapper
    *out_sz = ZXC_BLOCK_HEADER_SIZE + src_sz;
    return ZXC_OK;
}

/**
 * @brief Compresses one chunk into a single ZXC block (the compression hot path).
 *
 * Selects the GHI encoder at level <= 2, otherwise GLO; falls back to a RAW
 * block when the coded form would not shrink the data. When @c ctx->dict_size
 * is > 0, @p chunk is the [dict | block] concat and only the block tail counts
 * toward the expansion check. Appends the per-block checksum when enabled.
 *
 * @param[in,out] ctx     Compression context (level, dict_size, checksum, buffers).
 * @param[in]     chunk   Source bytes ([dict | block] when a dictionary is active).
 * @param[in]     src_sz  Length of @p chunk in bytes (includes any dict prefix).
 * @param[out]    dst     Destination block buffer.
 * @param[in]     dst_cap Capacity of @p dst in bytes.
 * @return Compressed block size in bytes on success, or a negative @ref zxc_error_t.
 */
// cppcheck-suppress unusedFunction
int zxc_compress_chunk_wrapper(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT chunk,
                               const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap) {
    const size_t dict_sz = ctx->dict_size;
    const size_t block_sz = src_sz - dict_sz;
    const uint8_t* block_data = chunk + dict_sz;
    size_t w = 0;
    int res = ZXC_OK;

    if (ctx->compression_level <= 2)
        res = zxc_encode_block_ghi(ctx, chunk, src_sz, dst, dst_cap, &w);
    else
        res = zxc_encode_block_glo(ctx, chunk, src_sz, dst, dst_cap, &w);

    // Check expansion against block data size (excluding dict prefix).
    if (UNLIKELY(res != ZXC_OK || w >= block_sz)) {
        res = zxc_encode_block_raw(block_data, block_sz, dst, dst_cap, &w);
        if (UNLIKELY(res != ZXC_OK)) return res;
    }

    if (ctx->checksum_enabled) {
        // Calculate checksum on the compressed payload (w currently excludes checksum)
        // Header is at dst, data starts at dst + ZXC_BLOCK_HEADER_SIZE
        if (UNLIKELY(w < ZXC_BLOCK_HEADER_SIZE || w + ZXC_BLOCK_CHECKSUM_SIZE > dst_cap))
            return ZXC_ERROR_OVERFLOW;

        uint32_t payload_sz = (uint32_t)(w - ZXC_BLOCK_HEADER_SIZE);
        uint32_t crc =
            zxc_checksum(dst + ZXC_BLOCK_HEADER_SIZE, payload_sz, ZXC_CHECKSUM_RAPIDHASH);
        zxc_store_le32(dst + w, crc);
        w += ZXC_BLOCK_CHECKSUM_SIZE;
    }

    return (int)w;
}
