/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */

/**
 * @file zxc_huffman.c
 * @brief Canonical, length-limited (ZXC_HUF_MAX_CODE_LEN) Huffman codec for the GLO literal
 *
 * Canonical, length-limited (ZXC_HUF_MAX_CODE_LEN) Huffman codec for the GLO literal
 * stream at compression level >= 6. Codes are emitted LSB-first; the
 * decoder uses a 2048-entry multi-symbol lookup table (11-bit lookup,
 * 1 or 2 symbols per lookup depending on the cumulative code length)
 * and a 4-way interleaved hot loop. Public declarations live in
 * zxc_internal.h; the rest is private to this translation unit.
 */

/*
 * Function Multi-Versioning Support
 * If ZXC_FUNCTION_SUFFIX is defined (e.g. _avx2, _neon), rename the public
 * entry points so each variant TU produces its own copy under a unique symbol
 * (e.g. zxc_huf_decode_section_avx2). The runtime dispatcher in
 * zxc_compress.c / zxc_decompress.c routes to the matching variant.
 *
 * The defines sit before zxc_internal.h so the header's prototypes are
 * rewritten with the same suffix as the definitions below.
 */
#ifdef ZXC_FUNCTION_SUFFIX
#define ZXC_CAT_IMPL(x, y) x##y
#define ZXC_CAT(x, y) ZXC_CAT_IMPL(x, y)
#define zxc_huf_build_code_lengths ZXC_CAT(zxc_huf_build_code_lengths, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_encode_section ZXC_CAT(zxc_huf_encode_section, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_decode_section ZXC_CAT(zxc_huf_decode_section, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_encode_section_dict ZXC_CAT(zxc_huf_encode_section_dict, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_decode_section_dict ZXC_CAT(zxc_huf_decode_section_dict, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_build_dec_table ZXC_CAT(zxc_huf_build_dec_table, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_pack_lengths ZXC_CAT(zxc_huf_pack_lengths, ZXC_FUNCTION_SUFFIX)
#define zxc_huf_unpack_lengths ZXC_CAT(zxc_huf_unpack_lengths, ZXC_FUNCTION_SUFFIX)
#endif

#include "../../include/zxc_error.h"
#include "zxc_internal.h"

/* The decoder lookup table entry type (zxc_huf_dec_entry_t) lives in
 * zxc_internal.h so the compression context can carry a prebuilt table for
 * the shared dictionary literal table. Bit layout recap:
 * sym1(0..7) | sym2(8..15) | len1(16..19) | len_total(20..23) | n_extra(24). */
#define ZXC_HUF_ENTRY(sym1, sym2, len1, len_total, n_extra)                  \
    ((uint32_t)(sym1) | ((uint32_t)(sym2) << 8) | ((uint32_t)(len1) << 16) | \
     ((uint32_t)(len_total) << 20) | ((uint32_t)(n_extra) << 24))

/* ===========================================================================
 * Length-limited Huffman: boundary package-merge
 * ===========================================================================
 *
 * Builds optimal length-limited Huffman code lengths (max length
 * ZXC_HUF_MAX_CODE_LEN) on 256-symbol alphabets. Package-merge is run for
 * ZXC_HUF_MAX_CODE_LEN levels; each level holds up to 2N items (leaves +
 * paired packages). Selection of the cheapest 2N - 2 items at level
 * ZXC_HUF_MAX_CODE_LEN gives the appearance count of each leaf, which is
 * its code length.
 */

typedef zxc_huf_pm_item_t pm_item_t;

typedef struct {
    uint32_t w;
    int16_t sym;
} pm_leaf_t;

typedef zxc_huf_pm_frame_t frame_t;

/**
 * @brief Sort `pm_leaf_t` array by ascending weight, ties broken by ascending symbol.
 *
 * Bucket sort on `floor(log2(weight))` (32 buckets), with insertion sort
 * inside each bucket. Replaces a libc `qsort` call: the comparator's
 * indirect call dominated, and frequency distributions cluster naturally
 * across ~10-14 magnitude buckets, so intra-bucket lists stay short and
 * insertion sort is branch-friendly. Deterministic tie-break on `sym` is
 * applied inside the insertion sort.
 *
 * Precondition: all weights are > 0 (zero-frequency symbols are filtered
 * by the caller before this runs).
 *
 * @param[in,out] leaves  Leaf array, sorted in place (ascending weight, then
 *                        ascending @c sym on ties).
 * @param[in]     n       Number of leaves; @c n < 2 is effectively a no-op.
 */
static void pm_leaves_sort(pm_leaf_t* RESTRICT leaves, const int n) {
    /* One bucket per possible value of floor(log2(weight)) for a 32-bit
     * weight, i.e. 32 buckets. */
    enum { NUM_BUCKETS = 32 };
    int count[NUM_BUCKETS];
    int offset[NUM_BUCKETS + 1]; /* +1 sentinel = n, avoids end-of-bucket branch. */
    uint8_t bucket_of[ZXC_HUF_NUM_SYMBOLS];
    pm_leaf_t tmp[ZXC_HUF_NUM_SYMBOLS];

    ZXC_MEMSET(count, 0, sizeof(count));
    for (int i = 0; i < n; i++) {
        const unsigned b = zxc_log2_u32(leaves[i].w);
        bucket_of[i] = (uint8_t)b;
        count[b]++;
    }

    int acc = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        offset[b] = acc;
        acc += count[b];
    }
    offset[NUM_BUCKETS] = n;

    int pos[NUM_BUCKETS];
    ZXC_MEMCPY(pos, offset, sizeof(pos));
    for (int i = 0; i < n; i++) {
        tmp[pos[bucket_of[i]]++] = leaves[i];
    }

    for (int b = 0; b < NUM_BUCKETS; b++) {
        if (count[b] < 2) continue;
        const int s = offset[b];
        const int e = offset[b + 1];
        for (int i = s + 1; i < e; i++) {
            const pm_leaf_t key = tmp[i];
            int j = i - 1;
            while (j >= s && (tmp[j].w > key.w || (tmp[j].w == key.w && tmp[j].sym > key.sym))) {
                tmp[j + 1] = tmp[j];
                j--;
            }
            tmp[j + 1] = key;
        }
    }

    ZXC_MEMCPY(leaves, tmp, (size_t)n * sizeof(pm_leaf_t));
}

/**
 * @brief Build length-limited canonical Huffman code lengths.
 *
 * Runs the boundary package-merge algorithm capped at `ZXC_HUF_MAX_CODE_LEN`.
 * Symbols with `freq[i] == 0` get `code_len[i] == 0`; every other symbol
 * receives a length in `[1, ZXC_HUF_MAX_CODE_LEN]`. The single-present-symbol
 * case is handled as a degenerate code of length 1.
 *
 * @param[in]  freq     Frequency table indexed by symbol (0..255).
 * @param[out] code_len Output code-length array, written in full.
 * @param[in]  scratch  Optional scratch of `ZXC_HUF_BUILD_SCRATCH_SIZE` bytes
 *                      (carved into items / counts / stack regions). If
 *                      `NULL`, the function allocates its own working memory
 *                      for the duration of the call.
 * @return `ZXC_OK` on success, `ZXC_ERROR_MEMORY` or `ZXC_ERROR_CORRUPT_DATA`
 *         on failure.
 */
int zxc_huf_build_code_lengths(const uint32_t* RESTRICT freq, uint8_t* RESTRICT code_len,
                               void* RESTRICT scratch) {
    ZXC_MEMSET(code_len, 0, ZXC_HUF_NUM_SYMBOLS);

    pm_leaf_t leaves[ZXC_HUF_NUM_SYMBOLS];
    int n = 0;
    for (int i = 0; i < ZXC_HUF_NUM_SYMBOLS; i++) {
        if (freq[i] > 0) {
            leaves[n].w = freq[i];
            leaves[n].sym = (int16_t)i;
            n++;
        }
    }
    if (UNLIKELY(n == 0)) return ZXC_ERROR_CORRUPT_DATA;
    if (n == 1) {
        code_len[leaves[0].sym] = 1;
        return ZXC_OK;
    }

    pm_leaves_sort(leaves, n);

    /* n <= 256 <= 2^ZXC_HUF_MAX_CODE_LEN, so length-limit is always feasible. */
    const int max_per_level = 2 * n;

    /* Working buffers: either carve from caller-provided scratch (sized for
     * the worst-case alphabet) or fall back to per-call malloc/free. */
    pm_item_t* items;
    int* counts;
    frame_t* stack;
    pm_item_t* owned_items = NULL;
    int* owned_counts = NULL;
    frame_t* owned_stack = NULL;
    if (scratch) {
        uint8_t* p = (uint8_t*)scratch;
        items = (pm_item_t*)p;
        p += (size_t)ZXC_HUF_MAX_CODE_LEN * (size_t)ZXC_HUF_PM_LEVEL_BOUND * sizeof(pm_item_t);
        p = (uint8_t*)(((uintptr_t)p + 7u) & ~(uintptr_t)7u);
        counts = (int*)p;
        ZXC_MEMSET(counts, 0, (size_t)ZXC_HUF_MAX_CODE_LEN * sizeof(int));
        p += (size_t)ZXC_HUF_MAX_CODE_LEN * sizeof(int);
        p = (uint8_t*)(((uintptr_t)p + 7u) & ~(uintptr_t)7u);
        stack = (frame_t*)p;
    } else {
        owned_items = (pm_item_t*)ZXC_MALLOC((size_t)ZXC_HUF_MAX_CODE_LEN * (size_t)max_per_level *
                                             sizeof(pm_item_t));
        owned_counts = (int*)ZXC_CALLOC((size_t)ZXC_HUF_MAX_CODE_LEN, sizeof(int));
        owned_stack = (frame_t*)ZXC_MALLOC((size_t)ZXC_HUF_MAX_CODE_LEN * (size_t)max_per_level *
                                           sizeof(frame_t));
        if (UNLIKELY(!owned_items || !owned_counts || !owned_stack)) {
            ZXC_FREE(owned_items);
            ZXC_FREE(owned_counts);
            ZXC_FREE(owned_stack);
            return ZXC_ERROR_MEMORY;
        }
        items = owned_items;
        counts = owned_counts;
        stack = owned_stack;
    }
#define ITEM(k, i) items[(size_t)(k) * (size_t)max_per_level + (size_t)(i)]

    /* Level 0 (logical level 1): the leaves themselves, already sorted. */
    for (int i = 0; i < n; i++) {
        ITEM(0, i).weight = leaves[i].w;
        ITEM(0, i).left = -1;
        ITEM(0, i).right = -1;
        ITEM(0, i).sym = leaves[i].sym;
    }
    counts[0] = n;

    /* Levels 1..ZXC_HUF_MAX_CODE_LEN-1: merge sorted leaves with sorted packages from the previous
     * level. */
    for (int k = 1; k < ZXC_HUF_MAX_CODE_LEN; k++) {
        const int prev = counts[k - 1];
        const int packs = prev / 2;
        int li = 0;
        int pi = 0;
        int n_lvl = 0;
        while (li < n || pi < packs) {
            const uint32_t wl = (li < n) ? leaves[li].w : UINT32_MAX;
            const uint32_t wp =
                (pi < packs)
                    ? (uint32_t)(ITEM(k - 1, 2 * pi).weight + ITEM(k - 1, 2 * pi + 1).weight)
                    : UINT32_MAX;
            if (wl <= wp && li < n) {
                ITEM(k, n_lvl).weight = wl;
                ITEM(k, n_lvl).left = -1;
                ITEM(k, n_lvl).right = -1;
                ITEM(k, n_lvl).sym = leaves[li].sym;
                li++;
            } else {
                ITEM(k, n_lvl).weight = wp;
                ITEM(k, n_lvl).left = (int16_t)(2 * pi);
                ITEM(k, n_lvl).right = (int16_t)(2 * pi + 1);
                ITEM(k, n_lvl).sym = -1;
                pi++;
            }
            n_lvl++;
        }
        counts[k] = n_lvl;
    }

    /* Step 3: take first 2n-2 items at level ZXC_HUF_MAX_CODE_LEN-1; trace back, counting leaf
     * appearances. */
    int n_take = 2 * n - 2;
    if (n_take > counts[ZXC_HUF_MAX_CODE_LEN - 1]) n_take = counts[ZXC_HUF_MAX_CODE_LEN - 1];

    /* Worst case stack depth: (ZXC_HUF_MAX_CODE_LEN * n_take) frames; bounded by
     * ZXC_HUF_MAX_CODE_LEN * 2n. `stack` was set up earlier from scratch (or
     * the local malloc fallback). */
    int sp = 0;
    for (int i = 0; i < n_take; i++) {
        stack[sp].lvl = (int8_t)(ZXC_HUF_MAX_CODE_LEN - 1);
        stack[sp].idx = (int16_t)i;
        sp++;
    }
    while (sp > 0) {
        frame_t f = stack[--sp];
        const pm_item_t* it = &ITEM(f.lvl, f.idx);
        if (it->sym >= 0) {
            code_len[it->sym]++;
        } else {
            stack[sp].lvl = (int8_t)(f.lvl - 1);
            stack[sp].idx = it->left;
            sp++;
            stack[sp].lvl = (int8_t)(f.lvl - 1);
            stack[sp].idx = it->right;
            sp++;
        }
    }

    if (owned_items) {
        ZXC_FREE(owned_items);
        ZXC_FREE(owned_counts);
        ZXC_FREE(owned_stack);
    }
#undef ITEM
    return ZXC_OK;
}

/* ===========================================================================
 * Canonical code construction (LSB-first by bit-reversing canonical MSB codes)
 * =========================================================================*/

/**
 * @brief Reverse the low @p n bits of @p v.
 *
 * Used to convert MSB-first canonical Huffman codes (the natural form
 * produced by the canonical-code construction) into LSB-first codes that
 * can be packed into the bit writer with a single shift-or.
 *
 * @param[in] v Value whose low @p n bits will be reversed.
 * @param[in] n Number of significant bits in @p v (1..32).
 * @return The bit-reversed value, with bits above position @p n set to 0.
 */
static uint32_t reverse_bits(uint32_t v, const int n) {
    uint32_t r = 0;
    for (int i = 0; i < n; i++) {
        r = (r << 1) | (v & 1u);
        v >>= 1;
    }
    return r;
}

/**
 * @brief Build the canonical LSB-first Huffman codes for a length table.
 *
 * Generates MSB-first canonical codes following RFC 1951 3.2.2, then
 * bit-reverses each so the encoder can emit them with a plain
 * `accum |= code << bits` step. Absent symbols (length 0) receive code 0.
 *
 * @param[in]  code_len Per-symbol code lengths.
 * @param[out] codes    Per-symbol LSB-first canonical codes.
 */
static void build_canonical_codes(const uint8_t* RESTRICT code_len, uint32_t* RESTRICT codes) {
    uint32_t bl_count[ZXC_HUF_MAX_CODE_LEN + 1] = {0};
    for (int i = 0; i < ZXC_HUF_NUM_SYMBOLS; i++) {
        bl_count[code_len[i]]++;
    }
    bl_count[0] = 0;

    uint32_t next_code[ZXC_HUF_MAX_CODE_LEN + 2] = {0};
    uint32_t code = 0;
    for (int k = 1; k <= ZXC_HUF_MAX_CODE_LEN + 1; k++) {
        code = (code + bl_count[k - 1]) << 1;
        next_code[k] = code;
    }

    for (int i = 0; i < ZXC_HUF_NUM_SYMBOLS; i++) {
        const int l = code_len[i];
        if (l == 0) {
            codes[i] = 0;
        } else {
            const uint32_t msb_code = next_code[l]++;
            codes[i] = reverse_bits(msb_code, l);
        }
    }
}

/* ===========================================================================
 * 128-byte length header: 256 x 4-bit lengths, low nibble first.
 * =========================================================================*/

/**
 * @brief Pack 256 4-bit code lengths into the 128-byte section header.
 *
 * The packing is little-endian within each byte: low nibble holds
 * `code_len[2*i]`, high nibble holds `code_len[2*i + 1]`. The function
 * silently truncates any length > 15; callers must enforce the cap of
 * `ZXC_HUF_MAX_CODE_LEN` (<= 15) before calling.
 *
 * @param[in]  code_len Per-symbol code lengths (length `ZXC_HUF_NUM_SYMBOLS`).
 * @param[out] out      Output header buffer of `ZXC_HUF_TABLE_SIZE` bytes.
 */
static void pack_lengths_header(const uint8_t* RESTRICT code_len, uint8_t* RESTRICT out) {
    for (int i = 0; i < ZXC_HUF_NUM_SYMBOLS; i += 2) {
        const uint8_t lo = code_len[i] & 0x0F;
        const uint8_t hi = code_len[i + 1] & 0x0F;
        out[i >> 1] = (uint8_t)(lo | (hi << 4));
    }
}

/**
 * @brief Decode the 128-byte length header back into 256 code lengths.
 *
 * Inverts ::pack_lengths_header and validates the two structural invariants:
 * no length exceeds `ZXC_HUF_MAX_CODE_LEN`, and at least one symbol is
 * present.
 *
 * @param[in]  in       Input header buffer of `ZXC_HUF_TABLE_SIZE` bytes.
 * @param[out] code_len Output code-length array of length `ZXC_HUF_NUM_SYMBOLS`.
 * @return `ZXC_OK` on success, `ZXC_ERROR_CORRUPT_DATA` if a length is too
 *         large or the table is empty.
 */
static int unpack_lengths_header(const uint8_t* RESTRICT in, uint8_t* RESTRICT code_len) {
    int max_len = 0;
    int n_present = 0;
    for (int i = 0; i < ZXC_HUF_NUM_SYMBOLS; i += 2) {
        const uint8_t b = in[i >> 1];
        const uint8_t lo = b & 0x0F;
        const uint8_t hi = (uint8_t)(b >> 4);
        code_len[i] = lo;
        code_len[i + 1] = hi;
        if (lo > max_len) max_len = lo;
        if (hi > max_len) max_len = hi;
        if (lo) n_present++;
        if (hi) n_present++;
    }
    if (UNLIKELY(max_len > ZXC_HUF_MAX_CODE_LEN)) return ZXC_ERROR_CORRUPT_DATA;
    if (UNLIKELY(n_present == 0)) return ZXC_ERROR_CORRUPT_DATA;
    return ZXC_OK;
}

/* ===========================================================================
 * Bit writer (LSB-first)
 * =========================================================================*/

typedef struct {
    uint8_t* ptr;
    uint8_t* end;
    uint64_t accum;
    int bits;
    int err;
} bit_writer_t;

/**
 * @brief Initialise an LSB-first bit writer over a caller-owned buffer.
 *
 * @param[out] bw  Writer to initialise.
 * @param[out] dst Output buffer (writer takes no ownership).
 * @param[in]  cap Capacity of @p dst in bytes.
 */
static ZXC_ALWAYS_INLINE void bw_init(bit_writer_t* RESTRICT bw, uint8_t* RESTRICT dst,
                                      const size_t cap) {
    bw->ptr = dst;
    bw->end = dst + cap;
    bw->accum = 0;
    bw->bits = 0;
    bw->err = 0;
}

/**
 * @brief Append the low @p len bits of @p code to the writer's bitstream.
 *
 * Bits are consumed from the LSB end. When the internal accumulator has
 * accumulated 8 or more bits, full bytes are flushed to the output buffer.
 * If the buffer is exhausted mid-flush the writer's `err` flag is set;
 * subsequent ::bw_finish reports `ZXC_ERROR_DST_TOO_SMALL`.
 *
 * @param[in,out] bw   Writer state.
 * @param[in]     code Code bits to emit (the low @p len bits matter).
 * @param[in]     len  Number of bits to emit (1..ZXC_HUF_MAX_CODE_LEN).
 */
static ZXC_ALWAYS_INLINE void bw_put(bit_writer_t* RESTRICT bw, const uint32_t code,
                                     const int len) {
    bw->accum |= ((uint64_t)code) << bw->bits;
    bw->bits += len;
    if (LIKELY((size_t)(bw->end - bw->ptr) >= sizeof(uint64_t))) {
        zxc_store_le64(bw->ptr, bw->accum);
    } else {
        if (UNLIKELY(bw->ptr >= bw->end)) {
            bw->err = 1;
            bw->bits = 0;
            return;
        }
        *bw->ptr = (uint8_t)bw->accum;
    }
    const int n = bw->bits >> 3; /* 0 or 1 full byte to flush */
    bw->ptr += n;
    bw->accum >>= n << 3;
    bw->bits &= 7;
}

/**
 * @brief Flush any partial trailing byte and finalise the bit writer.
 *
 * Writes the (zero-padded) trailing byte if the accumulator holds any bits.
 *
 * @param[in,out] bw Writer state.
 * @return `ZXC_OK` on success, `ZXC_ERROR_DST_TOO_SMALL` if the buffer was
 *         exhausted at any point.
 */
static ZXC_ALWAYS_INLINE int bw_finish(bit_writer_t* RESTRICT bw) {
    if (bw->bits > 0) {
        if (UNLIKELY(bw->ptr >= bw->end)) return ZXC_ERROR_DST_TOO_SMALL;
        *bw->ptr++ = (uint8_t)bw->accum;
        bw->accum = 0;
        bw->bits = 0;
    }
    return UNLIKELY(bw->err) ? ZXC_ERROR_DST_TOO_SMALL : ZXC_OK;
}

/* ===========================================================================
 * Encoder
 * =========================================================================*/

/**
 * @brief Shared encoder body: 6-byte sub-stream sizes header + 4 interleaved
 *        sub-streams, written at @p dst. The 128-byte lengths header, when
 *        wanted, is the caller's business (see the two public wrappers).
 *
 * @param[in]  literals    Source literal bytes (must not alias @p dst).
 * @param[in]  n_literals  Number of source bytes (must be > 0).
 * @param[in]  code_len    Per-symbol code lengths for the canonical codes.
 * @param[out] dst         Destination for the sizes header + sub-streams.
 * @param[in]  dst_cap     Capacity of @p dst in bytes.
 * @return Bytes written (>= ZXC_HUF_STREAM_SIZES_HEADER_SIZE) on success,
 *         negative `zxc_error_t` code on failure.
 */
static int zxc_huf_encode_streams(const uint8_t* RESTRICT literals, const size_t n_literals,
                                  const uint8_t* RESTRICT code_len, uint8_t* RESTRICT dst,
                                  const size_t dst_cap) {
    if (UNLIKELY(n_literals == 0)) return ZXC_ERROR_CORRUPT_DATA;
    if (UNLIKELY(dst_cap < (size_t)ZXC_HUF_STREAM_SIZES_HEADER_SIZE))
        return ZXC_ERROR_DST_TOO_SMALL;

    /* 1. Build canonical codes (LSB-first via bit-reversal). */
    uint32_t codes[ZXC_HUF_NUM_SYMBOLS];
    build_canonical_codes(code_len, codes);

    /* 2. Reserve 6 bytes for sub-stream sizes; encode 4 sub-streams after them. */
    uint8_t* const sizes_hdr = dst;
    uint8_t* const stream_base = sizes_hdr + ZXC_HUF_STREAM_SIZES_HEADER_SIZE;
    const uint8_t* const stream_end = dst + dst_cap;

    const size_t Q = (n_literals + ZXC_HUF_NUM_STREAMS - 1) / ZXC_HUF_NUM_STREAMS;

    bit_writer_t bw;
    uint8_t* p = stream_base;
    size_t s_sizes[ZXC_HUF_NUM_STREAMS];

    for (int s = 0; s < ZXC_HUF_NUM_STREAMS; s++) {
        const size_t start = (size_t)s * Q;
        size_t stop = start + Q;
        if (stop > n_literals) stop = n_literals;

        const uint8_t* const stream_start = p;
        bw_init(&bw, p, (size_t)(stream_end - p));
        for (size_t i = start; i < stop; i++) {
            const uint8_t sym = literals[i];
            const int len = code_len[sym];
            if (UNLIKELY(len == 0)) return ZXC_ERROR_CORRUPT_DATA; /* symbol absent from table */
            bw_put(&bw, codes[sym], len);
        }
        const int rc = bw_finish(&bw);
        if (UNLIKELY(rc != ZXC_OK)) return rc;
        s_sizes[s] = (size_t)(bw.ptr - stream_start);
        p = bw.ptr;
    }

    /* 3. Persist the 3 explicit sub-stream sizes (s4 is implied). */
    for (int s = 0; s < ZXC_HUF_NUM_STREAMS - 1; s++) {
        if (UNLIKELY(s_sizes[s] > 0xFFFFu)) return ZXC_ERROR_DST_TOO_SMALL;
        zxc_store_le16(sizes_hdr + 2 * s, (uint16_t)s_sizes[s]);
    }

    return (int)(p - dst);
}

/**
 * @brief Encode the literal stream into a full Huffman section payload.
 *
 * Packs the 128-byte lengths header, then delegates to
 * @ref zxc_huf_encode_streams for the 6-byte sub-stream sizes and the 4
 * interleaved LSB-first bit-streams.
 *
 * @param[in]  literals    Source literal bytes (must not alias @p dst).
 * @param[in]  n_literals  Number of source bytes (must be > 0).
 * @param[in]  code_len    Per-symbol code lengths (see @ref zxc_huf_build_code_lengths).
 * @param[out] dst         Destination buffer for the section payload.
 * @param[in]  dst_cap     Capacity of @p dst in bytes.
 * @return Total bytes written on success, negative `zxc_error_t` on failure.
 */
int zxc_huf_encode_section(const uint8_t* RESTRICT literals, const size_t n_literals,
                           const uint8_t* RESTRICT code_len, uint8_t* RESTRICT dst,
                           const size_t dst_cap) {
    if (UNLIKELY(n_literals == 0)) return ZXC_ERROR_CORRUPT_DATA;
    if (UNLIKELY(dst_cap < ZXC_HUF_HEADER_SIZE)) return ZXC_ERROR_DST_TOO_SMALL;

    /* Pack the 128-byte length header, then the streams after it. */
    pack_lengths_header(code_len, dst);
    const int rc = zxc_huf_encode_streams(literals, n_literals, code_len, dst + ZXC_HUF_TABLE_SIZE,
                                          dst_cap - ZXC_HUF_TABLE_SIZE);
    return (rc < 0) ? rc : rc + ZXC_HUF_TABLE_SIZE;
}

/**
 * @brief Encode a literal section using supplied code lengths, WITHOUT the
 *        128-byte lengths header (shared dictionary table).
 *
 * Emits only the 6-byte sub-stream sizes header + 4 sub-streams (a thin pass
 * through @ref zxc_huf_encode_streams); the lengths live in the dictionary.
 *
 * @param[in]  literals    Source literal bytes (must not alias @p dst).
 * @param[in]  n_literals  Number of source bytes (must be > 0).
 * @param[in]  code_len    Per-symbol code lengths from the shared dict table.
 * @param[out] dst         Destination buffer for the section payload.
 * @param[in]  dst_cap     Capacity of @p dst in bytes.
 * @return Bytes written on success, negative `zxc_error_t` on failure
 *         (incl. `ZXC_ERROR_CORRUPT_DATA` if a literal has no code).
 */
int zxc_huf_encode_section_dict(const uint8_t* RESTRICT literals, const size_t n_literals,
                                const uint8_t* RESTRICT code_len, uint8_t* RESTRICT dst,
                                const size_t dst_cap) {
    return zxc_huf_encode_streams(literals, n_literals, code_len, dst, dst_cap);
}

/* ===========================================================================
 * Decoder table builder + 4-way interleaved decoder
 * =========================================================================*/

/**
 * @brief Build the 2048-entry multi-symbol decoder lookup table.
 *
 * Strategy: build a temporary 256-entry single-symbol (8-bit) table, then
 * use it to populate the 2048-entry (11-bit) multi-symbol table. For each
 * 11-bit prefix p:
 *   1. (sym1, len1) = ss[p & 0xFF]   -- always valid, 1 <= len1 <= 8.
 *   2. rem = 11 - len1 E [3, 10] bits remain after consuming the first code.
 *   3. (sym2_cand, len2_cand) = ss[(p >> len1) & 0xFF]. If len2_cand <= rem,
 *      both codes fit in 11 bits -> encode 2-symbol entry. Otherwise the
 *      second code's bit window extends past the lookup width -> keep only
 *      the first symbol and let the next iteration handle the rest.
 *
 * Validates Kraft equality (or the single-present-symbol degenerate case).
 *
 * @param[in]  code_len Per-symbol code lengths from the section header.
 * @param[out] table    Destination 2048-entry lookup table (caller-aligned).
 * @return `ZXC_OK` on success, `ZXC_ERROR_CORRUPT_DATA` on validation failure.
 */
static int build_decode_table(const uint8_t* RESTRICT code_len,
                              zxc_huf_dec_entry_t* RESTRICT table) {
    uint32_t bl_count[ZXC_HUF_MAX_CODE_LEN + 1] = {0};
    int n_present = 0;
    for (int i = 0; i < ZXC_HUF_NUM_SYMBOLS; i++) {
        const uint8_t l = code_len[i];
        if (UNLIKELY(l > ZXC_HUF_MAX_CODE_LEN)) return ZXC_ERROR_CORRUPT_DATA;
        bl_count[l]++;
        if (l) n_present++;
    }
    if (UNLIKELY(n_present == 0)) return ZXC_ERROR_CORRUPT_DATA;
    bl_count[0] = 0;

    /* Validate Kraft equality on the ZXC_HUF_MAX_CODE_LEN axis. */
    {
        uint64_t kraft = 0;
        for (int k = 1; k <= ZXC_HUF_MAX_CODE_LEN; k++) {
            kraft += (uint64_t)bl_count[k] << (ZXC_HUF_MAX_CODE_LEN - k);
        }
        /* Degenerate: single symbol with length 1 (Kraft sum =
         * 2^(ZXC_HUF_MAX_CODE_LEN-1)). Otherwise: full Kraft equality
         * on the ZXC_HUF_MAX_CODE_LEN axis. */
        const int kraft_ok = (n_present == 1) ? (bl_count[1] == 1)
                                              : (kraft == ((uint64_t)1 << ZXC_HUF_MAX_CODE_LEN));
        if (UNLIKELY(!kraft_ok)) return ZXC_ERROR_CORRUPT_DATA;
    }

    uint32_t next_code[ZXC_HUF_MAX_CODE_LEN + 2] = {0};
    {
        uint32_t code = 0;
        for (int k = 1; k <= ZXC_HUF_MAX_CODE_LEN + 1; k++) {
            code = (code + bl_count[k - 1]) << 1;
            next_code[k] = code;
        }
    }

    /* Single-symbol intermediate (ZXC_HUF_MAX_CODE_LEN-bit lookup). Layout:
     * low byte = sym, high byte = len. Filled by replicating each canonical
     * code across all ZXC_HUF_MAX_CODE_LEN-bit windows that share its low
     * `len` bits. */
#define ZXC_HUF_SS_SIZE (1u << ZXC_HUF_MAX_CODE_LEN)
#define ZXC_HUF_SS_MASK ((uint32_t)(ZXC_HUF_SS_SIZE - 1))
    uint16_t ss[ZXC_HUF_SS_SIZE] = {0};

    for (int sym = 0; sym < ZXC_HUF_NUM_SYMBOLS; sym++) {
        const int l = code_len[sym];
        if (l == 0) continue;
        const uint32_t msb_code = next_code[l]++;
        const uint32_t lsb_code = reverse_bits(msb_code, l);
        const uint16_t entry = (uint16_t)((unsigned)l << 8 | (unsigned)sym);
        const uint32_t step = (uint32_t)1 << l;
        for (uint32_t fill = lsb_code; fill < ZXC_HUF_SS_SIZE; fill += step) {
            ss[fill] = entry;
        }
    }

    /* Single-symbol degenerate (Kraft sum = 2^(ZXC_HUF_MAX_CODE_LEN-1)): replicate the one
     * valid entry across every slot. */
    if (UNLIKELY(n_present == 1)) {
        uint16_t valid = 0;
        for (uint32_t i = 0; i < ZXC_HUF_SS_SIZE; i++) {
            if (ss[i] != 0) {
                valid = ss[i];
                break;
            }
        }
        for (uint32_t i = 0; i < ZXC_HUF_SS_SIZE; i++) {
            if (ss[i] == 0) ss[i] = valid;
        }
    }

    /* Build the multi-symbol table. */
    for (uint32_t p = 0; p < ZXC_HUF_DEC_TABLE_SIZE; p++) {
        const uint16_t e1 = ss[p & ZXC_HUF_SS_MASK];
        const uint8_t sym1 = (uint8_t)e1;
        const int len1 = e1 >> 8;
        const int rem = ZXC_HUF_LOOKUP_BITS - len1;

        uint8_t sym2 = 0;
        int len_total = len1;
        int n_extra = 0;

        const uint16_t e2 = ss[(p >> len1) & ZXC_HUF_SS_MASK];
        const int len2 = e2 >> 8;
        if (len2 <= rem) {
            sym2 = (uint8_t)e2;
            len_total = len1 + len2;
            n_extra = 1;
        }

        table[p].entry = ZXC_HUF_ENTRY(sym1, sym2, len1, len_total, n_extra);
    }
#undef ZXC_HUF_SS_SIZE
#undef ZXC_HUF_SS_MASK

    return ZXC_OK;
}

/**
 * @brief Shared decoder body: parses the 6-byte sub-stream sizes header at
 *        @p payload and runs the 4-way interleaved decode with @p table.
 *        The 128-byte lengths header, when present, has already been consumed
 *        by the caller (see the two public wrappers).
 *
 * @param[in]  payload       Sizes header followed by the 4 sub-streams.
 * @param[in]  payload_size  Size of @p payload in bytes.
 * @param[out] dst           Destination for the decoded literals.
 * @param[in]  n_literals    Number of literals to decode (must be > 0).
 * @param[in]  table         Multi-symbol decode table built for this section.
 * @return @c ZXC_OK on success, @c ZXC_ERROR_CORRUPT_DATA on a malformed stream.
 */
static int zxc_huf_decode_streams(const uint8_t* RESTRICT payload, const size_t payload_size,
                                  uint8_t* RESTRICT dst, const size_t n_literals,
                                  const zxc_huf_dec_entry_t* RESTRICT table) {
    if (UNLIKELY(payload_size < (size_t)ZXC_HUF_STREAM_SIZES_HEADER_SIZE || n_literals == 0))
        return ZXC_ERROR_CORRUPT_DATA;

    /* 1. Parse sub-stream sizes. */
    const uint8_t* const sizes_hdr = payload;
    const uint16_t s1 = zxc_le16(sizes_hdr + 0);
    const uint16_t s2 = zxc_le16(sizes_hdr + 2);
    const uint16_t s3 = zxc_le16(sizes_hdr + 4);

    const size_t streams_total = payload_size - ZXC_HUF_STREAM_SIZES_HEADER_SIZE;
    const size_t s123 = (size_t)s1 + (size_t)s2 + (size_t)s3;
    if (UNLIKELY(s123 > streams_total)) return ZXC_ERROR_CORRUPT_DATA;
    const size_t s4 = streams_total - s123;

    const uint8_t* const stream_base = payload + ZXC_HUF_STREAM_SIZES_HEADER_SIZE;
    const size_t off[ZXC_HUF_NUM_STREAMS] = {0, s1, (size_t)s1 + s2, s123};
    const size_t sz[ZXC_HUF_NUM_STREAMS] = {s1, s2, s3, s4};

    /* 4. Initialise 4 bit readers. */
    zxc_bit_reader_t br[ZXC_HUF_NUM_STREAMS];
    for (int s = 0; s < ZXC_HUF_NUM_STREAMS; s++) {
        zxc_br_init(&br[s], stream_base + off[s], sz[s]);
    }

    /* 5. 4-way interleaved multi-symbol decode. Each sub-stream owns a
     * contiguous slice of dst: stream s covers literal indices
     * [s*Q, min((s+1)*Q, N)). With Q = ceil(N/4) the first 3 streams have
     * exactly Q symbols and stream 3 has `N - 3Q` symbols. */
    const size_t Q = (n_literals + ZXC_HUF_NUM_STREAMS - 1) / ZXC_HUF_NUM_STREAMS;
    size_t s_count[ZXC_HUF_NUM_STREAMS];
    uint8_t* s_dst[ZXC_HUF_NUM_STREAMS];
    for (int s = 0; s < ZXC_HUF_NUM_STREAMS; s++) {
        size_t start = (size_t)s * Q;
        size_t stop = start + Q;
        if (start > n_literals) start = n_literals;
        if (stop > n_literals) stop = n_literals;
        s_count[s] = stop - start;
        s_dst[s] = dst + start;
    }

    /* Batched multi-symbol decode. Each ZXC_HUF_BATCH iterations consume
     * <= ZXC_HUF_BATCH_BITS bits per stream, fitting under the 57-bit cap
     * an 8-byte refill can guarantee.
     *
     * Each iter speculatively writes 2 bytes per stream and advances by 1
     * or 2. If only 1 symbol was decoded, the spec byte is overwritten by
     * the next iter, except at end-of-stream where it would corrupt the
     * adjacent stream. The batched loop therefore requires
     * ZXC_HUF_SAFE_MARGIN bytes of headroom per stream. */

    uint8_t* d0 = s_dst[0];
    uint8_t* d1 = s_dst[1];
    uint8_t* d2 = s_dst[2];
    uint8_t* d3 = s_dst[3];

    const uint8_t* const dend0 = s_dst[0] + s_count[0];
    const uint8_t* const dend1 = s_dst[1] + s_count[1];
    const uint8_t* const dend2 = s_dst[2] + s_count[2];
    const uint8_t* const dend3 = s_dst[3] + s_count[3];

    /* Hoist all four bit-reader hot fields into locals. They live in
     * registers for the full duration of the batched loop. */
    uint64_t a0 = br[0].accum;
    uint64_t a1 = br[1].accum;
    uint64_t a2 = br[2].accum;
    uint64_t a3 = br[3].accum;
    int bb0 = br[0].bits;
    int bb1 = br[1].bits;
    int bb2 = br[2].bits;
    int bb3 = br[3].bits;
    const uint8_t* p0 = br[0].ptr;
    const uint8_t* p1 = br[1].ptr;
    const uint8_t* p2 = br[2].ptr;
    const uint8_t* p3 = br[3].ptr;
    const uint8_t* const e0 = br[0].end;
    const uint8_t* const e1 = br[1].end;
    const uint8_t* const e2 = br[2].end;
    const uint8_t* const e3 = br[3].end;

    /* Refill the bit accumulator with up to (ZXC_HUF_ACCUM_BITS - nbits) more
     * bits read from src. Fast path reads 8 bytes at once (LE u64 load); slow
     * path reads byte-by-byte while at least one byte of free room remains. */
#define REFILL(accum, nbits, src, src_end)                                                   \
    do {                                                                                     \
        if (LIKELY((nbits) < ZXC_HUF_BATCH_BITS && (src) + sizeof(uint64_t) <= (src_end))) { \
            (accum) |= zxc_le64(src) << (nbits);                                             \
            const int _n = (ZXC_HUF_ACCUM_BITS - (nbits)) / CHAR_BIT;                        \
            (src) += _n;                                                                     \
            (nbits) += _n * CHAR_BIT;                                                        \
        } else {                                                                             \
            while ((nbits) <= ZXC_HUF_ACCUM_BITS - CHAR_BIT && (src) < (src_end)) {          \
                (accum) |= ((uint64_t)*(src)++) << (nbits);                                  \
                (nbits) += CHAR_BIT;                                                         \
            }                                                                                \
        }                                                                                    \
    } while (0)

    /* Decode one 11-bit window per stream. Always writes 2 bytes per stream
     * (sym1 + spec sym2); advances d_s by 1 + n_extra; advances accum by
     * len_total. Per-stream length accumulators sl0..sl3 collect consumed
     * bits across the batch and are folded into bb_s once at end of batch. */
#define DECODE_ONE()                                             \
    do {                                                         \
        const uint32_t _e0 = table[a0 & ZXC_HUF_TBL_MASK].entry; \
        const uint32_t _e1 = table[a1 & ZXC_HUF_TBL_MASK].entry; \
        const uint32_t _e2 = table[a2 & ZXC_HUF_TBL_MASK].entry; \
        const uint32_t _e3 = table[a3 & ZXC_HUF_TBL_MASK].entry; \
        zxc_store_le16(d0, (uint16_t)_e0);                       \
        zxc_store_le16(d1, (uint16_t)_e1);                       \
        zxc_store_le16(d2, (uint16_t)_e2);                       \
        zxc_store_le16(d3, (uint16_t)_e3);                       \
        const int _t0 = (int)((_e0 >> 20) & 0xF);                \
        const int _t1 = (int)((_e1 >> 20) & 0xF);                \
        const int _t2 = (int)((_e2 >> 20) & 0xF);                \
        const int _t3 = (int)((_e3 >> 20) & 0xF);                \
        d0 += 1 + (int)((_e0 >> 24) & 1);                        \
        d1 += 1 + (int)((_e1 >> 24) & 1);                        \
        d2 += 1 + (int)((_e2 >> 24) & 1);                        \
        d3 += 1 + (int)((_e3 >> 24) & 1);                        \
        a0 >>= _t0;                                              \
        a1 >>= _t1;                                              \
        a2 >>= _t2;                                              \
        a3 >>= _t3;                                              \
        sl0 += _t0;                                              \
        sl1 += _t1;                                              \
        sl2 += _t2;                                              \
        sl3 += _t3;                                              \
    } while (0)

    while ((size_t)(dend0 - d0) >= ZXC_HUF_SAFE_MARGIN &&
           (size_t)(dend1 - d1) >= ZXC_HUF_SAFE_MARGIN &&
           (size_t)(dend2 - d2) >= ZXC_HUF_SAFE_MARGIN &&
           (size_t)(dend3 - d3) >= ZXC_HUF_SAFE_MARGIN) {
        REFILL(a0, bb0, p0, e0);
        REFILL(a1, bb1, p1, e1);
        REFILL(a2, bb2, p2, e2);
        REFILL(a3, bb3, p3, e3);

        int sl0 = 0;
        int sl1 = 0;
        int sl2 = 0;
        int sl3 = 0;
        DECODE_ONE();
        DECODE_ONE();
        DECODE_ONE();
        DECODE_ONE();
        DECODE_ONE();
        bb0 -= sl0;
        bb1 -= sl1;
        bb2 -= sl2;
        bb3 -= sl3;
    }

    /* Per-stream scalar tail (<= ZXC_HUF_SAFE_MARGIN - 1 = 9 symbols per
     * stream). Single-symbol decode using the same 2048-entry table,
     * we read sym1 + len1 only and advance by 1 byte, no spec write. */
#define TAIL_ONE(accum, nbits, src, src_end, dst)                    \
    do {                                                             \
        REFILL(accum, nbits, src, src_end);                          \
        const uint32_t _e = table[(accum) & ZXC_HUF_TBL_MASK].entry; \
        *(dst)++ = (uint8_t)_e;                                      \
        const int _l1 = (int)((_e >> 16) & 0xF);                     \
        (accum) >>= _l1;                                             \
        (nbits) -= _l1;                                              \
    } while (0)

    while (d0 < dend0) TAIL_ONE(a0, bb0, p0, e0, d0);
    while (d1 < dend1) TAIL_ONE(a1, bb1, p1, e1, d1);
    while (d2 < dend2) TAIL_ONE(a2, bb2, p2, e2, d2);
    while (d3 < dend3) TAIL_ONE(a3, bb3, p3, e3, d3);

#undef TAIL_ONE
#undef DECODE_ONE
#undef REFILL
    return ZXC_OK;
}

/**
 * @brief Decode a full Huffman literal section payload.
 *
 * Unpacks the 128-byte lengths header, builds the multi-symbol decode table,
 * then runs the 4-way interleaved decode, writing exactly @p n_literals bytes.
 *
 * @param[in]  payload       Section payload (lengths header + sizes + 4 sub-streams).
 * @param[in]  payload_size  Total payload length in bytes.
 * @param[out] dst           Destination buffer (must not alias @p payload).
 * @param[in]  n_literals    Expected number of decoded bytes.
 * @return `ZXC_OK` on success, negative `zxc_error_t` on failure.
 */
int zxc_huf_decode_section(const uint8_t* RESTRICT payload, const size_t payload_size,
                           uint8_t* RESTRICT dst, const size_t n_literals) {
    if (UNLIKELY(payload_size < ZXC_HUF_HEADER_SIZE || n_literals == 0))
        return ZXC_ERROR_CORRUPT_DATA;

    /* 1. Parse length header. */
    uint8_t code_len[ZXC_HUF_NUM_SYMBOLS];
    {
        const int rc = unpack_lengths_header(payload, code_len);
        if (UNLIKELY(rc != ZXC_OK)) return rc;
    }

    /* 2. Build the 2048-entry multi-symbol decode table. Cache-line
     * aligned: the LUT spans 128 lines (8 KB / 64 B) and is hammered every
     * symbol, landing it on a 64-byte boundary avoids any cross-line
     * load split on the per-iteration entry fetch. */
    ZXC_ALIGN(ZXC_CACHE_LINE_SIZE) zxc_huf_dec_entry_t table[ZXC_HUF_DEC_TABLE_SIZE];
    {
        const int rc = build_decode_table(code_len, table);
        if (UNLIKELY(rc != ZXC_OK)) return rc;
    }

    /* 3. Decode the 4 interleaved sub-streams. */
    return zxc_huf_decode_streams(payload + ZXC_HUF_TABLE_SIZE, payload_size - ZXC_HUF_TABLE_SIZE,
                                  dst, n_literals, table);
}

/**
 * @brief Decode a literal section that carries no lengths header, using a
 *        prebuilt decode table (shared dictionary table).
 *
 * @param[in]  payload       Section payload (6-byte sizes header + 4 sub-streams).
 * @param[in]  payload_size  Total payload length in bytes.
 * @param[out] dst           Destination buffer (must not alias @p payload).
 * @param[in]  n_literals    Expected number of decoded bytes.
 * @param[in]  table         Prebuilt @ref ZXC_HUF_DEC_TABLE_SIZE-entry decode table.
 * @return `ZXC_OK` on success, `ZXC_ERROR_CORRUPT_DATA` if @p table is NULL or
 *         the stream is malformed.
 */
int zxc_huf_decode_section_dict(const uint8_t* RESTRICT payload, const size_t payload_size,
                                uint8_t* RESTRICT dst, const size_t n_literals,
                                const zxc_huf_dec_entry_t* RESTRICT table) {
    if (UNLIKELY(table == NULL)) return ZXC_ERROR_CORRUPT_DATA;
    return zxc_huf_decode_streams(payload, payload_size, dst, n_literals, table);
}

/**
 * @brief Build the @ref ZXC_HUF_DEC_TABLE_SIZE-entry decode table from
 *        per-symbol code lengths. Validates Kraft equality.
 *
 * Public wrapper around @ref build_decode_table.
 *
 * @param[in]  code_len  Per-symbol code lengths.
 * @param[out] table     Destination decode table (caller-aligned).
 * @return `ZXC_OK` on success, `ZXC_ERROR_CORRUPT_DATA` on invalid lengths.
 */
int zxc_huf_build_dec_table(const uint8_t* RESTRICT code_len, zxc_huf_dec_entry_t* RESTRICT table) {
    return build_decode_table(code_len, table);
}

/**
 * @brief Pack per-symbol code lengths into the 128-byte (4-bit nibble) header.
 *
 * @param[in]  code_len  Per-symbol code lengths (one byte each).
 * @param[out] out       Destination 128-byte packed header.
 */
void zxc_huf_pack_lengths(const uint8_t* RESTRICT code_len, uint8_t* RESTRICT out) {
    pack_lengths_header(code_len, out);
}

/**
 * @brief Unpack and structurally validate a 128-byte packed lengths header.
 *
 * @param[in]  in        128-byte packed lengths header.
 * @param[out] code_len  Destination per-symbol code lengths.
 * @return `ZXC_OK` on success, `ZXC_ERROR_CORRUPT_DATA` on invalid lengths.
 */
int zxc_huf_unpack_lengths(const uint8_t* RESTRICT in, uint8_t* RESTRICT code_len) {
    return unpack_lengths_header(in, code_len);
}
