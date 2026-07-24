#ifndef TREES_EMIT_H_
#define TREES_EMIT_H_

#include "zbuild.h"
#include "trees.h"

#ifdef ZLIB_DEBUG
#  include <ctype.h>
#  include <inttypes.h>
#endif


/* trees.h */
extern Z_INTERNAL const ct_data static_ltree[L_CODES+2];
extern Z_INTERNAL const ct_data static_dtree[D_CODES];

extern const unsigned char Z_INTERNAL zng_dist_code[DIST_CODE_LEN];
extern const unsigned char Z_INTERNAL zng_length_code[STD_MAX_MATCH-STD_MIN_MATCH+1];

extern Z_INTERNAL const int base_length[LENGTH_CODES];
extern Z_INTERNAL const int base_dist[D_CODES];

/* Bit buffer and deflate code stderr tracing */
#ifdef ZLIB_DEBUG
#  define send_bits_trace(s, value, length) { \
        Tracevv((stderr, " l %2d v %4llx ", (int)(length), (long long)(value))); \
        Assert(length > 0 && length <= BIT_BUF_SIZE, "invalid length"); \
    }
#  define send_code_trace(s, c) \
    if (z_verbose > 2) { \
        fprintf(stderr, "\ncd %3d ", (c)); \
    }
#else
#  define send_bits_trace(s, value, length)
#  define send_code_trace(s, c)
#endif

/* If not enough room in bi_buf, use (valid) bits from bi_buf and
 * (64 - bi_valid) bits from value, leaving (width - (64-bi_valid))
 * unused bits in value.
 *
 * NOTE: Static analyzers can't evaluate value of total_bits, so we
 *       also need to make sure bi_valid is within acceptable range,
 *       otherwise the shifts will overflow.
 */
#define send_bits(s, t_val, t_len, bi_buf, bi_valid) {\
    uint64_t val = (uint64_t)t_val;\
    uint32_t len = (uint32_t)t_len;\
    uint32_t total_bits = bi_valid + len;\
    send_bits_trace(s, val, len);\
    sent_bits_add(s, len);\
    if (total_bits < BIT_BUF_SIZE && bi_valid < BIT_BUF_SIZE) {\
        bi_buf |= val << bi_valid;\
        bi_valid = total_bits;\
    } else if (bi_valid >= BIT_BUF_SIZE) {\
        put_uint64(s, bi_buf);\
        bi_buf = val;\
        bi_valid = len;\
    } else {\
        bi_buf |= val << bi_valid;\
        put_uint64(s, bi_buf);\
        bi_buf = val >> (BIT_BUF_SIZE - bi_valid);\
        bi_valid = total_bits - BIT_BUF_SIZE;\
    }\
}

/* Send a code of the given tree. c and tree must not have side effects */
#ifdef ZLIB_DEBUG
#  define send_code(s, c, tree, bi_buf, bi_valid) { \
    send_code_trace(s, c); \
    send_bits(s, tree[c].Code, tree[c].Len, bi_buf, bi_valid); \
}
#else
#  define send_code(s, c, tree, bi_buf, bi_valid) \
    send_bits(s, tree[c].Code, tree[c].Len, bi_buf, bi_valid)
#endif

/* ===========================================================================
 * Flush the bit buffer and align the output on a byte boundary
 */
static void bi_windup(deflate_state *s) {
    if (s->bi_valid > 56) {
        put_uint64(s, s->bi_buf);
    } else {
        if (s->bi_valid > 24) {
            put_uint32(s, (uint32_t)s->bi_buf);
            s->bi_buf >>= 32;
            s->bi_valid -= 32;
        }
        if (s->bi_valid > 8) {
            put_short(s, (uint16_t)s->bi_buf);
            s->bi_buf >>= 16;
            s->bi_valid -= 16;
        }
        if (s->bi_valid > 0) {
            put_byte(s, s->bi_buf);
        }
    }
    s->bi_buf = 0;
    s->bi_valid = 0;
}

/* ===========================================================================
 * Emit literal code
 */
static inline uint32_t zng_emit_lit(deflate_state *s, const ct_data *ltree, unsigned c) {
    uint32_t bi_valid = s->bi_valid;
    uint64_t bi_buf = s->bi_buf;

    send_code(s, c, ltree, bi_buf, bi_valid);

    s->bi_valid = bi_valid;
    s->bi_buf = bi_buf;

    Tracecv(isgraph(c & 0xff), (stderr, " '%c' ", c));

    return ltree[c].Len;
}

/* ===========================================================================
 * Emit match distance/length code
 */
static inline uint32_t zng_emit_dist(deflate_state *s, const ct_data *ltree, const ct_data *dtree,
    uint32_t lc, uint32_t dist) {
    uint32_t c, extra;
    uint8_t code;
    uint64_t match_bits;
    uint32_t match_bits_len;
    uint32_t bi_valid = s->bi_valid;
    uint64_t bi_buf = s->bi_buf;

    /* Send the length code, len is the match length - STD_MIN_MATCH */
    code = zng_length_code[lc];
    c = code+LITERALS+1;
    Assert(c < L_CODES, "bad l_code");
    send_code_trace(s, c);

    match_bits = ltree[c].Code;
    match_bits_len = ltree[c].Len;
    extra = extra_lbits[code];
    if (extra != 0) {
        lc -= base_length[code];
        match_bits |= ((uint64_t)lc << match_bits_len);
        match_bits_len += extra;
    }

    dist--; /* dist is now the match distance - 1 */
    code = d_code(dist);
    Assert(code < D_CODES, "bad d_code");
    send_code_trace(s, code);

    /* Send the distance code */
    match_bits |= ((uint64_t)dtree[code].Code << match_bits_len);
    match_bits_len += dtree[code].Len;
    extra = extra_dbits[code];
    if (extra != 0) {
        dist -= base_dist[code];
        match_bits |= ((uint64_t)dist << match_bits_len);
        match_bits_len += extra;
    }

    send_bits(s, match_bits, match_bits_len, bi_buf, bi_valid);

    s->bi_valid = bi_valid;
    s->bi_buf = bi_buf;

    return match_bits_len;
}

/* ===========================================================================
 * Emit end block
 */
static inline void zng_emit_end_block(deflate_state *s, const ct_data *ltree, const int last) {
    uint32_t bi_valid = s->bi_valid;
    uint64_t bi_buf = s->bi_buf;
    send_code(s, END_BLOCK, ltree, bi_buf, bi_valid);
    s->bi_valid = bi_valid;
    s->bi_buf = bi_buf;
    Tracev((stderr, "\n+++ Emit End Block: Last: %u Pending: %u Total Out: %" PRIu64 "\n",
        last, s->pending, (uint64_t)s->strm->total_out));
    Z_UNUSED(last);
}

/* ===========================================================================
 * Emit literal and count bits
 */
static inline void zng_tr_emit_lit(deflate_state *s, const ct_data *ltree, unsigned c) {
    cmpr_bits_add(s, zng_emit_lit(s, ltree, c));
}

/* ===========================================================================
 * Emit match and count bits
 */
static inline void zng_tr_emit_dist(deflate_state *s, const ct_data *ltree, const ct_data *dtree,
    uint32_t lc, uint32_t dist) {
    cmpr_bits_add(s, zng_emit_dist(s, ltree, dtree, lc, dist));
}

/* ===========================================================================
 * Emit start of block
 */
static inline void zng_tr_emit_tree(deflate_state *s, int type, const int last) {
    uint32_t bi_valid = s->bi_valid;
    uint64_t bi_buf = s->bi_buf;
    uint32_t header_bits = (type << 1) + last;
    send_bits(s, header_bits, 3, bi_buf, bi_valid);
    cmpr_bits_add(s, 3);
    s->bi_valid = bi_valid;
    s->bi_buf = bi_buf;
    Tracev((stderr, "\n--- Emit Tree: Last: %u\n", last));
}

/* ===========================================================================
 * Align bit buffer on a byte boundary and count bits
 */
static inline void zng_tr_emit_align(deflate_state *s) {
    bi_windup(s); /* align on byte boundary */
    sent_bits_align(s);
}

/* ===========================================================================
 * Emit an end block and align bit buffer if last block
 */
static inline void zng_tr_emit_end_block(deflate_state *s, const ct_data *ltree, const int last) {
    zng_emit_end_block(s, ltree, last);
    cmpr_bits_add(s, 7);
    if (last)
        zng_tr_emit_align(s);
}

#endif
