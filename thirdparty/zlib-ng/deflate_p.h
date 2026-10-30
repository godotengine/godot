/* deflate_p.h -- Private inline functions and macros shared with more than
 *                one deflate method
 *
 * Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 */

#ifndef DEFLATE_P_H
#define DEFLATE_P_H

#include "functable.h"

/* Forward declare common non-inlined functions declared in deflate.c */

#ifdef ZLIB_DEBUG
/* ===========================================================================
 * Check that the match at match_start is indeed a match.
 */
static inline void check_match(deflate_state *s, Pos start, Pos match, int length) {
    /* check that the match length is valid*/
    if (length < STD_MIN_MATCH || length > STD_MAX_MATCH) {
        fprintf(stderr, " start %u, match %u, length %d\n", start, match, length);
        z_error("invalid match length");
    }
    /* check that the match isn't at the same position as the start string */
    if (match == start) {
        fprintf(stderr, " start %u, match %u, length %d\n", start, match, length);
        z_error("invalid match position");
    }
    /* check that the match is indeed a match */
    if (memcmp(s->window + match, s->window + start, length) != 0) {
        int32_t i = 0;
        fprintf(stderr, " start %u, match %u, length %d\n", start, match, length);
        do {
            fprintf(stderr, "  %03d: match [%02x] start [%02x]\n", i++,
                s->window[match++], s->window[start++]);
        } while (--length != 0);
        z_error("invalid match");
    }
    if (z_verbose > 1) {
        fprintf(stderr, "\\[%u,%d]", start-match, length);
        do {
            putc(s->window[start++], stderr);
        } while (--length != 0);
    }
}
#else
#define check_match(s, start, match, length)
#endif

Z_INTERNAL void PREFIX(flush_pending)(PREFIX3(stream) *strm);

/* ===========================================================================
 * Save the match info and tally the frequency counts. Return true if
 * the current block must be flushed.
 */

extern const unsigned char Z_INTERNAL zng_length_code[];
extern const unsigned char Z_INTERNAL zng_dist_code[];

static inline int zng_tr_tally_lit(deflate_state *s, unsigned char c) {
    /* c is the unmatched char */
#ifdef LIT_MEM
    s->d_buf[s->sym_next] = 0;
    s->l_buf[s->sym_next++] = c;
#else
    s->sym_buf[s->sym_next++] = 0;
    s->sym_buf[s->sym_next++] = 0;
    s->sym_buf[s->sym_next++] = c;
#endif
    s->dyn_ltree[c].Freq++;
    Tracevv((stderr, "%c", c));
    Assert(c <= (STD_MAX_MATCH-STD_MIN_MATCH), "zng_tr_tally: bad literal");
    return (s->sym_next == s->sym_end);
}

static inline int zng_tr_tally_dist(deflate_state* s, uint32_t dist, uint32_t len) {
    /* dist: distance of matched string */
    /* len: match length-STD_MIN_MATCH */
#ifdef LIT_MEM
    Assert(dist <= UINT16_MAX, "dist should fit in uint16_t");
    Assert(len <= UINT8_MAX, "len should fit in uint8_t");
    s->d_buf[s->sym_next] = (uint16_t)dist;
    s->l_buf[s->sym_next++] = (uint8_t)len;
#else
    s->sym_buf[s->sym_next++] = (uint8_t)(dist);
    s->sym_buf[s->sym_next++] = (uint8_t)(dist >> 8);
    s->sym_buf[s->sym_next++] = (uint8_t)len;
#endif
    s->matches++;
    dist--;
    Assert(dist < MAX_DIST(s) && (uint16_t)d_code(dist) < (uint16_t)D_CODES,
        "zng_tr_tally: bad match");

    s->dyn_ltree[zng_length_code[len] + LITERALS + 1].Freq++;
    s->dyn_dtree[d_code(dist)].Freq++;
    return (s->sym_next == s->sym_end);
}

/* =========================================================================
 * Flush as much pending output as possible. All deflate() output, except for some
 * deflate_stored() output, goes through this function so some applications may wish to
 * modify it to avoid allocating a large strm->next_out buffer and copying into it.
 * See also read_buf().
 */
Z_FORCEINLINE static void flush_pending_inline(PREFIX3(stream) *strm) {
    uint32_t len;
    deflate_state *s = strm->state;

    zng_tr_flush_bits(s);
    len = MIN(s->pending, strm->avail_out);
    if (len == 0)
        return;

    Tracev((stderr, "[FLUSH]"));
    memcpy(strm->next_out, s->pending_out, len);
    strm->next_out  += len;
    s->pending_out  += len;
    strm->total_out += len;
    strm->avail_out -= len;
    s->pending      -= len;
    if (s->pending == 0)
        s->pending_out = s->pending_buf;
}

/* ===========================================================================
 * Reverse the first len bits of a code using bit manipulation
 */
static inline uint16_t bi_reverse(unsigned code, int len) {
    /* code: the value to invert */
    /* len: its bit length */
    Assert(len >= 1 && len <= 15, "code length must be 1-15");
#define bitrev8(b) \
    (uint8_t)((((uint8_t)(b) * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32)
    return (bitrev8(code >> 8) | (uint16_t)bitrev8(code) << 8) >> (16 - len);
}

/* ===========================================================================
 * Read a new buffer from the current input stream, update the adler32 and total number of
 * bytes read.  All deflate() input goes through this function so some applications may wish
 * to modify it to avoid allocating a large strm->next_in buffer and copying from it.
 * See also flush_pending_inline().
 */
Z_FORCEINLINE static unsigned read_buf(PREFIX3(stream) *strm, unsigned char *buf, unsigned size) {
    deflate_state *s = strm->state;
    uint32_t len = MIN(strm->avail_in, size);

    if (len == 0)
        return 0;

    if (!DEFLATE_NEED_CHECKSUM(strm)) {
        memcpy(buf, strm->next_in, len);
#ifdef GZIP
    } else if (s->wrap == 2) {
        FUNCTABLE_CALL(crc32_fold_copy)(&s->crc_fold, buf, strm->next_in, len);
#endif
    } else if (s->wrap == 1) {
        strm->adler = FUNCTABLE_CALL(adler32_fold_copy)(strm->adler, buf, strm->next_in, len);
    } else {
        memcpy(buf, strm->next_in, len);
    }

    strm->avail_in -= len;
    strm->next_in  += len;
    strm->total_in += len;
    return len;
}

/* ===========================================================================
 * Flush the current block, with given end-of-file flag.
 * IN assertion: strstart is set to the end of the current match.
 */
#define FLUSH_BLOCK_ONLY(s, last) { \
    zng_tr_flush_block(s, (s->block_start >= 0 ? \
                   (char *)&s->window[(unsigned)s->block_start] : \
                   NULL), \
                   (uint32_t)((int)s->strstart - s->block_start), \
                   (last)); \
    s->block_start = (int)s->strstart; \
    PREFIX(flush_pending)(s->strm); \
}

/* Same but force premature exit if necessary. */
#define FLUSH_BLOCK(s, last) { \
    FLUSH_BLOCK_ONLY(s, last); \
    if (s->strm->avail_out == 0) return (last) ? finish_started : need_more; \
}

/* Maximum stored block length in deflate format (not including header). */
#define MAX_STORED 65535

/* Compression function. Returns the block state after the call. */
typedef block_state (*compress_func) (deflate_state *s, int flush);
/* Match function. Returns the longest match. */
typedef uint32_t    (*match_func)    (deflate_state *const s, Pos cur_match);

#endif
