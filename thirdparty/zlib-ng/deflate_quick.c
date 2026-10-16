/*
 * The deflate_quick deflate strategy, designed to be used when cycles are
 * at a premium.
 *
 * Copyright (C) 2013 Intel Corporation. All rights reserved.
 * Authors:
 *  Wajdi Feghali   <wajdi.k.feghali@intel.com>
 *  Jim Guilford    <james.guilford@intel.com>
 *  Vinodh Gopal    <vinodh.gopal@intel.com>
 *     Erdinc Ozturk   <erdinc.ozturk@intel.com>
 *  Jim Kukunas     <james.t.kukunas@linux.intel.com>
 *
 * Portions are Copyright (C) 2016 12Sided Technology, LLC.
 * Author:
 *  Phil Vachon     <pvachon@12sidedtech.com>
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zmemory.h"
#include "deflate.h"
#include "deflate_p.h"
#include "functable.h"
#include "trees_emit.h"

extern const ct_data static_ltree[L_CODES+2];
extern const ct_data static_dtree[D_CODES];

#define QUICK_START_BLOCK(s, last) { \
    zng_tr_emit_tree(s, STATIC_TREES, last); \
    s->block_open = 1 + (int)last; \
    s->block_start = (int)s->strstart; \
}

#define QUICK_END_BLOCK(s, last) { \
    if (s->block_open) { \
        zng_tr_emit_end_block(s, static_ltree, last); \
        s->block_open = 0; \
        s->block_start = (int)s->strstart; \
        PREFIX(flush_pending)(s->strm); \
        if (s->strm->avail_out == 0) \
            return (last) ? finish_started : need_more; \
    } \
}

Z_INTERNAL block_state deflate_quick(deflate_state *s, int flush) {
    Pos hash_head;
    int64_t dist;
    unsigned match_len, last;


    last = (flush == Z_FINISH) ? 1 : 0;
    if (UNLIKELY(last && s->block_open != 2)) {
        /* Emit end of previous block */
        QUICK_END_BLOCK(s, 0);
        /* Emit start of last block */
        QUICK_START_BLOCK(s, last);
    } else if (UNLIKELY(s->block_open == 0 && s->lookahead > 0)) {
        /* Start new block only when we have lookahead data, so that if no
           input data is given an empty block will not be written */
        QUICK_START_BLOCK(s, last);
    }

    for (;;) {
        if (UNLIKELY(s->pending + ((BIT_BUF_SIZE + 7) >> 3) >= s->pending_buf_size)) {
            PREFIX(flush_pending)(s->strm);
            if (s->strm->avail_out == 0) {
                return (last && s->strm->avail_in == 0 && s->bi_valid == 0 && s->block_open == 0) ? finish_started : need_more;
            }
        }

        if (UNLIKELY(s->lookahead < MIN_LOOKAHEAD)) {
            PREFIX(fill_window)(s);
            if (UNLIKELY(s->lookahead < MIN_LOOKAHEAD && flush == Z_NO_FLUSH)) {
                return need_more;
            }
            if (UNLIKELY(s->lookahead == 0))
                break;

            if (UNLIKELY(s->block_open == 0)) {
                /* Start new block when we have lookahead data, so that if no
                   input data is given an empty block will not be written */
                QUICK_START_BLOCK(s, last);
            }
        }

        if (LIKELY(s->lookahead >= WANT_MIN_MATCH)) {
            hash_head = quick_insert_string(s, s->strstart);
            dist = (int64_t)s->strstart - hash_head;

            if (dist <= MAX_DIST(s) && dist > 0) {
                const uint8_t *str_start = s->window + s->strstart;
                const uint8_t *match_start = s->window + hash_head;

                if (zng_memcmp_2(str_start, match_start) == 0) {
                    match_len = FUNCTABLE_CALL(compare256)(str_start+2, match_start+2) + 2;

                    if (match_len >= WANT_MIN_MATCH) {
                        if (UNLIKELY(match_len > s->lookahead))
                            match_len = s->lookahead;
                        if (UNLIKELY(match_len > STD_MAX_MATCH))
                            match_len = STD_MAX_MATCH;

                        Assert(s->strstart <= UINT16_MAX, "strstart should fit in uint16_t");
                        check_match(s, (Pos)s->strstart, hash_head, match_len);

                        zng_tr_emit_dist(s, static_ltree, static_dtree, match_len - STD_MIN_MATCH, (uint32_t)dist);
                        s->lookahead -= match_len;
                        s->strstart += match_len;
                        continue;
                    }
                }
            }
        }

        zng_tr_emit_lit(s, static_ltree, s->window[s->strstart]);
        s->strstart++;
        s->lookahead--;
    }

    s->insert = s->strstart < (STD_MIN_MATCH - 1) ? s->strstart : (STD_MIN_MATCH - 1);
    if (UNLIKELY(last)) {
        QUICK_END_BLOCK(s, 1);
        return finish_done;
    }

    QUICK_END_BLOCK(s, 0);
    return block_done;
}
