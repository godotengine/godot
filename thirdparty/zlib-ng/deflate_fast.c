/* deflate_fast.c -- compress data using the fast strategy of deflation algorithm
 *
 * Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "deflate.h"
#include "deflate_p.h"
#include "functable.h"

/* ===========================================================================
 * Compress as much as possible from the input stream, return the current
 * block state.
 * This function does not perform lazy evaluation of matches and inserts
 * new strings in the dictionary only for unmatched strings or for short
 * matches. It is used only for the fast compression options.
 */
Z_INTERNAL block_state deflate_fast(deflate_state *s, int flush) {
    Pos hash_head;        /* head of the hash chain */
    int bflush = 0;       /* set if current block must be flushed */
    int64_t dist;
    uint32_t match_len = 0;

    for (;;) {
        /* Make sure that we always have enough lookahead, except
         * at the end of the input file. We need STD_MAX_MATCH bytes
         * for the next match, plus WANT_MIN_MATCH bytes to insert the
         * string following the next match.
         */
        if (s->lookahead < MIN_LOOKAHEAD) {
            PREFIX(fill_window)(s);
            if (UNLIKELY(s->lookahead < MIN_LOOKAHEAD && flush == Z_NO_FLUSH)) {
                return need_more;
            }
            if (UNLIKELY(s->lookahead == 0))
                break; /* flush the current block */
        }

        /* Insert the string window[strstart .. strstart+2] in the
         * dictionary, and set hash_head to the head of the hash chain:
         */
        if (s->lookahead >= WANT_MIN_MATCH) {
            hash_head = quick_insert_string(s, s->strstart);
            dist = (int64_t)s->strstart - hash_head;

            /* Find the longest match, discarding those <= prev_length.
             * At this point we have always match length < WANT_MIN_MATCH
             */
            if (dist <= MAX_DIST(s) && dist > 0 && hash_head != 0) {
                /* To simplify the code, we prevent matches with the string
                 * of window index 0 (in particular we have to avoid a match
                 * of the string with itself at the start of the input file).
                 */
                match_len = FUNCTABLE_CALL(longest_match)(s, hash_head);
                /* longest_match() sets match_start */
            }
        }

        if (match_len >= WANT_MIN_MATCH) {
            Assert(s->strstart <= UINT16_MAX, "strstart should fit in uint16_t");
            Assert(s->match_start <= UINT16_MAX, "match_start should fit in uint16_t");
            check_match(s, (Pos)s->strstart, (Pos)s->match_start, match_len);

            bflush = zng_tr_tally_dist(s, s->strstart - s->match_start, match_len - STD_MIN_MATCH);

            s->lookahead -= match_len;

            /* Insert new strings in the hash table only if the match length
             * is not too large. This saves time but degrades compression.
             */
            if (match_len <= s->max_insert_length && s->lookahead >= WANT_MIN_MATCH) {
                match_len--; /* string at strstart already in table */
                s->strstart++;

                insert_string(s, s->strstart, match_len);
                s->strstart += match_len;
            } else {
                s->strstart += match_len;
                quick_insert_string(s, s->strstart + 2 - STD_MIN_MATCH);

                /* If lookahead < STD_MIN_MATCH, ins_h is garbage, but it does not
                 * matter since it will be recomputed at next deflate call.
                 */
            }
            match_len = 0;
        } else {
            /* No match, output a literal byte */
            bflush = zng_tr_tally_lit(s, s->window[s->strstart]);
            s->lookahead--;
            s->strstart++;
        }
        if (UNLIKELY(bflush))
            FLUSH_BLOCK(s, 0);
    }
    s->insert = s->strstart < (STD_MIN_MATCH - 1) ? s->strstart : (STD_MIN_MATCH - 1);
    if (UNLIKELY(flush == Z_FINISH)) {
        FLUSH_BLOCK(s, 1);
        return finish_done;
    }
    if (UNLIKELY(s->sym_next))
        FLUSH_BLOCK(s, 0);
    return block_done;
}
