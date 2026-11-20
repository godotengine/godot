/* deflate_slow.c -- compress data using the slow strategy of deflation algorithm
 *
 * Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "deflate.h"
#include "deflate_p.h"
#include "functable.h"

/* ===========================================================================
 * Same as deflate_medium, but achieves better compression. We use a lazy
 * evaluation for matches: a match is finally adopted only if there is
 * no better match at the next window position.
 */
Z_INTERNAL block_state deflate_slow(deflate_state *s, int flush) {
    Pos hash_head;           /* head of hash chain */
    int bflush;              /* set if current block must be flushed */
    int64_t dist;
    uint32_t match_len;
    match_func longest_match;

    if (s->max_chain_length <= 1024)
        longest_match = FUNCTABLE_FPTR(longest_match);
    else
        longest_match = FUNCTABLE_FPTR(longest_match_slow);

    /* Process the input block. */
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
        hash_head = 0;
        if (LIKELY(s->lookahead >= WANT_MIN_MATCH)) {
            hash_head = s->quick_insert_string(s, s->strstart);
        }

        /* Find the longest match, discarding those <= prev_length.
         */
        s->prev_match = (Pos)s->match_start;
        match_len = STD_MIN_MATCH - 1;
        dist = (int64_t)s->strstart - hash_head;

        if (dist <= MAX_DIST(s) && dist > 0 && s->prev_length < s->max_lazy_match && hash_head != 0) {
            /* To simplify the code, we prevent matches with the string
             * of window index 0 (in particular we have to avoid a match
             * of the string with itself at the start of the input file).
             */
            match_len = longest_match(s, hash_head);
            /* longest_match() sets match_start */

            if (match_len <= 5 && (s->strategy == Z_FILTERED)) {
                /* If prev_match is also WANT_MIN_MATCH, match_start is garbage
                 * but we will ignore the current match anyway.
                 */
                match_len = STD_MIN_MATCH - 1;
            }
        }
        /* If there was a match at the previous step and the current
         * match is not better, output the previous match:
         */
        if (s->prev_length >= STD_MIN_MATCH && match_len <= s->prev_length) {
            unsigned int max_insert = s->strstart + s->lookahead - STD_MIN_MATCH;
            /* Do not insert strings in hash table beyond this. */

            Assert((s->strstart-1) <= UINT16_MAX, "strstart-1 should fit in uint16_t");
            check_match(s, (Pos)(s->strstart - 1), s->prev_match, s->prev_length);

            bflush = zng_tr_tally_dist(s, s->strstart -1 - s->prev_match, s->prev_length - STD_MIN_MATCH);

            /* Insert in hash table all strings up to the end of the match.
             * strstart-1 and strstart are already inserted. If there is not
             * enough lookahead, the last two strings are not inserted in
             * the hash table.
             */
            s->prev_length -= 1;
            s->lookahead -= s->prev_length;

            unsigned int mov_fwd = s->prev_length - 1;
            if (max_insert > s->strstart) {
                unsigned int insert_cnt = mov_fwd;
                if (UNLIKELY(insert_cnt > max_insert - s->strstart))
                    insert_cnt = max_insert - s->strstart;
                s->insert_string(s, s->strstart + 1, insert_cnt);
            }
            s->prev_length = 0;
            s->match_available = 0;
            s->strstart += mov_fwd + 1;

            if (UNLIKELY(bflush))
                FLUSH_BLOCK(s, 0);

        } else if (s->match_available) {
            /* If there was no match at the previous position, output a
             * single literal. If there was a match but the current match
             * is longer, truncate the previous match to a single literal.
             */
            bflush = zng_tr_tally_lit(s, s->window[s->strstart-1]);
            if (UNLIKELY(bflush))
                FLUSH_BLOCK_ONLY(s, 0);
            s->prev_length = match_len;
            s->strstart++;
            s->lookahead--;
            if (UNLIKELY(s->strm->avail_out == 0))
                return need_more;
        } else {
            /* There is no previous match to compare with, wait for
             * the next step to decide.
             */
            s->prev_length = match_len;
            s->match_available = 1;
            s->strstart++;
            s->lookahead--;
        }
    }
    Assert(flush != Z_NO_FLUSH, "no flush?");
    if (UNLIKELY(s->match_available)) {
        Z_UNUSED(zng_tr_tally_lit(s, s->window[s->strstart-1]));
        s->match_available = 0;
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
