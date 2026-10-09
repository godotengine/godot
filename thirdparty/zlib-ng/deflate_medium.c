/* deflate_medium.c -- The deflate_medium deflate strategy
 *
 * Copyright (C) 2013 Intel Corporation. All rights reserved.
 * Authors:
 *  Arjan van de Ven    <arjan@linux.intel.com>
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 */
#ifndef NO_MEDIUM_STRATEGY
#include "zbuild.h"
#include "deflate.h"
#include "deflate_p.h"
#include "functable.h"

struct match {
    uint16_t match_start;
    uint16_t match_length;
    uint16_t strstart;
    uint16_t orgstart;
};

static int emit_match(deflate_state *s, struct match match) {
    int bflush = 0;

    /* matches that are not long enough we need to emit as literals */
    if (match.match_length < WANT_MIN_MATCH) {
        while (match.match_length) {
            bflush += zng_tr_tally_lit(s, s->window[match.strstart]);
            s->lookahead--;
            match.strstart++;
            match.match_length--;
        }
        return bflush;
    }

    check_match(s, match.strstart, match.match_start, match.match_length);

    bflush += zng_tr_tally_dist(s, match.strstart - match.match_start, match.match_length - STD_MIN_MATCH);

    s->lookahead -= match.match_length;
    return bflush;
}

static void insert_match(deflate_state *s, struct match match) {
    if (UNLIKELY(s->lookahead <= (unsigned int)(match.match_length + WANT_MIN_MATCH)))
        return;

    /* matches that are not long enough we need to emit as literals */
    if (LIKELY(match.match_length < WANT_MIN_MATCH)) {
        match.strstart++;
        match.match_length--;
        if (UNLIKELY(match.match_length > 0)) {
            if (match.strstart >= match.orgstart) {
                if (match.strstart + match.match_length - 1 >= match.orgstart) {
                    insert_string(s, match.strstart, match.match_length);
                } else {
                    insert_string(s, match.strstart, match.orgstart - match.strstart + 1);
                }
                match.strstart += match.match_length;
                match.match_length = 0;
            }
        }
        return;
    }

    /* Insert new strings in the hash table only if the match length
     * is not too large. This saves time but degrades compression.
     */
    if (match.match_length <= 16 * s->max_insert_length && s->lookahead >= WANT_MIN_MATCH) {
        match.match_length--; /* string at strstart already in table */
        match.strstart++;

        if (LIKELY(match.strstart >= match.orgstart)) {
            if (LIKELY(match.strstart + match.match_length - 1 >= match.orgstart)) {
                insert_string(s, match.strstart, match.match_length);
            } else {
                insert_string(s, match.strstart, match.orgstart - match.strstart + 1);
            }
        } else if (match.orgstart < match.strstart + match.match_length) {
            insert_string(s, match.orgstart, match.strstart + match.match_length - match.orgstart);
        }
        match.strstart += match.match_length;
        match.match_length = 0;
    } else {
        match.strstart += match.match_length;
        match.match_length = 0;

        if (match.strstart >= (STD_MIN_MATCH - 2))
            quick_insert_string(s, match.strstart + 2 - STD_MIN_MATCH);

        /* If lookahead < WANT_MIN_MATCH, ins_h is garbage, but it does not
         * matter since it will be recomputed at next deflate call.
         */
    }
}

static void fizzle_matches(deflate_state *s, struct match *current, struct match *next) {
    Pos limit;
    unsigned char *match, *orig;
    int changed = 0;
    struct match c, n;
    /* step zero: sanity checks */

    if (current->match_length <= 1)
        return;

    if (UNLIKELY(current->match_length > 1 + next->match_start))
        return;

    if (UNLIKELY(current->match_length > 1 + next->strstart))
        return;

    match = s->window - current->match_length + 1 + next->match_start;
    orig  = s->window - current->match_length + 1 + next->strstart;

    /* quick exit check.. if this fails then don't bother with anything else */
    if (LIKELY(*match != *orig))
        return;

    c = *current;
    n = *next;

    /* step one: try to move the "next" match to the left as much as possible */
    limit = next->strstart > MAX_DIST(s) ? next->strstart - (Pos)MAX_DIST(s) : 0;

    match = s->window + n.match_start - 1;
    orig = s->window + n.strstart - 1;

    while (*match == *orig) {
        if (UNLIKELY(c.match_length < 1))
            break;
        if (UNLIKELY(n.strstart <= limit))
            break;
        if (UNLIKELY(n.match_length >= 256))
            break;
        if (UNLIKELY(n.match_start <= 1))
            break;

        n.strstart--;
        n.match_start--;
        n.match_length++;
        c.match_length--;
        match--;
        orig--;
        changed++;
    }

    if (!changed)
        return;

    if (c.match_length <= 1 && n.match_length != 2) {
        n.orgstart++;
        *current = c;
        *next = n;
    } else {
        return;
    }
}

Z_INTERNAL block_state deflate_medium(deflate_state *s, int flush) {
    /* Align the first struct to start on a new cacheline, this allows us to fit both structs in one cacheline */
    ALIGNED_(16) struct match current_match;
                 struct match next_match;

    /* For levels below 5, don't check the next position for a better match */
    int early_exit = s->level < 5;

    memset(&current_match, 0, sizeof(struct match));
    memset(&next_match, 0, sizeof(struct match));

    for (;;) {
        Pos hash_head = 0;    /* head of the hash chain */
        int bflush = 0;       /* set if current block must be flushed */
        int64_t dist;

        /* Make sure that we always have enough lookahead, except
         * at the end of the input file. We need STD_MAX_MATCH bytes
         * for the next match, plus WANT_MIN_MATCH bytes to insert the
         * string following the next current_match.
         */
        if (s->lookahead < MIN_LOOKAHEAD) {
            PREFIX(fill_window)(s);
            if (s->lookahead < MIN_LOOKAHEAD && flush == Z_NO_FLUSH) {
                return need_more;
            }
            if (UNLIKELY(s->lookahead == 0))
                break; /* flush the current block */
            next_match.match_length = 0;
        }

        /* Insert the string window[strstart .. strstart+2] in the
         * dictionary, and set hash_head to the head of the hash chain:
         */

        /* If we already have a future match from a previous round, just use that */
        if (!early_exit && next_match.match_length > 0) {
            current_match = next_match;
            next_match.match_length = 0;
        } else {
            hash_head = 0;
            if (s->lookahead >= WANT_MIN_MATCH) {
                hash_head = quick_insert_string(s, s->strstart);
            }

            current_match.strstart = (uint16_t)s->strstart;
            current_match.orgstart = current_match.strstart;

            /* Find the longest match, discarding those <= prev_length.
             * At this point we have always match_length < WANT_MIN_MATCH
             */

            dist = (int64_t)s->strstart - hash_head;
            if (dist <= MAX_DIST(s) && dist > 0 && hash_head != 0) {
                /* To simplify the code, we prevent matches with the string
                 * of window index 0 (in particular we have to avoid a match
                 * of the string with itself at the start of the input file).
                 */
                current_match.match_length = (uint16_t)FUNCTABLE_CALL(longest_match)(s, hash_head);
                current_match.match_start = (uint16_t)s->match_start;
                if (UNLIKELY(current_match.match_length < WANT_MIN_MATCH))
                    current_match.match_length = 1;
                if (UNLIKELY(current_match.match_start >= current_match.strstart)) {
                    /* this can happen due to some restarts */
                    current_match.match_length = 1;
                }
            } else {
                /* Set up the match to be a 1 byte literal */
                current_match.match_start = 0;
                current_match.match_length = 1;
            }
        }

        insert_match(s, current_match);

        /* now, look ahead one */
        if (LIKELY(!early_exit && s->lookahead > MIN_LOOKAHEAD && (uint32_t)(current_match.strstart + current_match.match_length) < (s->window_size - MIN_LOOKAHEAD))) {
            s->strstart = current_match.strstart + current_match.match_length;
            hash_head = quick_insert_string(s, s->strstart);

            next_match.strstart = (uint16_t)s->strstart;
            next_match.orgstart = next_match.strstart;

            /* Find the longest match, discarding those <= prev_length.
             * At this point we have always match_length < WANT_MIN_MATCH
             */

            dist = (int64_t)s->strstart - hash_head;
            if (dist <= MAX_DIST(s) && dist > 0 && hash_head != 0) {
                /* To simplify the code, we prevent matches with the string
                 * of window index 0 (in particular we have to avoid a match
                 * of the string with itself at the start of the input file).
                 */
                next_match.match_length = (uint16_t)FUNCTABLE_CALL(longest_match)(s, hash_head);
                next_match.match_start = (uint16_t)s->match_start;
                if (UNLIKELY(next_match.match_start >= next_match.strstart)) {
                    /* this can happen due to some restarts */
                    next_match.match_length = 1;
                }
                if (next_match.match_length < WANT_MIN_MATCH)
                    next_match.match_length = 1;
                else
                    fizzle_matches(s, &current_match, &next_match);
            } else {
                /* Set up the match to be a 1 byte literal */
                next_match.match_start = 0;
                next_match.match_length = 1;
            }

            s->strstart = current_match.strstart;
        } else {
            next_match.match_length = 0;
        }

        /* now emit the current match */
        bflush = emit_match(s, current_match);

        /* move the "cursor" forward */
        s->strstart += current_match.match_length;

        if (UNLIKELY(bflush))
            FLUSH_BLOCK(s, 0);
    }
    s->insert = s->strstart < (STD_MIN_MATCH - 1) ? s->strstart : (STD_MIN_MATCH - 1);
    if (flush == Z_FINISH) {
        FLUSH_BLOCK(s, 1);
        return finish_done;
    }
    if (UNLIKELY(s->sym_next))
        FLUSH_BLOCK(s, 0);

    return block_done;
}
#endif
