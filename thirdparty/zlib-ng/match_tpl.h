/* match_tpl.h -- find longest match template for compare256 variants
 *
 * Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 * Portions copyright (C) 2014-2021 Konstantin Nosov
 *  Fast-zlib optimized longest_match
 *  https://github.com/gildor2/fast_zlib
 */

#ifndef MATCH_TPL_H
#define MATCH_TPL_H

#define EARLY_EXIT_TRIGGER_LEVEL 5

#endif

/* Set match_start to the longest match starting at the given string and
 * return its length. Matches shorter or equal to prev_length are discarded,
 * in which case the result is equal to prev_length and match_start is garbage.
 *
 * IN assertions: cur_match is the head of the hash chain for the current
 * string (strstart) and its distance is <= MAX_DIST, and prev_length >=1
 * OUT assertion: the match length is not greater than s->lookahead
 *
 * The LONGEST_MATCH_SLOW variant spends more time to attempt to find longer
 * matches once a match has already been found.
 */
Z_INTERNAL uint32_t LONGEST_MATCH(deflate_state *const s, Pos cur_match) {
    unsigned int strstart = s->strstart;
    const unsigned wmask = s->w_mask;
    unsigned char *window = s->window;
    unsigned char *scan = window + strstart;
    Z_REGISTER unsigned char *mbase_start = window;
    Z_REGISTER unsigned char *mbase_end;
    const Pos *prev = s->prev;
    Pos limit;
#ifdef LONGEST_MATCH_SLOW
    Pos limit_base;
#else
    int32_t early_exit;
#endif
    uint32_t chain_length, nice_match, best_len, offset;
    uint32_t lookahead = s->lookahead;
    Pos match_offset = 0;
    uint64_t scan_start;
    uint64_t scan_end;

#define GOTO_NEXT_CHAIN \
    if (--chain_length && (cur_match = prev[cur_match & wmask]) > limit) \
        continue; \
    return best_len;

    /* The code is optimized for STD_MAX_MATCH-2 multiple of 16. */
    Assert(STD_MAX_MATCH == 258, "Code too clever");

    best_len = s->prev_length ? s->prev_length : STD_MIN_MATCH-1;

    /* Calculate read offset which should only extend an extra byte
     * to find the next best match length.
     */
    offset = best_len-1;
    if (best_len >= sizeof(uint32_t)) {
        offset -= 2;
        if (best_len >= sizeof(uint64_t))
            offset -= 4;
    }

    scan_start = zng_memread_8(scan);
    scan_end = zng_memread_8(scan+offset);
    mbase_end  = (mbase_start+offset);

    /* Do not waste too much time if we already have a good match */
    chain_length = s->max_chain_length;
    if (best_len >= s->good_match)
        chain_length >>= 2;
    nice_match = (uint32_t)s->nice_match;

    /* Stop when cur_match becomes <= limit. To simplify the code,
     * we prevent matches with the string of window index 0
     */
    limit = strstart > MAX_DIST(s) ? (Pos)(strstart - MAX_DIST(s)) : 0;
#ifdef LONGEST_MATCH_SLOW
    limit_base = limit;
    if (best_len >= STD_MIN_MATCH) {
        /* We're continuing search (lazy evaluation). */
        uint32_t i, hash;
        Pos pos;

        /* Find a most distant chain starting from scan with index=1 (index=0 corresponds
         * to cur_match). We cannot use s->prev[strstart+1,...] immediately, because
         * these strings are not yet inserted into the hash table.
         */
        hash = s->update_hash(0, scan[1]);
        hash = s->update_hash(hash, scan[2]);

        for (i = 3; i <= best_len; i++) {
            hash = s->update_hash(hash, scan[i]);

            /* If we're starting with best_len >= 3, we can use offset search. */
            pos = s->head[hash];
            if (pos < cur_match) {
                match_offset = (Pos)(i - 2);
                cur_match = pos;
            }
        }

        /* Update offset-dependent variables */
        limit = limit_base+match_offset;
        if (cur_match <= limit)
            goto break_matching;
        mbase_start -= match_offset;
        mbase_end -= match_offset;
    }
#else
    early_exit = s->level < EARLY_EXIT_TRIGGER_LEVEL;
#endif
    Assert((unsigned long)strstart <= s->window_size - MIN_LOOKAHEAD, "need lookahead");
    for (;;) {
        if (cur_match >= strstart)
            break;

        /* Skip to next match if the match length cannot increase or if the match length is
         * less than 2. Note that the checks below for insufficient lookahead only occur
         * occasionally for performance reasons.
         * Therefore uninitialized memory will be accessed and conditional jumps will be made
         * that depend on those values. However the length of the match is limited to the
         * lookahead, so the output of deflate is not affected by the uninitialized values.
         */
        if (best_len < sizeof(uint32_t)) {
            for (;;) {
                if (zng_memcmp_2(mbase_end+cur_match, &scan_end) == 0 &&
                    zng_memcmp_2(mbase_start+cur_match, &scan_start) == 0)
                    break;
                GOTO_NEXT_CHAIN;
            }
        } else if (best_len >= sizeof(uint64_t)) {
            for (;;) {
                if (zng_memcmp_8(mbase_end+cur_match, &scan_end) == 0 &&
                    zng_memcmp_8(mbase_start+cur_match, &scan_start) == 0)
                    break;
                GOTO_NEXT_CHAIN;
            }
        } else {
            for (;;) {
                if (zng_memcmp_4(mbase_end+cur_match, &scan_end) == 0 &&
                    zng_memcmp_4(mbase_start+cur_match, &scan_start) == 0)
                    break;
                GOTO_NEXT_CHAIN;
            }
        }
        uint32_t len = COMPARE256(scan+2, mbase_start+cur_match+2) + 2;
        Assert(scan+len <= window+(unsigned)(s->window_size-1), "wild scan");

        if (len > best_len) {
            uint32_t match_start = cur_match - match_offset;
            s->match_start = match_start;

            /* Do not look for better matches if the current match reaches
             * or exceeds the end of the input.
             */
            if (len >= lookahead)
                return lookahead;
            best_len = len;
            if (best_len >= nice_match)
                return best_len;

            offset = best_len-1;
            if (best_len >= sizeof(uint32_t)) {
                offset -= 2;
                if (best_len >= sizeof(uint64_t))
                    offset -= 4;
            }

            scan_end = zng_memread_8(scan+offset);

#ifdef LONGEST_MATCH_SLOW
            /* Look for a better string offset */
            if (UNLIKELY(len > STD_MIN_MATCH && match_start + len < strstart)) {
                Pos pos, next_pos;
                uint32_t i, hash;
                unsigned char *scan_endstr;

                /* Go back to offset 0 */
                cur_match -= match_offset;
                match_offset = 0;
                next_pos = cur_match;
                for (i = 0; i <= len - STD_MIN_MATCH; i++) {
                    pos = prev[(cur_match + i) & wmask];
                    if (pos < next_pos) {
                        /* Hash chain is more distant, use it */
                        if (pos <= limit_base + i)
                            goto break_matching;
                        next_pos = pos;
                        match_offset = (Pos)i;
                    }
                }
                /* Switch cur_match to next_pos chain */
                cur_match = next_pos;

                /* Try hash head at len-(STD_MIN_MATCH-1) position to see if we could get
                 * a better cur_match at the end of string. Using (STD_MIN_MATCH-1) lets
                 * us include one more byte into hash - the byte which will be checked
                 * in main loop now, and which allows to grow match by 1.
                 */
                scan_endstr = scan + len - (STD_MIN_MATCH+1);

                hash = s->update_hash(0, scan_endstr[0]);
                hash = s->update_hash(hash, scan_endstr[1]);
                hash = s->update_hash(hash, scan_endstr[2]);

                pos = s->head[hash];
                if (pos < cur_match) {
                    match_offset = (Pos)(len - (STD_MIN_MATCH+1));
                    if (pos <= limit_base + match_offset)
                        goto break_matching;
                    cur_match = pos;
                }

                /* Update offset-dependent variables */
                limit = limit_base+match_offset;
                mbase_start = window-match_offset;
                mbase_end = (mbase_start+offset);
                continue;
            }
#endif
            mbase_end = (mbase_start+offset);
        }
#ifndef LONGEST_MATCH_SLOW
        else if (UNLIKELY(early_exit)) {
            /* The probability of finding a match later if we here is pretty low, so for
             * performance it's best to outright stop here for the lower compression levels
             */
            break;
        }
#endif
        GOTO_NEXT_CHAIN;
    }
    return best_len;

#ifdef LONGEST_MATCH_SLOW
break_matching:

    if (best_len < s->lookahead)
        return best_len;

    return s->lookahead;
#endif
}

#undef LONGEST_MATCH_SLOW
#undef LONGEST_MATCH
