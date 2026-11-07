/* slide_hash.c -- slide hash table C implementation
 *
 * Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "deflate.h"

/* ===========================================================================
 * Slide the hash table when sliding the window down (could be avoided with 32
 * bit values at the expense of memory usage). We slide even when level == 0 to
 * keep the hash table consistent if we switch back to level > 0 later.
 */
static inline void slide_hash_c_chain(Pos *table, uint32_t entries, uint16_t wsize) {
#ifdef NOT_TWEAK_COMPILER
    table += entries;
    do {
        unsigned m;
        m = *--table;
        *table = (Pos)(m >= wsize ? m-wsize : 0);
        /* If entries is not on any hash chain, prev[entries] is garbage but
         * its value will never be used.
         */
    } while (--entries);
#else
    {
    /* As of I make this change, gcc (4.8.*) isn't able to vectorize
     * this hot loop using saturated-subtraction on x86-64 architecture.
     * To avoid this defect, we can change the loop such that
     *    o. the pointer advance forward, and
     *    o. demote the variable 'm' to be local to the loop, and
     *       choose type "Pos" (instead of 'unsigned int') for the
     *       variable to avoid unnecessary zero-extension.
     */
        unsigned int i;
        Pos *q = table;
        for (i = 0; i < entries; i++) {
            Pos m = *q;
            Pos t = (Pos)wsize;
            *q++ = (Pos)(m >= t ? m-t: 0);
        }
    }
#endif /* NOT_TWEAK_COMPILER */
}

Z_INTERNAL void slide_hash_c(deflate_state *s) {
    uint16_t wsize = (uint16_t)s->w_size;

    slide_hash_c_chain(s->head, HASH_SIZE, wsize);
    slide_hash_c_chain(s->prev, wsize, wsize);
}
