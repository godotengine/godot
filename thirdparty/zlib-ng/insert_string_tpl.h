#ifndef INSERT_STRING_H_
#define INSERT_STRING_H_

/* insert_string_tpl.h -- Private insert_string functions shared with more than
 *                        one insert string implementation
 *
 * Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler
 *
 * Copyright (C) 2013 Intel Corporation. All rights reserved.
 * Authors:
 *  Wajdi Feghali   <wajdi.k.feghali@intel.com>
 *  Jim Guilford    <james.guilford@intel.com>
 *  Vinodh Gopal    <vinodh.gopal@intel.com>
 *  Erdinc Ozturk   <erdinc.ozturk@intel.com>
 *  Jim Kukunas     <james.t.kukunas@linux.intel.com>
 *
 * Portions are Copyright (C) 2016 12Sided Technology, LLC.
 * Author:
 *  Phil Vachon     <pvachon@12sidedtech.com>
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 */

#include "zmemory.h"

#ifndef HASH_CALC_OFFSET
#  define HASH_CALC_OFFSET 0
#endif
#ifndef HASH_CALC_MASK
#  define HASH_CALC_MASK HASH_MASK
#endif
#ifndef HASH_CALC_READ
#  if BYTE_ORDER == LITTLE_ENDIAN
#    define HASH_CALC_READ \
        val = zng_memread_4(strstart);
#  else
#    define HASH_CALC_READ \
        val = ZSWAP32(zng_memread_4(strstart));
#  endif
#endif

/* ===========================================================================
 * Update a hash value with the given input byte
 * IN  assertion: all calls to UPDATE_HASH are made with consecutive
 *    input characters, so that a running hash key can be computed from the
 *    previous key instead of complete recalculation each time.
 */
Z_INTERNAL uint32_t UPDATE_HASH(uint32_t h, uint32_t val) {
    HASH_CALC(h, val);
    return h & HASH_CALC_MASK;
}

/* ===========================================================================
 * Quick insert string str in the dictionary and set match_head to the previous head
 * of the hash chain (the most recent string with same hash key). Return
 * the previous length of the hash chain.
 */
Z_INTERNAL Pos QUICK_INSERT_STRING(deflate_state *const s, uint32_t str) {
    uint8_t *strstart = s->window + str + HASH_CALC_OFFSET;
    uint32_t val, hm;
    Pos head;

    HASH_CALC_VAR_INIT;
    HASH_CALC_READ;
    HASH_CALC(HASH_CALC_VAR, val);
    HASH_CALC_VAR &= HASH_CALC_MASK;
    hm = HASH_CALC_VAR;

    head = s->head[hm];
    if (LIKELY(head != str)) {
        s->prev[str & s->w_mask] = head;
        s->head[hm] = (Pos)str;
    }
    return head;
}

/* ===========================================================================
 * Insert string str in the dictionary and set match_head to the previous head
 * of the hash chain (the most recent string with same hash key). Return
 * the previous length of the hash chain.
 * IN  assertion: all calls to INSERT_STRING are made with consecutive
 *    input characters and the first STD_MIN_MATCH bytes of str are valid
 *    (except for the last STD_MIN_MATCH-1 bytes of the input file).
 */
Z_INTERNAL void INSERT_STRING(deflate_state *const s, uint32_t str, uint32_t count) {
    uint8_t *strstart = s->window + str + HASH_CALC_OFFSET;
    uint8_t *strend = strstart + count;

    /* Local pointers to avoid indirection */
    Pos *headp = s->head;
    Pos *prevp = s->prev;
    const unsigned int w_mask = s->w_mask;

    for (Pos idx = (Pos)str; strstart < strend; idx++, strstart++) {
        uint32_t val, hm;

        HASH_CALC_VAR_INIT;
        HASH_CALC_READ;
        HASH_CALC(HASH_CALC_VAR, val);
        HASH_CALC_VAR &= HASH_CALC_MASK;
        hm = HASH_CALC_VAR;

        Pos head = headp[hm];
        if (LIKELY(head != idx)) {
            prevp[idx & w_mask] = head;
            headp[hm] = idx;
        }
    }
}
#endif
