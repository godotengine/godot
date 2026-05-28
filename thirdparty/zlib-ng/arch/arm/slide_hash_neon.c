/* slide_hash_neon.c -- Optimized hash table shifting for ARM with support for NEON instructions
 * Copyright (C) 2017-2020 Mika T. Lindqvist
 *
 * Authors:
 * Mika T. Lindqvist <postmaster@raasu.org>
 * Jun He <jun.he@arm.com>
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef ARM_NEON
#include "neon_intrins.h"
#include "zbuild.h"
#include "deflate.h"

/* SIMD version of hash_chain rebase */
static inline void slide_hash_chain(Pos *table, uint32_t entries, uint16_t wsize) {
    Z_REGISTER uint16x8_t v;
    uint16x8x4_t p0, p1;
    Z_REGISTER size_t n;

    size_t size = entries*sizeof(table[0]);
    Assert((size % sizeof(uint16x8_t) * 8 == 0), "hash table size err");

    Assert(sizeof(Pos) == 2, "Wrong Pos size");
    v = vdupq_n_u16(wsize);

    n = size / (sizeof(uint16x8_t) * 8);
    do {
        p0 = vld1q_u16_x4(table);
        p1 = vld1q_u16_x4(table+32);
        vqsubq_u16_x4_x1(p0, p0, v);
        vqsubq_u16_x4_x1(p1, p1, v);
        vst1q_u16_x4(table, p0);
        vst1q_u16_x4(table+32, p1);
        table += 64;
    } while (--n);
}

Z_INTERNAL void slide_hash_neon(deflate_state *s) {
    Assert(s->w_size <= UINT16_MAX, "w_size should fit in uint16_t");
    uint16_t wsize = (uint16_t)s->w_size;

    slide_hash_chain(s->head, HASH_SIZE, wsize);
    slide_hash_chain(s->prev, wsize, wsize);
}
#endif
