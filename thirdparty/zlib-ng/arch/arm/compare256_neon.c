/* compare256_neon.c - NEON version of compare256
 * Copyright (C) 2022 Nathan Moinvaziri
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zmemory.h"
#include "deflate.h"
#include "fallback_builtins.h"

#if defined(ARM_NEON) && defined(HAVE_BUILTIN_CTZLL)
#include "neon_intrins.h"

static inline uint32_t compare256_neon_static(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;

    do {
        uint8x16_t a, b, cmp;
        uint64_t lane;

        a = vld1q_u8(src0);
        b = vld1q_u8(src1);

        cmp = veorq_u8(a, b);

        lane = vgetq_lane_u64(vreinterpretq_u64_u8(cmp), 0);
        if (lane) {
            uint32_t match_byte = (uint32_t)__builtin_ctzll(lane) / 8;
            return len + match_byte;
        }
        len += 8;
        lane = vgetq_lane_u64(vreinterpretq_u64_u8(cmp), 1);
        if (lane) {
            uint32_t match_byte = (uint32_t)__builtin_ctzll(lane) / 8;
            return len + match_byte;
        }
        len += 8;

        src0 += 16, src1 += 16;
    } while (len < 256);

    return 256;
}

Z_INTERNAL uint32_t compare256_neon(const uint8_t *src0, const uint8_t *src1) {
    return compare256_neon_static(src0, src1);
}

#define LONGEST_MATCH       longest_match_neon
#define COMPARE256          compare256_neon_static

#include "match_tpl.h"

#define LONGEST_MATCH_SLOW
#define LONGEST_MATCH       longest_match_slow_neon
#define COMPARE256          compare256_neon_static

#include "match_tpl.h"

#endif
