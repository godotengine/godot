/* compare256.c -- 256 byte memory comparison with match length return
 * Copyright (C) 2020 Nathan Moinvaziri
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "compare256_p.h"

// Set optimal COMPARE256 function variant
#if OPTIMAL_CMP == 8
#  define COMPARE256          compare256_8
#elif defined(HAVE_BUILTIN_CTZLL)
#  define COMPARE256          compare256_64
#elif defined(HAVE_BUILTIN_CTZ)
#  define COMPARE256          compare256_32
#else
#  define COMPARE256          compare256_16
#endif

Z_INTERNAL uint32_t compare256_c(const uint8_t *src0, const uint8_t *src1) {
    return COMPARE256(src0, src1);
}

// Generate longest_match_c
#define LONGEST_MATCH       longest_match_c
#include "match_tpl.h"

// Generate longest_match_slow_c
#define LONGEST_MATCH_SLOW
#define LONGEST_MATCH       longest_match_slow_c
#include "match_tpl.h"
