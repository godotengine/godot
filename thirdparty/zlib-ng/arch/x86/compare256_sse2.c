/* compare256_sse2.c -- SSE2 version of compare256
 * Copyright Adam Stylinski <kungfujesus06@gmail.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zmemory.h"
#include "deflate.h"
#include "fallback_builtins.h"

#if defined(X86_SSE2) && defined(HAVE_BUILTIN_CTZ)

#include <emmintrin.h>

static inline uint32_t compare256_sse2_static(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;
    int align_offset = ((uintptr_t)src0) & 15;
    const uint8_t *end0 = src0 + 256;
    const uint8_t *end1 = src1 + 256;
    __m128i xmm_src0, xmm_src1, xmm_cmp;

    /* Do the first load unaligned, than all subsequent ones we have at least
     * one aligned load. Sadly aligning both loads is probably unrealistic */
    xmm_src0 = _mm_loadu_si128((__m128i*)src0);
    xmm_src1 = _mm_loadu_si128((__m128i*)src1);
    xmm_cmp = _mm_cmpeq_epi8(xmm_src0, xmm_src1);

    unsigned mask = (unsigned)_mm_movemask_epi8(xmm_cmp);

    /* Compiler _may_ turn this branch into a ptest + movemask,
     * since a lot of those uops are shared and fused */
    if (mask != 0xFFFF) {
        uint32_t match_byte = (uint32_t)__builtin_ctz(~mask);
        return len + match_byte;
    }

    int align_adv = 16 - align_offset;
    len += align_adv;
    src0 += align_adv;
    src1 += align_adv;

    /* Do a flooring division (should just be a shift right) */
    int num_iter = (256 - len) / 16;

    for (int i = 0; i < num_iter; ++i) {
        xmm_src0 = _mm_load_si128((__m128i*)src0);
        xmm_src1 = _mm_loadu_si128((__m128i*)src1);
        xmm_cmp = _mm_cmpeq_epi8(xmm_src0, xmm_src1);

        mask = (unsigned)_mm_movemask_epi8(xmm_cmp);

        /* Compiler _may_ turn this branch into a ptest + movemask,
         * since a lot of those uops are shared and fused */
        if (mask != 0xFFFF) {
            uint32_t match_byte = (uint32_t)__builtin_ctz(~mask);
            return len + match_byte;
        }

        len += 16, src0 += 16, src1 += 16;
    }

    if (align_offset) {
        src0 = end0 - 16;
        src1 = end1 - 16;
        len = 256 - 16;

        xmm_src0 = _mm_loadu_si128((__m128i*)src0);
        xmm_src1 = _mm_loadu_si128((__m128i*)src1);
        xmm_cmp = _mm_cmpeq_epi8(xmm_src0, xmm_src1);

        mask = (unsigned)_mm_movemask_epi8(xmm_cmp);

        if (mask != 0xFFFF) {
            uint32_t match_byte = (uint32_t)__builtin_ctz(~mask);
            return len + match_byte;
        }
    }

    return 256;
}

Z_INTERNAL uint32_t compare256_sse2(const uint8_t *src0, const uint8_t *src1) {
    return compare256_sse2_static(src0, src1);
}

#define LONGEST_MATCH       longest_match_sse2
#define COMPARE256          compare256_sse2_static

#include "match_tpl.h"

#define LONGEST_MATCH_SLOW
#define LONGEST_MATCH       longest_match_slow_sse2
#define COMPARE256          compare256_sse2_static

#include "match_tpl.h"

#endif
