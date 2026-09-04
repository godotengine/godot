/* adler32_ssse3.c -- compute the Adler-32 checksum of a data stream
 * Copyright (C) 1995-2011 Mark Adler
 * Authors:
 *   Adam Stylinski <kungfujesus06@gmail.com>
 *   Brian Bockelman <bockelman@gmail.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "adler32_p.h"
#include "adler32_ssse3_p.h"

#ifdef X86_SSSE3

#include <immintrin.h>

Z_INTERNAL uint32_t adler32_ssse3(uint32_t adler, const uint8_t *buf, size_t len) {
    uint32_t sum2;

     /* split Adler-32 into component sums */
    sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* in case user likes doing a byte at a time, keep it fast */
    if (UNLIKELY(len == 1))
        return adler32_len_1(adler, buf, sum2);

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (UNLIKELY(buf == NULL))
        return 1L;

    /* in case short lengths are provided, keep it somewhat fast */
    if (UNLIKELY(len < 16))
        return adler32_len_16(adler, buf, len, sum2);

    const __m128i dot2v = _mm_setr_epi8(32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17);
    const __m128i dot2v_0 = _mm_setr_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    const __m128i dot3v = _mm_set1_epi16(1);
    const __m128i zero = _mm_setzero_si128();

    __m128i vbuf, vs1_0, vs3, vs1, vs2, vs2_0, v_sad_sum1, v_short_sum2, v_short_sum2_0,
            vbuf_0, v_sad_sum2, vsum2, vsum2_0;

    /* If our buffer is unaligned (likely), make the determination whether
     * or not there's enough of a buffer to consume to make the scalar, aligning
     * additions worthwhile or if it's worth it to just eat the cost of an unaligned
     * load. This is a pretty simple test, just test if 16 - the remainder + len is
     * < 16 */
    size_t max_iters = NMAX;
    size_t rem = (uintptr_t)buf & 15;
    size_t align_offset = 16 - rem;
    size_t k = 0;
    if (rem) {
        if (len < 16 + align_offset) {
            /* Let's eat the cost of this one unaligned load so that
             * we don't completely skip over the vectorization. Doing
             * 16 bytes at a time unaligned is better than 16 + <= 15
             * sums */
            vbuf = _mm_loadu_si128((__m128i*)buf);
            len -= 16;
            buf += 16;
            vs1 = _mm_cvtsi32_si128(adler);
            vs2 = _mm_cvtsi32_si128(sum2);
            vs3 = _mm_setzero_si128();
            vs1_0 = vs1;
            goto unaligned_jmp;
        }

        for (size_t i = 0; i < align_offset; ++i) {
            adler += *(buf++);
            sum2 += adler;
        }

        /* lop off the max number of sums based on the scalar sums done
         * above */
        len -= align_offset;
        max_iters -= align_offset;
    }


    while (len >= 16) {
        vs1 = _mm_cvtsi32_si128(adler);
        vs2 = _mm_cvtsi32_si128(sum2);
        vs3 = _mm_setzero_si128();
        vs2_0 = _mm_setzero_si128();
        vs1_0 = vs1;

        k = (len < max_iters ? len : max_iters);
        k -= k % 16;
        len -= k;

        while (k >= 32) {
            /*
               vs1 = adler + sum(c[i])
               vs2 = sum2 + 16 vs1 + sum( (16-i+1) c[i] )
            */
            vbuf = _mm_load_si128((__m128i*)buf);
            vbuf_0 = _mm_load_si128((__m128i*)(buf + 16));
            buf += 32;
            k -= 32;

            v_sad_sum1 = _mm_sad_epu8(vbuf, zero);
            v_sad_sum2 = _mm_sad_epu8(vbuf_0, zero);
            vs1 = _mm_add_epi32(v_sad_sum1, vs1);
            vs3 = _mm_add_epi32(vs1_0, vs3);

            vs1 = _mm_add_epi32(v_sad_sum2, vs1);
            v_short_sum2 = _mm_maddubs_epi16(vbuf, dot2v);
            vsum2 = _mm_madd_epi16(v_short_sum2, dot3v);
            v_short_sum2_0 = _mm_maddubs_epi16(vbuf_0, dot2v_0);
            vs2 = _mm_add_epi32(vsum2, vs2);
            vsum2_0 = _mm_madd_epi16(v_short_sum2_0, dot3v);
            vs2_0 = _mm_add_epi32(vsum2_0, vs2_0);
            vs1_0 = vs1;
        }

        vs2 = _mm_add_epi32(vs2_0, vs2);
        vs3 = _mm_slli_epi32(vs3, 5);
        vs2 = _mm_add_epi32(vs3, vs2);
        vs3 = _mm_setzero_si128();

        while (k >= 16) {
            /*
               vs1 = adler + sum(c[i])
               vs2 = sum2 + 16 vs1 + sum( (16-i+1) c[i] )
            */
            vbuf = _mm_load_si128((__m128i*)buf);
            buf += 16;
            k -= 16;

unaligned_jmp:
            v_sad_sum1 = _mm_sad_epu8(vbuf, zero);
            vs1 = _mm_add_epi32(v_sad_sum1, vs1);
            vs3 = _mm_add_epi32(vs1_0, vs3);
            v_short_sum2 = _mm_maddubs_epi16(vbuf, dot2v_0);
            vsum2 = _mm_madd_epi16(v_short_sum2, dot3v);
            vs2 = _mm_add_epi32(vsum2, vs2);
            vs1_0 = vs1;
        }

        vs3 = _mm_slli_epi32(vs3, 4);
        vs2 = _mm_add_epi32(vs2, vs3);

        /* We don't actually need to do a full horizontal sum, since psadbw is actually doing
         * a partial reduction sum implicitly and only summing to integers in vector positions
         * 0 and 2. This saves us some contention on the shuffle port(s) */
        adler = partial_hsum(vs1) % BASE;
        sum2 = hsum(vs2) % BASE;
        max_iters = NMAX;
    }

    /* Process tail (len < 16).  */
    return adler32_len_16(adler, buf, len, sum2);
}

#endif
