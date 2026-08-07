#if defined(X86_SSE2) && !defined(WITHOUT_CHORBA_SSE)

#include "zbuild.h"
#include "crc32_braid_p.h"
#include "crc32_braid_tbl.h"
#include "crc32.h"
#include <emmintrin.h>
#include "arch/x86/x86_intrins.h"
#include "arch_functions.h"

#define READ_NEXT(in, off, a, b) do { \
        a = _mm_load_si128((__m128i*)(in + off / sizeof(uint64_t))); \
        b = _mm_load_si128((__m128i*)(in + off / sizeof(uint64_t) + 2)); \
        } while (0);

#define NEXT_ROUND(invec, a, b, c, d) do { \
        a = _mm_xor_si128(_mm_slli_epi64(invec, 17), _mm_slli_epi64(invec, 55)); \
        b = _mm_xor_si128(_mm_xor_si128(_mm_srli_epi64(invec, 47), _mm_srli_epi64(invec, 9)), _mm_slli_epi64(invec, 19)); \
        c = _mm_xor_si128(_mm_srli_epi64(invec, 45), _mm_slli_epi64(invec, 44)); \
        d  = _mm_srli_epi64(invec, 20); \
        } while (0);

Z_INTERNAL uint32_t chorba_small_nondestructive_sse2(uint32_t crc, const uint64_t* buf, size_t len) {
    const uint64_t* input = buf;
    ALIGNED_(16) uint64_t final[9] = {0};
    uint64_t next1 = crc;
    crc = 0;
    uint64_t next2 = 0;
    uint64_t next3 = 0;
    uint64_t next4 = 0;
    uint64_t next5 = 0;

    __m128i next12 = _mm_cvtsi64_si128(next1);
    __m128i next34 = _mm_setzero_si128();
    __m128i next56 = _mm_setzero_si128();
    __m128i ab1, ab2, ab3, ab4, cd1, cd2, cd3, cd4;

    size_t i = 0;

    /* This is weird, doing for vs while drops 10% off the exec time */
    for(; (i + 256 + 40 + 32 + 32) < len; i += 32) {
        __m128i in1in2, in3in4;

        /*
        uint64_t chorba1 = input[i / sizeof(uint64_t)];
        uint64_t chorba2 = input[i / sizeof(uint64_t) + 1];
        uint64_t chorba3 = input[i / sizeof(uint64_t) + 2];
        uint64_t chorba4 = input[i / sizeof(uint64_t) + 3];
        uint64_t chorba5 = input[i / sizeof(uint64_t) + 4];
        uint64_t chorba6 = input[i / sizeof(uint64_t) + 5];
        uint64_t chorba7 = input[i / sizeof(uint64_t) + 6];
        uint64_t chorba8 = input[i / sizeof(uint64_t) + 7];
        */

        const uint64_t *inputPtr = input + (i / sizeof(uint64_t));
        const __m128i *inputPtr128 = (__m128i*)inputPtr;
        __m128i chorba12 = _mm_load_si128(inputPtr128++);
        __m128i chorba34 = _mm_load_si128(inputPtr128++);
        __m128i chorba56 = _mm_load_si128(inputPtr128++);
        __m128i chorba78 = _mm_load_si128(inputPtr128++);

        chorba12 = _mm_xor_si128(chorba12, next12);
        chorba34 = _mm_xor_si128(chorba34, next34);
        chorba56 = _mm_xor_si128(chorba56, next56);
        chorba78 = _mm_xor_si128(chorba78, chorba12);
        __m128i chorba45 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(chorba34), _mm_castsi128_pd(chorba56), 1));
        __m128i chorba23 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(chorba12),
                                                           _mm_castsi128_pd(chorba34), 1));
        /*
        chorba1 ^= next1;
        chorba2 ^= next2;
        chorba3 ^= next3;
        chorba4 ^= next4;
        chorba5 ^= next5;
        chorba7 ^= chorba1;
        chorba8 ^= chorba2;
        */
        i += 8 * 8;

        /* 0-3 */
        /*in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];*/
        READ_NEXT(input, i, in1in2, in3in4);
        __m128i chorba34xor = _mm_xor_si128(chorba34, _mm_unpacklo_epi64(_mm_setzero_si128(), chorba12));
        in1in2 = _mm_xor_si128(in1in2, chorba34xor);
        /*
        in1 ^= chorba3;
        in2 ^= chorba4 ^ chorba1;
        */

        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);
        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);

        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);

        */

        in3in4 = _mm_xor_si128(in3in4, ab1);
        /* _hopefully_ we don't get a huge domain switching penalty for this. This seems to be the best sequence */
        __m128i chorba56xor = _mm_xor_si128(chorba56, _mm_unpacklo_epi64(_mm_setzero_si128(), ab2));

        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba56xor, chorba23));
        in3in4 = _mm_xor_si128(in3in4, chorba12);

        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);
        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        in3 ^= a1 ^ chorba5 ^ chorba2 ^ chorba1;
        in4 ^= b1 ^a2 ^ chorba6 ^ chorba3 ^ chorba2;

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);

        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */

        __m128i b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1));
        __m128i a4_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab4);
        a4_ = _mm_xor_si128(b2c2, a4_);
        next12 = _mm_xor_si128(ab3, a4_);
        next12 = _mm_xor_si128(next12, cd1);

        __m128i d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        __m128i b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1));

        /*out1 = a3 ^ b2 ^ c1;
        out2 = b3 ^ c2 ^ d1 ^ a4;*/
        next34 = _mm_xor_si128(cd3, _mm_xor_si128(b4c4, d2_));
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());

        //out3 = b4 ^ c3 ^ d2;
        //out4 = c4 ^ d3;

        //out5 = d4;

        /*
        next1 = out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        i += 32;

        /* 4-7 */
        /*in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];*/
        READ_NEXT(input, i, in1in2, in3in4);

        in1in2 = _mm_xor_si128(in1in2, next12);
        in1in2 = _mm_xor_si128(in1in2, chorba78);
        in1in2 = _mm_xor_si128(in1in2, chorba45);
        in1in2 = _mm_xor_si128(in1in2, chorba34);

        /*
        in1 ^= next1 ^ chorba7 ^ chorba4 ^ chorba3;
        in2 ^= next2 ^ chorba8 ^ chorba5 ^ chorba4;
        */

        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);

        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];

        in3 ^= next3 ^ a1 ^ chorba6 ^ chorba5;
        in4 ^= next4 ^ b1 ^ a2  ^ chorba7 ^ chorba6;
        */
        in3in4 = _mm_xor_si128(in3in4, next34);
        in3in4 = _mm_xor_si128(in3in4, ab1);
        in3in4 = _mm_xor_si128(in3in4, chorba56);
        __m128i chorba67 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(chorba56), _mm_castsi128_pd(chorba78), 1));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba67, _mm_unpacklo_epi64(_mm_setzero_si128(), ab2)));

        /*
        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);

        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */

        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        ///*
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1));
        a4_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab4);
        a4_ = _mm_xor_si128(b2c2, a4_);
        next12 = _mm_xor_si128(ab3, cd1);

        next12 = _mm_xor_si128(next12, a4_);
        next12 = _mm_xor_si128(next12, next56);
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1));
        next34 = _mm_xor_si128(b4c4, cd3);
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());
        //*/

        /*
        out1 = a3 ^ b2 ^ c1;
        out2 = b3 ^ c2 ^ d1 ^ a4;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        i += 32;

        /* 8-11 */
        /*
        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        in1 ^= next1 ^ chorba8 ^ chorba7 ^ chorba1;
        in2 ^= next2 ^ chorba8 ^ chorba2;
        */

        READ_NEXT(input, i, in1in2, in3in4);

        __m128i chorba80 = _mm_unpackhi_epi64(chorba78, _mm_setzero_si128());
        __m128i next12_chorba12 = _mm_xor_si128(next12, chorba12);
        in1in2 = _mm_xor_si128(in1in2, chorba80);
        in1in2 = _mm_xor_si128(in1in2, chorba78);
        in1in2 = _mm_xor_si128(in1in2, next12_chorba12);

        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);

        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        /*in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];*/
        in3in4 = _mm_xor_si128(next34, in3in4);
        in3in4 = _mm_xor_si128(in3in4, ab1);
        __m128i a2_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab2);
        in3in4 = _mm_xor_si128(in3in4, chorba34);
        in3in4 = _mm_xor_si128(in3in4, a2_);

        /*
        in3 ^= next3 ^ a1 ^ chorba3;
        in4 ^= next4 ^ a2 ^ b1 ^ chorba4;

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);

        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */


        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        a4_ = _mm_unpacklo_epi64(next56, ab4);
        next12 = _mm_xor_si128(a4_, ab3);
        next12 = _mm_xor_si128(next12, cd1);
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1));
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1));
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next12 = _mm_xor_si128(next12, b2c2);
        next34 = _mm_xor_si128(b4c4, cd3);
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());

        /*
        out1 =      a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        i += 32;

        /* 12-15 */
        /*
        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        */
        READ_NEXT(input, i, in1in2, in3in4);
        in1in2 = _mm_xor_si128(in1in2, next12);
        __m128i chorb56xorchorb12 = _mm_xor_si128(chorba56, chorba12);
        in1in2 = _mm_xor_si128(in1in2, chorb56xorchorb12);
        __m128i chorb1_ = _mm_unpacklo_epi64(_mm_setzero_si128(), chorba12);
        in1in2 = _mm_xor_si128(in1in2, chorb1_);


        /*
        in1 ^= next1 ^ chorba5 ^ chorba1;
        in2 ^= next2 ^ chorba6 ^ chorba2 ^ chorba1;

        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);

        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        in3 ^= next3 ^ a1 ^ chorba7 ^ chorba3 ^ chorba2 ^ chorba1;
        in4 ^= next4 ^ a2 ^ b1 ^ chorba8 ^ chorba4 ^ chorba3 ^ chorba2;
        */

        in3in4 = _mm_xor_si128(next34, in3in4);
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(ab1, chorba78));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba34, chorba12));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba23, _mm_unpacklo_epi64(_mm_setzero_si128(), ab2)));
        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        /*

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);

        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */

        ///*
        a4_ = _mm_unpacklo_epi64(next56, ab4);
        next12 = _mm_xor_si128(_mm_xor_si128(a4_, ab3), cd1);
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1));
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1));
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next12 = _mm_xor_si128(next12, b2c2);
        next34 = _mm_xor_si128(b4c4, cd3);
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());
        //*/

        /*
        out1 =      a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        i += 32;

        /* 16-19 */
        /*
        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        in1 ^= next1 ^ chorba5 ^ chorba4 ^ chorba3 ^ chorba1;
        in2 ^= next2 ^ chorba6 ^ chorba5 ^ chorba4 ^ chorba1 ^ chorba2;
        */
        ///*
        READ_NEXT(input, i, in1in2, in3in4);
        __m128i chorba1_ = _mm_unpacklo_epi64(_mm_setzero_si128(), chorba12);
        in1in2 = _mm_xor_si128(_mm_xor_si128(next12, in1in2), _mm_xor_si128(chorba56, chorba45));
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(chorba12, chorba34));
        in1in2 = _mm_xor_si128(chorba1_, in1in2);

        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);
        //*/

        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);

        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        */
        ///*
        a2_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab2);
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(ab1, chorba78));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba56, chorba34));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba23, chorba67));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba1_, a2_));
        in3in4 = _mm_xor_si128(in3in4, next34);
        //*/
        /*
        in3 ^= next3 ^ a1 ^ chorba7 ^ chorba6 ^ chorba5 ^ chorba2 ^ chorba3;
        in4 ^= next4 ^ a2 ^ b1 ^ chorba8 ^ chorba7 ^ chorba6 ^ chorba3 ^ chorba4 ^ chorba1;
        */
        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        /*
        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);

        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */

        a4_ = _mm_unpacklo_epi64(next56, ab4);
        next12 = _mm_xor_si128(_mm_xor_si128(a4_, ab3), cd1);
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1));
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1));
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next12 = _mm_xor_si128(next12, b2c2);
        next34 = _mm_xor_si128(b4c4, cd3);
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());

        /*
        out1 =      a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        i += 32;

        /* 20-23 */
        /*
        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        in1 ^= next1 ^ chorba8 ^ chorba7 ^ chorba4 ^ chorba5 ^ chorba2 ^ chorba1;
        in2 ^= next2 ^ chorba8 ^ chorba5 ^ chorba6 ^ chorba3 ^ chorba2;
        */

        READ_NEXT(input, i, in1in2, in3in4);
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(next12, chorba78));
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(chorba45, chorba56));
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(chorba23, chorba12));
        in1in2 = _mm_xor_si128(in1in2, chorba80);
        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);

        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        in3 ^= next3 ^ a1 ^ chorba7 ^ chorba6 ^ chorba4 ^ chorba3 ^ chorba1;
        in4 ^= next4 ^ a2 ^ b1 ^ chorba8 ^ chorba7 ^ chorba5 ^ chorba4 ^ chorba2 ^ chorba1;
        */
        a2_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab2);
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(next34, ab1));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba78, chorba67));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba45, chorba34));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba1_, a2_));
        in3in4 = _mm_xor_si128(in3in4, chorba12);
        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        /*
        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);

        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */

        /*
        out1 =      a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        a4_ = _mm_unpacklo_epi64(next56, ab4);
        next12 = _mm_xor_si128(_mm_xor_si128(a4_, ab3), cd1);
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1));
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1));
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next12 = _mm_xor_si128(next12, b2c2);
        next34 = _mm_xor_si128(b4c4, cd3);
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());

        i += 32;

        /* 24-27 */
        /*
        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        in1 ^= next1 ^ chorba8 ^ chorba6 ^ chorba5 ^ chorba3 ^ chorba2 ^ chorba1;
        in2 ^= next2 ^ chorba7 ^ chorba6 ^ chorba4 ^ chorba3 ^ chorba2;
        */

        READ_NEXT(input, i, in1in2, in3in4);
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(next12, chorba67));
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(chorba56, chorba34));
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(chorba23, chorba12));
        in1in2 = _mm_xor_si128(in1in2, chorba80);
        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);

        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        /*in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        in3 ^= next3 ^ a1 ^ chorba8 ^ chorba7 ^ chorba5 ^ chorba4 ^ chorba3;
        in4 ^= next4 ^ a2 ^ b1 ^ chorba8 ^ chorba6 ^ chorba5 ^ chorba4;

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);

        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */
        a2_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab2);
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(next34, ab1));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba78, chorba56));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba45, chorba34));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba80, a2_));
        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        a4_ = _mm_unpacklo_epi64(next56, ab4);
        next12 = _mm_xor_si128(_mm_xor_si128(a4_, ab3), cd1);
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1));
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1));
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next12 = _mm_xor_si128(next12, b2c2);
        next34 = _mm_xor_si128(b4c4, cd3);
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());

        /*
        out1 =      a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */
        i += 32;

        /* 28-31 */
        /*
        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        in1 ^= next1 ^ chorba7 ^ chorba6 ^ chorba5;
        in2 ^= next2 ^ chorba8 ^ chorba7 ^ chorba6;
        */
        READ_NEXT(input, i, in1in2, in3in4);
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(next12, chorba78));
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(chorba67, chorba56));
        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);

        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        in3 ^= next3 ^ a1 ^ chorba8 ^ chorba7;
        in4 ^= next4 ^ a2 ^ b1 ^ chorba8;

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);

        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */
        a2_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab2);
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(next34, ab1));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba78, chorba80));
        in3in4 = _mm_xor_si128(a2_, in3in4);
        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        /*
        out1 =      a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;
        */

        /*
        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        a4_ = _mm_unpacklo_epi64(next56, ab4);
        next12 = _mm_xor_si128(_mm_xor_si128(a4_, ab3), cd1);
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1));
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1));
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next12 = _mm_xor_si128(next12, b2c2);
        next34 = _mm_xor_si128(b4c4, cd3);
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());
    }

    for(; (i + 40 + 32) < len; i += 32) {
        __m128i in1in2, in3in4;

        /*in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];*/
        //READ_NEXT_UNALIGNED(input, i, in1in2, in3in4);
        READ_NEXT(input, i, in1in2, in3in4);
        in1in2 = _mm_xor_si128(in1in2, next12);

        /*
        in1 ^=next1;
        in2 ^=next2;
        */

        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);
        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);

        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        in3 ^= next3 ^ a1;
        in4 ^= next4 ^ a2 ^ b1;

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);

        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */

        __m128i a2_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab2);
        __m128i ab1_next34 = _mm_xor_si128(next34, ab1);
        in3in4 = _mm_xor_si128(in3in4, ab1_next34);
        in3in4 = _mm_xor_si128(a2_, in3in4);
        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        /*

        out1 = a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        __m128i b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1));
        __m128i a4_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab4);
        a4_ = _mm_xor_si128(b2c2, a4_);
        next12 = _mm_xor_si128(ab3, a4_);
        next12 = _mm_xor_si128(next12, cd1);

        __m128i d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        __m128i b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1));
        next12 = _mm_xor_si128(next12, next56);
        next34 = _mm_xor_si128(cd3, _mm_xor_si128(b4c4, d2_));
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());
    }

    next1 = _mm_cvtsi128_si64(next12);
    next2 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(next12, next12));
    next3 = _mm_cvtsi128_si64(next34);
    next4 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(next34, next34));
    next5 = _mm_cvtsi128_si64(next56);

    /* Skip the call to memcpy */
    size_t copy_len = len - i;
    __m128i *final128 = (__m128i*)final;
    __m128i *input128 = (__m128i*)(input + i/ sizeof(uint64_t));
    while (copy_len >= 64) {
        _mm_store_si128(final128++, _mm_load_si128(input128++));
        _mm_store_si128(final128++, _mm_load_si128(input128++));
        _mm_store_si128(final128++, _mm_load_si128(input128++));
        _mm_store_si128(final128++, _mm_load_si128(input128++));
         copy_len -= 64;
    }

    while (copy_len >= 16) {
        _mm_store_si128(final128++, _mm_load_si128(input128++));
        copy_len -= 16;
    }

    uint8_t *src_bytes = (uint8_t*)input128;
    uint8_t *dst_bytes = (uint8_t*)final128;
    while (copy_len--) {
       *dst_bytes++ = *src_bytes++;
    }

    final[0] ^= next1;
    final[1] ^= next2;
    final[2] ^= next3;
    final[3] ^= next4;
    final[4] ^= next5;

    /* We perform the same loop that braid_internal is doing but we'll skip
     * the function call for this tiny tail */
    uint8_t *final_bytes = (uint8_t*)final;
    size_t rem = len - i;

    while (rem--) {
        crc = crc_table[(crc ^ *final_bytes++) & 0xff] ^ (crc >> 8);
    }

    return crc;
}

Z_INTERNAL uint32_t crc32_chorba_sse2(uint32_t crc, const uint8_t *buf, size_t len) {
    uint64_t* aligned_buf;
    uint32_t c = (~crc) & 0xffffffff;
    uintptr_t algn_diff = ((uintptr_t)16 - ((uintptr_t)buf & 15)) & 15;

    if (len > algn_diff + CHORBA_SMALL_THRESHOLD_64BIT) {
        if (algn_diff) {
            c = crc32_braid_internal(c, buf, algn_diff);
            len -= algn_diff;
        }
        aligned_buf = (uint64_t*) (buf + algn_diff);
#if !defined(WITHOUT_CHORBA)
        if(len > CHORBA_LARGE_THRESHOLD) {
            c = crc32_chorba_118960_nondestructive(c, (z_word_t*) aligned_buf, len);
        } else
#endif
        {
            c = chorba_small_nondestructive_sse2(c, aligned_buf, len);
        }
    } else {
        // Process too short lengths using crc32_braid
        c = crc32_braid_internal(c, buf, len);
    }

    /* Return the CRC, post-conditioned. */
    return c ^ 0xffffffff;
}
#endif
