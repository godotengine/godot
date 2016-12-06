/* Copyright (c) 2014, Cisco Systems, INC
   Written by XiangMingZhu WeiZhou MinPeng YanWang

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <xmmintrin.h>
#include <emmintrin.h>

#include "macros.h"
#include "celt_lpc.h"
#include "stack_alloc.h"
#include "mathops.h"
#include "pitch.h"

#if defined(OPUS_X86_MAY_HAVE_SSE4_1) && defined(FIXED_POINT)
#include <smmintrin.h>
#include "x86cpu.h"

opus_val32 celt_inner_prod_sse4_1(const opus_val16 *x, const opus_val16 *y,
      int N)
{
    opus_int  i, dataSize16;
    opus_int32 sum;
    __m128i inVec1_76543210, inVec1_FEDCBA98, acc1;
    __m128i inVec2_76543210, inVec2_FEDCBA98, acc2;
    __m128i inVec1_3210, inVec2_3210;

    sum = 0;
    dataSize16 = N & ~15;

    acc1 = _mm_setzero_si128();
    acc2 = _mm_setzero_si128();

    for (i=0;i<dataSize16;i+=16) {
        inVec1_76543210 = _mm_loadu_si128((__m128i *)(&x[i + 0]));
        inVec2_76543210 = _mm_loadu_si128((__m128i *)(&y[i + 0]));

        inVec1_FEDCBA98 = _mm_loadu_si128((__m128i *)(&x[i + 8]));
        inVec2_FEDCBA98 = _mm_loadu_si128((__m128i *)(&y[i + 8]));

        inVec1_76543210 = _mm_madd_epi16(inVec1_76543210, inVec2_76543210);
        inVec1_FEDCBA98 = _mm_madd_epi16(inVec1_FEDCBA98, inVec2_FEDCBA98);

        acc1 = _mm_add_epi32(acc1, inVec1_76543210);
        acc2 = _mm_add_epi32(acc2, inVec1_FEDCBA98);
    }

    acc1 = _mm_add_epi32(acc1, acc2);

    if (N - i >= 8)
    {
        inVec1_76543210 = _mm_loadu_si128((__m128i *)(&x[i + 0]));
        inVec2_76543210 = _mm_loadu_si128((__m128i *)(&y[i + 0]));

        inVec1_76543210 = _mm_madd_epi16(inVec1_76543210, inVec2_76543210);

        acc1 = _mm_add_epi32(acc1, inVec1_76543210);
        i += 8;
    }

    if (N - i >= 4)
    {
        inVec1_3210 = OP_CVTEPI16_EPI32_M64(&x[i + 0]);
        inVec2_3210 = OP_CVTEPI16_EPI32_M64(&y[i + 0]);

        inVec1_3210 = _mm_mullo_epi32(inVec1_3210, inVec2_3210);

        acc1 = _mm_add_epi32(acc1, inVec1_3210);
        i += 4;
    }

    acc1 = _mm_add_epi32(acc1, _mm_unpackhi_epi64(acc1, acc1));
    acc1 = _mm_add_epi32(acc1, _mm_shufflelo_epi16(acc1, 0x0E));

    sum += _mm_cvtsi128_si32(acc1);

    for (;i<N;i++)
    {
        sum = silk_SMLABB(sum, x[i], y[i]);
    }

    return sum;
}

void xcorr_kernel_sse4_1(const opus_val16 * x, const opus_val16 * y, opus_val32 sum[ 4 ], int len)
{
    int j;

    __m128i vecX, vecX0, vecX1, vecX2, vecX3;
    __m128i vecY0, vecY1, vecY2, vecY3;
    __m128i sum0, sum1, sum2, sum3, vecSum;
    __m128i initSum;

    celt_assert(len >= 3);

    sum0 = _mm_setzero_si128();
    sum1 = _mm_setzero_si128();
    sum2 = _mm_setzero_si128();
    sum3 = _mm_setzero_si128();

    for (j=0;j<(len-7);j+=8)
    {
        vecX = _mm_loadu_si128((__m128i *)(&x[j + 0]));
        vecY0 = _mm_loadu_si128((__m128i *)(&y[j + 0]));
        vecY1 = _mm_loadu_si128((__m128i *)(&y[j + 1]));
        vecY2 = _mm_loadu_si128((__m128i *)(&y[j + 2]));
        vecY3 = _mm_loadu_si128((__m128i *)(&y[j + 3]));

        sum0 = _mm_add_epi32(sum0, _mm_madd_epi16(vecX, vecY0));
        sum1 = _mm_add_epi32(sum1, _mm_madd_epi16(vecX, vecY1));
        sum2 = _mm_add_epi32(sum2, _mm_madd_epi16(vecX, vecY2));
        sum3 = _mm_add_epi32(sum3, _mm_madd_epi16(vecX, vecY3));
    }

    sum0 = _mm_add_epi32(sum0, _mm_unpackhi_epi64( sum0, sum0));
    sum0 = _mm_add_epi32(sum0, _mm_shufflelo_epi16( sum0, 0x0E));

    sum1 = _mm_add_epi32(sum1, _mm_unpackhi_epi64( sum1, sum1));
    sum1 = _mm_add_epi32(sum1, _mm_shufflelo_epi16( sum1, 0x0E));

    sum2 = _mm_add_epi32(sum2, _mm_unpackhi_epi64( sum2, sum2));
    sum2 = _mm_add_epi32(sum2, _mm_shufflelo_epi16( sum2, 0x0E));

    sum3 = _mm_add_epi32(sum3, _mm_unpackhi_epi64( sum3, sum3));
    sum3 = _mm_add_epi32(sum3, _mm_shufflelo_epi16( sum3, 0x0E));

    vecSum = _mm_unpacklo_epi64(_mm_unpacklo_epi32(sum0, sum1),
          _mm_unpacklo_epi32(sum2, sum3));

    for (;j<(len-3);j+=4)
    {
        vecX = OP_CVTEPI16_EPI32_M64(&x[j + 0]);
        vecX0 = _mm_shuffle_epi32(vecX, 0x00);
        vecX1 = _mm_shuffle_epi32(vecX, 0x55);
        vecX2 = _mm_shuffle_epi32(vecX, 0xaa);
        vecX3 = _mm_shuffle_epi32(vecX, 0xff);

        vecY0 = OP_CVTEPI16_EPI32_M64(&y[j + 0]);
        vecY1 = OP_CVTEPI16_EPI32_M64(&y[j + 1]);
        vecY2 = OP_CVTEPI16_EPI32_M64(&y[j + 2]);
        vecY3 = OP_CVTEPI16_EPI32_M64(&y[j + 3]);

        sum0 = _mm_mullo_epi32(vecX0, vecY0);
        sum1 = _mm_mullo_epi32(vecX1, vecY1);
        sum2 = _mm_mullo_epi32(vecX2, vecY2);
        sum3 = _mm_mullo_epi32(vecX3, vecY3);

        sum0 = _mm_add_epi32(sum0, sum1);
        sum2 = _mm_add_epi32(sum2, sum3);
        vecSum = _mm_add_epi32(vecSum, sum0);
        vecSum = _mm_add_epi32(vecSum, sum2);
    }

    for (;j<len;j++)
    {
        vecX = OP_CVTEPI16_EPI32_M64(&x[j + 0]);
        vecX0 = _mm_shuffle_epi32(vecX, 0x00);

        vecY0 = OP_CVTEPI16_EPI32_M64(&y[j + 0]);

        sum0 = _mm_mullo_epi32(vecX0, vecY0);
        vecSum = _mm_add_epi32(vecSum, sum0);
    }

    initSum = _mm_loadu_si128((__m128i *)(&sum[0]));
    initSum = _mm_add_epi32(initSum, vecSum);
    _mm_storeu_si128((__m128i *)sum, initSum);
}
#endif
