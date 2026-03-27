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

#if defined(OPUS_X86_MAY_HAVE_SSE2) && defined(FIXED_POINT)
opus_val32 celt_inner_prod_sse2(const opus_val16 *x, const opus_val16 *y,
      int N)
{
    opus_int  i, dataSize16;
    opus_int32 sum;

    __m128i inVec1_76543210, inVec1_FEDCBA98, acc1;
    __m128i inVec2_76543210, inVec2_FEDCBA98, acc2;

    sum = 0;
    dataSize16 = N & ~15;

    acc1 = _mm_setzero_si128();
    acc2 = _mm_setzero_si128();

    for (i=0;i<dataSize16;i+=16)
    {
        inVec1_76543210 = _mm_loadu_si128((__m128i *)(void*)(&x[i + 0]));
        inVec2_76543210 = _mm_loadu_si128((__m128i *)(void*)(&y[i + 0]));

        inVec1_FEDCBA98 = _mm_loadu_si128((__m128i *)(void*)(&x[i + 8]));
        inVec2_FEDCBA98 = _mm_loadu_si128((__m128i *)(void*)(&y[i + 8]));

        inVec1_76543210 = _mm_madd_epi16(inVec1_76543210, inVec2_76543210);
        inVec1_FEDCBA98 = _mm_madd_epi16(inVec1_FEDCBA98, inVec2_FEDCBA98);

        acc1 = _mm_add_epi32(acc1, inVec1_76543210);
        acc2 = _mm_add_epi32(acc2, inVec1_FEDCBA98);
    }

    acc1 = _mm_add_epi32( acc1, acc2 );

    if (N - i >= 8)
    {
        inVec1_76543210 = _mm_loadu_si128((__m128i *)(void*)(&x[i + 0]));
        inVec2_76543210 = _mm_loadu_si128((__m128i *)(void*)(&y[i + 0]));

        inVec1_76543210 = _mm_madd_epi16(inVec1_76543210, inVec2_76543210);

        acc1 = _mm_add_epi32(acc1, inVec1_76543210);
        i += 8;
    }

    acc1 = _mm_add_epi32(acc1, _mm_unpackhi_epi64( acc1, acc1));
    acc1 = _mm_add_epi32(acc1, _mm_shufflelo_epi16( acc1, 0x0E));
    sum += _mm_cvtsi128_si32(acc1);

    for (;i<N;i++) {
        sum = silk_SMLABB(sum, x[i], y[i]);
    }

    return sum;
}
#endif
