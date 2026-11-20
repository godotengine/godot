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
#include <smmintrin.h>
#include "main.h"

#include "SigProc_FIX.h"
#include "pitch.h"
#include "celt/x86/x86cpu.h"

opus_int64 silk_inner_prod16_sse4_1(
    const opus_int16            *inVec1,            /*    I input vector 1                                              */
    const opus_int16            *inVec2,            /*    I input vector 2                                              */
    const opus_int              len                 /*    I vector lengths                                              */
)
{
    opus_int  i, dataSize4;
    opus_int64 sum;

    __m128i xmm_prod_20, xmm_prod_31;
    __m128i inVec1_3210, acc1;
    __m128i inVec2_3210, acc2;

    sum = 0;
    dataSize4 = len & ~3;

    acc1 = _mm_setzero_si128();
    acc2 = _mm_setzero_si128();

    for( i = 0; i < dataSize4; i += 4 ) {
        inVec1_3210 = OP_CVTEPI16_EPI32_M64( &inVec1[i + 0] );
        inVec2_3210 = OP_CVTEPI16_EPI32_M64( &inVec2[i + 0] );
        xmm_prod_20 = _mm_mul_epi32( inVec1_3210, inVec2_3210 );

        inVec1_3210 = _mm_shuffle_epi32( inVec1_3210, _MM_SHUFFLE( 0, 3, 2, 1 ) );
        inVec2_3210 = _mm_shuffle_epi32( inVec2_3210, _MM_SHUFFLE( 0, 3, 2, 1 ) );
        xmm_prod_31 = _mm_mul_epi32( inVec1_3210, inVec2_3210 );

        acc1 = _mm_add_epi64( acc1, xmm_prod_20 );
        acc2 = _mm_add_epi64( acc2, xmm_prod_31 );
    }

    acc1 = _mm_add_epi64( acc1, acc2 );

    /* equal shift right 8 bytes */
    acc2 = _mm_shuffle_epi32( acc1, _MM_SHUFFLE( 0, 0, 3, 2 ) );
    acc1 = _mm_add_epi64( acc1, acc2 );

    _mm_storel_epi64( (__m128i *)(void *)&sum, acc1 );

    for( ; i < len; i++ ) {
        sum = silk_SMLALBB( sum, inVec1[ i ], inVec2[ i ] );
    }

#ifdef OPUS_CHECK_ASM
    {
        opus_int64 sum_c = silk_inner_prod16_c( inVec1, inVec2, len );
        silk_assert( sum == sum_c );
    }
#endif

    return sum;
}
