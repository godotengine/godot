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
#include "celt/x86/x86cpu.h"

void silk_warped_LPC_analysis_filter_FIX_sse4_1(
    opus_int32                  state[],                    /* I/O  State [order + 1]                   */
    opus_int32                  res_Q2[],                   /* O    Residual signal [length]            */
    const opus_int16            coef_Q13[],                 /* I    Coefficients [order]                */
    const opus_int16            input[],                    /* I    Input signal [length]               */
    const opus_int16            lambda_Q16,                 /* I    Warping factor                      */
    const opus_int              length,                     /* I    Length of input signal              */
    const opus_int              order                       /* I    Filter order (even)                 */
)
{
    opus_int     n, i;
    opus_int32   acc_Q11, tmp1, tmp2;

    /* Order must be even */
    silk_assert( ( order & 1 ) == 0 );

    if (order == 10)
    {
        if (0 == lambda_Q16)
        {
            __m128i coef_Q13_3210, coef_Q13_7654;
            __m128i coef_Q13_0123, coef_Q13_4567;
            __m128i state_0123, state_4567;
            __m128i xmm_product1, xmm_product2;
            __m128i xmm_tempa, xmm_tempb;

            register opus_int32 sum;
            register opus_int32 state_8, state_9, state_a;
            register opus_int64 coef_Q13_8, coef_Q13_9;

            silk_assert( length > 0 );

            coef_Q13_3210 = OP_CVTEPI16_EPI32_M64( &coef_Q13[ 0 ] );
            coef_Q13_7654 = OP_CVTEPI16_EPI32_M64( &coef_Q13[ 4 ] );

            coef_Q13_0123 = _mm_shuffle_epi32( coef_Q13_3210, _MM_SHUFFLE( 0, 1, 2, 3 ) );
            coef_Q13_4567 = _mm_shuffle_epi32( coef_Q13_7654, _MM_SHUFFLE( 0, 1, 2, 3 ) );

            coef_Q13_8 = (opus_int64) coef_Q13[ 8 ];
            coef_Q13_9 = (opus_int64) coef_Q13[ 9 ];

            state_0123 = _mm_loadu_si128( (__m128i *)(&state[ 0 ] ) );
            state_4567 = _mm_loadu_si128( (__m128i *)(&state[ 4 ] ) );

            state_0123 = _mm_shuffle_epi32( state_0123, _MM_SHUFFLE( 0, 1, 2, 3 ) );
            state_4567 = _mm_shuffle_epi32( state_4567, _MM_SHUFFLE( 0, 1, 2, 3 ) );

            state_8 = state[ 8 ];
            state_9 = state[ 9 ];
            state_a = 0;

            for( n = 0; n < length; n++ )
            {
                xmm_product1 = _mm_mul_epi32( coef_Q13_0123, state_0123 ); /* 64-bit multiply, only 2 pairs */
                xmm_product2 = _mm_mul_epi32( coef_Q13_4567, state_4567 );

                xmm_tempa = _mm_shuffle_epi32( state_0123, _MM_SHUFFLE( 0, 1, 2, 3 ) );
                xmm_tempb = _mm_shuffle_epi32( state_4567, _MM_SHUFFLE( 0, 1, 2, 3 ) );

                xmm_product1 = _mm_srli_epi64( xmm_product1, 16 ); /* >> 16, zero extending works */
                xmm_product2 = _mm_srli_epi64( xmm_product2, 16 );

                xmm_tempa = _mm_mul_epi32( coef_Q13_3210, xmm_tempa );
                xmm_tempb = _mm_mul_epi32( coef_Q13_7654, xmm_tempb );

                xmm_tempa = _mm_srli_epi64( xmm_tempa, 16 );
                xmm_tempb = _mm_srli_epi64( xmm_tempb, 16 );

                xmm_tempa = _mm_add_epi32( xmm_tempa, xmm_product1 );
                xmm_tempb = _mm_add_epi32( xmm_tempb, xmm_product2 );
                xmm_tempa = _mm_add_epi32( xmm_tempa, xmm_tempb );

                sum  = (coef_Q13_8 * state_8) >> 16;
                sum += (coef_Q13_9 * state_9) >> 16;

                xmm_tempa = _mm_add_epi32( xmm_tempa, _mm_shuffle_epi32( xmm_tempa, _MM_SHUFFLE( 0, 0, 0, 2 ) ) );
                sum += _mm_cvtsi128_si32( xmm_tempa);
                res_Q2[ n ] = silk_LSHIFT( (opus_int32)input[ n ], 2 ) - silk_RSHIFT_ROUND( ( 5 + sum ), 9);

                /* move right */
                state_a = state_9;
                state_9 = state_8;
                state_8 = _mm_cvtsi128_si32( state_4567 );
                state_4567 = _mm_alignr_epi8( state_0123, state_4567, 4 );

                state_0123 = _mm_alignr_epi8( _mm_cvtsi32_si128( silk_LSHIFT( input[ n ], 14 ) ), state_0123, 4 );
            }

            _mm_storeu_si128( (__m128i *)( &state[ 0 ] ), _mm_shuffle_epi32( state_0123, _MM_SHUFFLE( 0, 1, 2, 3 ) ) );
            _mm_storeu_si128( (__m128i *)( &state[ 4 ] ), _mm_shuffle_epi32( state_4567, _MM_SHUFFLE( 0, 1, 2, 3 ) ) );
            state[ 8 ] = state_8;
            state[ 9 ] = state_9;
            state[ 10 ] = state_a;

            return;
        }
    }

    for( n = 0; n < length; n++ ) {
        /* Output of lowpass section */
        tmp2 = silk_SMLAWB( state[ 0 ], state[ 1 ], lambda_Q16 );
        state[ 0 ] = silk_LSHIFT( input[ n ], 14 );
        /* Output of allpass section */
        tmp1 = silk_SMLAWB( state[ 1 ], state[ 2 ] - tmp2, lambda_Q16 );
        state[ 1 ] = tmp2;
        acc_Q11 = silk_RSHIFT( order, 1 );
        acc_Q11 = silk_SMLAWB( acc_Q11, tmp2, coef_Q13[ 0 ] );
        /* Loop over allpass sections */
        for( i = 2; i < order; i += 2 ) {
            /* Output of allpass section */
            tmp2 = silk_SMLAWB( state[ i ], state[ i + 1 ] - tmp1, lambda_Q16 );
            state[ i ] = tmp1;
            acc_Q11 = silk_SMLAWB( acc_Q11, tmp1, coef_Q13[ i - 1 ] );
            /* Output of allpass section */
            tmp1 = silk_SMLAWB( state[ i + 1 ], state[ i + 2 ] - tmp2, lambda_Q16 );
            state[ i + 1 ] = tmp2;
            acc_Q11 = silk_SMLAWB( acc_Q11, tmp2, coef_Q13[ i ] );
        }
        state[ order ] = tmp1;
        acc_Q11 = silk_SMLAWB( acc_Q11, tmp1, coef_Q13[ order - 1 ] );
        res_Q2[ n ] = silk_LSHIFT( (opus_int32)input[ n ], 2 ) - silk_RSHIFT_ROUND( acc_Q11, 9 );
    }
}
