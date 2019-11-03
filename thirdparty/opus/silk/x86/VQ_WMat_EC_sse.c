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

/* Entropy constrained matrix-weighted VQ, hard-coded to 5-element vectors, for a single input data vector */
void silk_VQ_WMat_EC_sse4_1(
    opus_int8                   *ind,                           /* O    index of best codebook vector               */
    opus_int32                  *rate_dist_Q14,                 /* O    best weighted quant error + mu * rate       */
    opus_int                    *gain_Q7,                       /* O    sum of absolute LTP coefficients            */
    const opus_int16            *in_Q14,                        /* I    input vector to be quantized                */
    const opus_int32            *W_Q18,                         /* I    weighting matrix                            */
    const opus_int8             *cb_Q7,                         /* I    codebook                                    */
    const opus_uint8            *cb_gain_Q7,                    /* I    codebook effective gain                     */
    const opus_uint8            *cl_Q5,                         /* I    code length for each codebook vector        */
    const opus_int              mu_Q9,                          /* I    tradeoff betw. weighted error and rate      */
    const opus_int32            max_gain_Q7,                    /* I    maximum sum of absolute LTP coefficients    */
    opus_int                    L                               /* I    number of vectors in codebook               */
)
{
    opus_int   k, gain_tmp_Q7;
    const opus_int8 *cb_row_Q7;
    opus_int16 diff_Q14[ 5 ];
    opus_int32 sum1_Q14, sum2_Q16;

    __m128i C_tmp1, C_tmp2, C_tmp3, C_tmp4, C_tmp5;
    /* Loop over codebook */
    *rate_dist_Q14 = silk_int32_MAX;
    cb_row_Q7 = cb_Q7;
    for( k = 0; k < L; k++ ) {
        gain_tmp_Q7 = cb_gain_Q7[k];

        diff_Q14[ 0 ] = in_Q14[ 0 ] - silk_LSHIFT( cb_row_Q7[ 0 ], 7 );

        C_tmp1 = OP_CVTEPI16_EPI32_M64( &in_Q14[ 1 ] );
        C_tmp2 = OP_CVTEPI8_EPI32_M32( &cb_row_Q7[ 1 ] );
        C_tmp2 = _mm_slli_epi32( C_tmp2, 7 );
        C_tmp1 = _mm_sub_epi32( C_tmp1, C_tmp2 );

        diff_Q14[ 1 ] = _mm_extract_epi16( C_tmp1, 0 );
        diff_Q14[ 2 ] = _mm_extract_epi16( C_tmp1, 2 );
        diff_Q14[ 3 ] = _mm_extract_epi16( C_tmp1, 4 );
        diff_Q14[ 4 ] = _mm_extract_epi16( C_tmp1, 6 );

        /* Weighted rate */
        sum1_Q14 = silk_SMULBB( mu_Q9, cl_Q5[ k ] );

        /* Penalty for too large gain */
        sum1_Q14 = silk_ADD_LSHIFT32( sum1_Q14, silk_max( silk_SUB32( gain_tmp_Q7, max_gain_Q7 ), 0 ), 10 );

        silk_assert( sum1_Q14 >= 0 );

        /* first row of W_Q18 */
        C_tmp3 = _mm_loadu_si128( (__m128i *)(&W_Q18[ 1 ] ) );
        C_tmp4 = _mm_mul_epi32( C_tmp3, C_tmp1 );
        C_tmp4 = _mm_srli_si128( C_tmp4, 2 );

        C_tmp1 = _mm_shuffle_epi32( C_tmp1, _MM_SHUFFLE( 0, 3, 2, 1 ) ); /* shift right 4 bytes */
        C_tmp3 = _mm_shuffle_epi32( C_tmp3, _MM_SHUFFLE( 0, 3, 2, 1 ) ); /* shift right 4 bytes */

        C_tmp5 = _mm_mul_epi32( C_tmp3, C_tmp1 );
        C_tmp5 = _mm_srli_si128( C_tmp5, 2 );

        C_tmp5 = _mm_add_epi32( C_tmp4, C_tmp5 );
        C_tmp5 = _mm_slli_epi32( C_tmp5, 1 );

        C_tmp5 = _mm_add_epi32( C_tmp5, _mm_shuffle_epi32( C_tmp5, _MM_SHUFFLE( 0, 0, 0, 2 ) ) );
        sum2_Q16 = _mm_cvtsi128_si32( C_tmp5 );

        sum2_Q16 = silk_SMLAWB( sum2_Q16, W_Q18[  0 ], diff_Q14[ 0 ] );
        sum1_Q14 = silk_SMLAWB( sum1_Q14, sum2_Q16,    diff_Q14[ 0 ] );

        /* second row of W_Q18 */
        sum2_Q16 = silk_SMULWB(           W_Q18[  7 ], diff_Q14[ 2 ] );
        sum2_Q16 = silk_SMLAWB( sum2_Q16, W_Q18[  8 ], diff_Q14[ 3 ] );
        sum2_Q16 = silk_SMLAWB( sum2_Q16, W_Q18[  9 ], diff_Q14[ 4 ] );
        sum2_Q16 = silk_LSHIFT( sum2_Q16, 1 );
        sum2_Q16 = silk_SMLAWB( sum2_Q16, W_Q18[  6 ], diff_Q14[ 1 ] );
        sum1_Q14 = silk_SMLAWB( sum1_Q14, sum2_Q16,    diff_Q14[ 1 ] );

        /* third row of W_Q18 */
        sum2_Q16 = silk_SMULWB(           W_Q18[ 13 ], diff_Q14[ 3 ] );
        sum2_Q16 = silk_SMLAWB( sum2_Q16, W_Q18[ 14 ], diff_Q14[ 4 ] );
        sum2_Q16 = silk_LSHIFT( sum2_Q16, 1 );
        sum2_Q16 = silk_SMLAWB( sum2_Q16, W_Q18[ 12 ], diff_Q14[ 2 ] );
        sum1_Q14 = silk_SMLAWB( sum1_Q14, sum2_Q16,    diff_Q14[ 2 ] );

        /* fourth row of W_Q18 */
        sum2_Q16 = silk_SMULWB(           W_Q18[ 19 ], diff_Q14[ 4 ] );
        sum2_Q16 = silk_LSHIFT( sum2_Q16, 1 );
        sum2_Q16 = silk_SMLAWB( sum2_Q16, W_Q18[ 18 ], diff_Q14[ 3 ] );
        sum1_Q14 = silk_SMLAWB( sum1_Q14, sum2_Q16,    diff_Q14[ 3 ] );

        /* last row of W_Q18 */
        sum2_Q16 = silk_SMULWB(           W_Q18[ 24 ], diff_Q14[ 4 ] );
        sum1_Q14 = silk_SMLAWB( sum1_Q14, sum2_Q16,    diff_Q14[ 4 ] );

        silk_assert( sum1_Q14 >= 0 );

        /* find best */
        if( sum1_Q14 < *rate_dist_Q14 ) {
            *rate_dist_Q14 = sum1_Q14;
            *ind = (opus_int8)k;
            *gain_Q7 = gain_tmp_Q7;
        }

        /* Go to next cbk vector */
        cb_row_Q7 += LTP_ORDER;
    }
}
