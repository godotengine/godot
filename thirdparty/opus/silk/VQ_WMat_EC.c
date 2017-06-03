/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "main.h"

/* Entropy constrained matrix-weighted VQ, hard-coded to 5-element vectors, for a single input data vector */
void silk_VQ_WMat_EC_c(
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

    /* Loop over codebook */
    *rate_dist_Q14 = silk_int32_MAX;
    cb_row_Q7 = cb_Q7;
    for( k = 0; k < L; k++ ) {
        gain_tmp_Q7 = cb_gain_Q7[k];

        diff_Q14[ 0 ] = in_Q14[ 0 ] - silk_LSHIFT( cb_row_Q7[ 0 ], 7 );
        diff_Q14[ 1 ] = in_Q14[ 1 ] - silk_LSHIFT( cb_row_Q7[ 1 ], 7 );
        diff_Q14[ 2 ] = in_Q14[ 2 ] - silk_LSHIFT( cb_row_Q7[ 2 ], 7 );
        diff_Q14[ 3 ] = in_Q14[ 3 ] - silk_LSHIFT( cb_row_Q7[ 3 ], 7 );
        diff_Q14[ 4 ] = in_Q14[ 4 ] - silk_LSHIFT( cb_row_Q7[ 4 ], 7 );

        /* Weighted rate */
        sum1_Q14 = silk_SMULBB( mu_Q9, cl_Q5[ k ] );

        /* Penalty for too large gain */
        sum1_Q14 = silk_ADD_LSHIFT32( sum1_Q14, silk_max( silk_SUB32( gain_tmp_Q7, max_gain_Q7 ), 0 ), 10 );

        silk_assert( sum1_Q14 >= 0 );

        /* first row of W_Q18 */
        sum2_Q16 = silk_SMULWB(           W_Q18[  1 ], diff_Q14[ 1 ] );
        sum2_Q16 = silk_SMLAWB( sum2_Q16, W_Q18[  2 ], diff_Q14[ 2 ] );
        sum2_Q16 = silk_SMLAWB( sum2_Q16, W_Q18[  3 ], diff_Q14[ 3 ] );
        sum2_Q16 = silk_SMLAWB( sum2_Q16, W_Q18[  4 ], diff_Q14[ 4 ] );
        sum2_Q16 = silk_LSHIFT( sum2_Q16, 1 );
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
