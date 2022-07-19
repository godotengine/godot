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
#include "tuning_parameters.h"

void silk_quant_LTP_gains(
    opus_int16                  B_Q14[ MAX_NB_SUBFR * LTP_ORDER ],          /* I/O  (un)quantized LTP gains         */
    opus_int8                   cbk_index[ MAX_NB_SUBFR ],                  /* O    Codebook Index                  */
    opus_int8                   *periodicity_index,                         /* O    Periodicity Index               */
    opus_int32                  *sum_log_gain_Q7,                           /* I/O  Cumulative max prediction gain  */
    const opus_int32            W_Q18[ MAX_NB_SUBFR*LTP_ORDER*LTP_ORDER ],  /* I    Error Weights in Q18            */
    opus_int                    mu_Q9,                                      /* I    Mu value (R/D tradeoff)         */
    opus_int                    lowComplexity,                              /* I    Flag for low complexity         */
    const opus_int              nb_subfr,                                   /* I    number of subframes             */
    int                         arch                                        /* I    Run-time architecture           */
)
{
    opus_int             j, k, cbk_size;
    opus_int8            temp_idx[ MAX_NB_SUBFR ];
    const opus_uint8     *cl_ptr_Q5;
    const opus_int8      *cbk_ptr_Q7;
    const opus_uint8     *cbk_gain_ptr_Q7;
    const opus_int16     *b_Q14_ptr;
    const opus_int32     *W_Q18_ptr;
    opus_int32           rate_dist_Q14_subfr, rate_dist_Q14, min_rate_dist_Q14;
    opus_int32           sum_log_gain_tmp_Q7, best_sum_log_gain_Q7, max_gain_Q7, gain_Q7;

    /***************************************************/
    /* iterate over different codebooks with different */
    /* rates/distortions, and choose best */
    /***************************************************/
    min_rate_dist_Q14 = silk_int32_MAX;
    best_sum_log_gain_Q7 = 0;
    for( k = 0; k < 3; k++ ) {
        /* Safety margin for pitch gain control, to take into account factors
           such as state rescaling/rewhitening. */
        opus_int32 gain_safety = SILK_FIX_CONST( 0.4, 7 );

        cl_ptr_Q5  = silk_LTP_gain_BITS_Q5_ptrs[ k ];
        cbk_ptr_Q7 = silk_LTP_vq_ptrs_Q7[        k ];
        cbk_gain_ptr_Q7 = silk_LTP_vq_gain_ptrs_Q7[ k ];
        cbk_size   = silk_LTP_vq_sizes[          k ];

        /* Set up pointer to first subframe */
        W_Q18_ptr = W_Q18;
        b_Q14_ptr = B_Q14;

        rate_dist_Q14 = 0;
        sum_log_gain_tmp_Q7 = *sum_log_gain_Q7;
        for( j = 0; j < nb_subfr; j++ ) {
            max_gain_Q7 = silk_log2lin( ( SILK_FIX_CONST( MAX_SUM_LOG_GAIN_DB / 6.0, 7 ) - sum_log_gain_tmp_Q7 )
                                        + SILK_FIX_CONST( 7, 7 ) ) - gain_safety;

            silk_VQ_WMat_EC(
                &temp_idx[ j ],         /* O    index of best codebook vector                           */
                &rate_dist_Q14_subfr,   /* O    best weighted quantization error + mu * rate            */
                &gain_Q7,               /* O    sum of absolute LTP coefficients                        */
                b_Q14_ptr,              /* I    input vector to be quantized                            */
                W_Q18_ptr,              /* I    weighting matrix                                        */
                cbk_ptr_Q7,             /* I    codebook                                                */
                cbk_gain_ptr_Q7,        /* I    codebook effective gains                                */
                cl_ptr_Q5,              /* I    code length for each codebook vector                    */
                mu_Q9,                  /* I    tradeoff between weighted error and rate                */
                max_gain_Q7,            /* I    maximum sum of absolute LTP coefficients                */
                cbk_size,               /* I    number of vectors in codebook                           */
                arch                    /* I    Run-time architecture                                   */
            );

            rate_dist_Q14 = silk_ADD_POS_SAT32( rate_dist_Q14, rate_dist_Q14_subfr );
            sum_log_gain_tmp_Q7 = silk_max(0, sum_log_gain_tmp_Q7
                                + silk_lin2log( gain_safety + gain_Q7 ) - SILK_FIX_CONST( 7, 7 ));

            b_Q14_ptr += LTP_ORDER;
            W_Q18_ptr += LTP_ORDER * LTP_ORDER;
        }

        /* Avoid never finding a codebook */
        rate_dist_Q14 = silk_min( silk_int32_MAX - 1, rate_dist_Q14 );

        if( rate_dist_Q14 < min_rate_dist_Q14 ) {
            min_rate_dist_Q14 = rate_dist_Q14;
            *periodicity_index = (opus_int8)k;
            silk_memcpy( cbk_index, temp_idx, nb_subfr * sizeof( opus_int8 ) );
            best_sum_log_gain_Q7 = sum_log_gain_tmp_Q7;
        }

        /* Break early in low-complexity mode if rate distortion is below threshold */
        if( lowComplexity && ( rate_dist_Q14 < silk_LTP_gain_middle_avg_RD_Q14 ) ) {
            break;
        }
    }

    cbk_ptr_Q7 = silk_LTP_vq_ptrs_Q7[ *periodicity_index ];
    for( j = 0; j < nb_subfr; j++ ) {
        for( k = 0; k < LTP_ORDER; k++ ) {
            B_Q14[ j * LTP_ORDER + k ] = silk_LSHIFT( cbk_ptr_Q7[ cbk_index[ j ] * LTP_ORDER + k ], 7 );
        }
    }
    *sum_log_gain_Q7 = best_sum_log_gain_Q7;
}
