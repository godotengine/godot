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

#ifndef __NSQ_DEL_DEC_MIPSR1_H__
#define __NSQ_DEL_DEC_MIPSR1_H__

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "main.h"
#include "stack_alloc.h"

#define OVERRIDE_silk_noise_shape_quantizer_del_dec
static inline void silk_noise_shape_quantizer_del_dec(
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                           */
    NSQ_del_dec_struct  psDelDec[],             /* I/O  Delayed decision states             */
    opus_int            signalType,             /* I    Signal type                         */
    const opus_int32    x_Q10[],                /* I                                        */
    opus_int8           pulses[],               /* O                                        */
    opus_int16          xq[],                   /* O                                        */
    opus_int32          sLTP_Q15[],             /* I/O  LTP filter state                    */
    opus_int32          delayedGain_Q10[],      /* I/O  Gain delay buffer                   */
    const opus_int16    a_Q12[],                /* I    Short term prediction coefs         */
    const opus_int16    b_Q14[],                /* I    Long term prediction coefs          */
    const opus_int16    AR_shp_Q13[],           /* I    Noise shaping coefs                 */
    opus_int            lag,                    /* I    Pitch lag                           */
    opus_int32          HarmShapeFIRPacked_Q14, /* I                                        */
    opus_int            Tilt_Q14,               /* I    Spectral tilt                       */
    opus_int32          LF_shp_Q14,             /* I                                        */
    opus_int32          Gain_Q16,               /* I                                        */
    opus_int            Lambda_Q10,             /* I                                        */
    opus_int            offset_Q10,             /* I                                        */
    opus_int            length,                 /* I    Input length                        */
    opus_int            subfr,                  /* I    Subframe number                     */
    opus_int            shapingLPCOrder,        /* I    Shaping LPC filter order            */
    opus_int            predictLPCOrder,        /* I    Prediction filter order             */
    opus_int            warping_Q16,            /* I                                        */
    opus_int            nStatesDelayedDecision, /* I    Number of states in decision tree   */
    opus_int            *smpl_buf_idx,          /* I    Index to newest samples in buffers  */
    opus_int            decisionDelay,          /* I                                        */
    int                 arch                    /* I                                        */
)
{
    opus_int     i, j, k, Winner_ind, RDmin_ind, RDmax_ind, last_smple_idx;
    opus_int32   Winner_rand_state;
    opus_int32   LTP_pred_Q14, LPC_pred_Q14, n_AR_Q14, n_LTP_Q14;
    opus_int32   n_LF_Q14, r_Q10, rr_Q10, rd1_Q10, rd2_Q10, RDmin_Q10, RDmax_Q10;
    opus_int32   q1_Q0, q1_Q10, q2_Q10, exc_Q14, LPC_exc_Q14, xq_Q14, Gain_Q10;
    opus_int32   tmp1, tmp2, sLF_AR_shp_Q14;
    opus_int32   *pred_lag_ptr, *shp_lag_ptr, *psLPC_Q14;
    NSQ_sample_struct  psSampleState[ MAX_DEL_DEC_STATES ][ 2 ];
    NSQ_del_dec_struct *psDD;
    NSQ_sample_struct  *psSS;
    opus_int16 b_Q14_0, b_Q14_1, b_Q14_2, b_Q14_3, b_Q14_4;
    opus_int16 a_Q12_0, a_Q12_1, a_Q12_2, a_Q12_3, a_Q12_4, a_Q12_5, a_Q12_6;
    opus_int16 a_Q12_7, a_Q12_8, a_Q12_9, a_Q12_10, a_Q12_11, a_Q12_12, a_Q12_13;
    opus_int16 a_Q12_14, a_Q12_15;

    opus_int32 cur, prev, next;

    /*Unused.*/
    (void)arch;

    //Intialize b_Q14 variables
    b_Q14_0 = b_Q14[ 0 ];
    b_Q14_1 = b_Q14[ 1 ];
    b_Q14_2 = b_Q14[ 2 ];
    b_Q14_3 = b_Q14[ 3 ];
    b_Q14_4 = b_Q14[ 4 ];

    //Intialize a_Q12 variables
    a_Q12_0 = a_Q12[0];
    a_Q12_1 = a_Q12[1];
    a_Q12_2 = a_Q12[2];
    a_Q12_3 = a_Q12[3];
    a_Q12_4 = a_Q12[4];
    a_Q12_5 = a_Q12[5];
    a_Q12_6 = a_Q12[6];
    a_Q12_7 = a_Q12[7];
    a_Q12_8 = a_Q12[8];
    a_Q12_9 = a_Q12[9];
    a_Q12_10 = a_Q12[10];
    a_Q12_11 = a_Q12[11];
    a_Q12_12 = a_Q12[12];
    a_Q12_13 = a_Q12[13];
    a_Q12_14 = a_Q12[14];
    a_Q12_15 = a_Q12[15];

    long long temp64;

    silk_assert( nStatesDelayedDecision > 0 );

    shp_lag_ptr  = &NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2 ];
    pred_lag_ptr = &sLTP_Q15[ NSQ->sLTP_buf_idx - lag + LTP_ORDER / 2 ];
    Gain_Q10     = silk_RSHIFT( Gain_Q16, 6 );

    for( i = 0; i < length; i++ ) {
        /* Perform common calculations used in all states */

        /* Long-term prediction */
        if( signalType == TYPE_VOICED ) {
            /* Unrolled loop */
            /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
            temp64 = __builtin_mips_mult(pred_lag_ptr[ 0 ], b_Q14_0 );
            temp64 = __builtin_mips_madd( temp64, pred_lag_ptr[ -1 ], b_Q14_1 );
            temp64 = __builtin_mips_madd( temp64, pred_lag_ptr[ -2 ], b_Q14_2 );
            temp64 = __builtin_mips_madd( temp64, pred_lag_ptr[ -3 ], b_Q14_3 );
            temp64 = __builtin_mips_madd( temp64, pred_lag_ptr[ -4 ], b_Q14_4 );
            temp64 += 32768;
            LTP_pred_Q14 = __builtin_mips_extr_w(temp64, 16);
            LTP_pred_Q14 = silk_LSHIFT( LTP_pred_Q14, 1 );                          /* Q13 -> Q14 */
            pred_lag_ptr++;
        } else {
            LTP_pred_Q14 = 0;
        }

        /* Long-term shaping */
        if( lag > 0 ) {
            /* Symmetric, packed FIR coefficients */
            n_LTP_Q14 = silk_SMULWB( silk_ADD32( shp_lag_ptr[ 0 ], shp_lag_ptr[ -2 ] ), HarmShapeFIRPacked_Q14 );
            n_LTP_Q14 = silk_SMLAWT( n_LTP_Q14, shp_lag_ptr[ -1 ],                      HarmShapeFIRPacked_Q14 );
            n_LTP_Q14 = silk_SUB_LSHIFT32( LTP_pred_Q14, n_LTP_Q14, 2 );            /* Q12 -> Q14 */
            shp_lag_ptr++;
        } else {
            n_LTP_Q14 = 0;
        }

        for( k = 0; k < nStatesDelayedDecision; k++ ) {
            /* Delayed decision state */
            psDD = &psDelDec[ k ];

            /* Sample state */
            psSS = psSampleState[ k ];

            /* Generate dither */
            psDD->Seed = silk_RAND( psDD->Seed );

            /* Pointer used in short term prediction and shaping */
            psLPC_Q14 = &psDD->sLPC_Q14[ NSQ_LPC_BUF_LENGTH - 1 + i ];
            /* Short-term prediction */
            silk_assert( predictLPCOrder == 10 || predictLPCOrder == 16 );
            temp64 = __builtin_mips_mult(psLPC_Q14[  0 ], a_Q12_0 );
            temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -1 ], a_Q12_1 );
            temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -2 ], a_Q12_2 );
            temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -3 ], a_Q12_3 );
            temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -4 ], a_Q12_4 );
            temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -5 ], a_Q12_5 );
            temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -6 ], a_Q12_6 );
            temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -7 ], a_Q12_7 );
            temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -8 ], a_Q12_8 );
            temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -9 ], a_Q12_9 );
            if( predictLPCOrder == 16 ) {
                temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -10 ], a_Q12_10 );
                temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -11 ], a_Q12_11 );
                temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -12 ], a_Q12_12 );
                temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -13 ], a_Q12_13 );
                temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -14 ], a_Q12_14 );
                temp64 = __builtin_mips_madd( temp64, psLPC_Q14[ -15 ], a_Q12_15 );
            }
            temp64 += 32768;
            LPC_pred_Q14 = __builtin_mips_extr_w(temp64, 16);

            LPC_pred_Q14 = silk_LSHIFT( LPC_pred_Q14, 4 );                              /* Q10 -> Q14 */

            /* Noise shape feedback */
            silk_assert( ( shapingLPCOrder & 1 ) == 0 );   /* check that order is even */
            /* Output of lowpass section */
            tmp2 = silk_SMLAWB( psLPC_Q14[ 0 ], psDD->sAR2_Q14[ 0 ], warping_Q16 );
            /* Output of allpass section */
            tmp1 = silk_SMLAWB( psDD->sAR2_Q14[ 0 ], psDD->sAR2_Q14[ 1 ] - tmp2, warping_Q16 );
            psDD->sAR2_Q14[ 0 ] = tmp2;

            temp64 = __builtin_mips_mult(tmp2, AR_shp_Q13[ 0 ] );

            prev = psDD->sAR2_Q14[ 1 ];

            /* Loop over allpass sections */
            for( j = 2; j < shapingLPCOrder; j += 2 ) {
                cur = psDD->sAR2_Q14[ j ];
                next = psDD->sAR2_Q14[ j+1 ];
                /* Output of allpass section */
                tmp2 = silk_SMLAWB( prev, cur - tmp1, warping_Q16 );
                psDD->sAR2_Q14[ j - 1 ] = tmp1;
                temp64 = __builtin_mips_madd( temp64, tmp1, AR_shp_Q13[ j - 1 ] );
                temp64 = __builtin_mips_madd( temp64, tmp2, AR_shp_Q13[ j ] );
                /* Output of allpass section */
                tmp1 = silk_SMLAWB( cur, next - tmp2, warping_Q16 );
                psDD->sAR2_Q14[ j + 0 ] = tmp2;
                prev = next;
            }
            psDD->sAR2_Q14[ shapingLPCOrder - 1 ] = tmp1;
            temp64 = __builtin_mips_madd( temp64, tmp1, AR_shp_Q13[ shapingLPCOrder - 1 ] );
            temp64 += 32768;
            n_AR_Q14 = __builtin_mips_extr_w(temp64, 16);
            n_AR_Q14 = silk_LSHIFT( n_AR_Q14, 1 );                                      /* Q11 -> Q12 */
            n_AR_Q14 = silk_SMLAWB( n_AR_Q14, psDD->LF_AR_Q14, Tilt_Q14 );              /* Q12 */
            n_AR_Q14 = silk_LSHIFT( n_AR_Q14, 2 );                                      /* Q12 -> Q14 */

            n_LF_Q14 = silk_SMULWB( psDD->Shape_Q14[ *smpl_buf_idx ], LF_shp_Q14 );     /* Q12 */
            n_LF_Q14 = silk_SMLAWT( n_LF_Q14, psDD->LF_AR_Q14, LF_shp_Q14 );            /* Q12 */
            n_LF_Q14 = silk_LSHIFT( n_LF_Q14, 2 );                                      /* Q12 -> Q14 */

            /* Input minus prediction plus noise feedback                       */
            /* r = x[ i ] - LTP_pred - LPC_pred + n_AR + n_Tilt + n_LF + n_LTP  */
            tmp1 = silk_ADD32( n_AR_Q14, n_LF_Q14 );                                    /* Q14 */
            tmp2 = silk_ADD32( n_LTP_Q14, LPC_pred_Q14 );                               /* Q13 */
            tmp1 = silk_SUB32( tmp2, tmp1 );                                            /* Q13 */
            tmp1 = silk_RSHIFT_ROUND( tmp1, 4 );                                        /* Q10 */

            r_Q10 = silk_SUB32( x_Q10[ i ], tmp1 );                                     /* residual error Q10 */

            /* Flip sign depending on dither */
            if ( psDD->Seed < 0 ) {
                r_Q10 = -r_Q10;
            }
            r_Q10 = silk_LIMIT_32( r_Q10, -(31 << 10), 30 << 10 );

            /* Find two quantization level candidates and measure their rate-distortion */
            q1_Q10 = silk_SUB32( r_Q10, offset_Q10 );
            q1_Q0 = silk_RSHIFT( q1_Q10, 10 );
            if( q1_Q0 > 0 ) {
                q1_Q10  = silk_SUB32( silk_LSHIFT( q1_Q0, 10 ), QUANT_LEVEL_ADJUST_Q10 );
                q1_Q10  = silk_ADD32( q1_Q10, offset_Q10 );
                q2_Q10  = silk_ADD32( q1_Q10, 1024 );
                rd1_Q10 = silk_SMULBB( q1_Q10, Lambda_Q10 );
                rd2_Q10 = silk_SMULBB( q2_Q10, Lambda_Q10 );
            } else if( q1_Q0 == 0 ) {
                q1_Q10  = offset_Q10;
                q2_Q10  = silk_ADD32( q1_Q10, 1024 - QUANT_LEVEL_ADJUST_Q10 );
                rd1_Q10 = silk_SMULBB( q1_Q10, Lambda_Q10 );
                rd2_Q10 = silk_SMULBB( q2_Q10, Lambda_Q10 );
            } else if( q1_Q0 == -1 ) {
                q2_Q10  = offset_Q10;
                q1_Q10  = silk_SUB32( q2_Q10, 1024 - QUANT_LEVEL_ADJUST_Q10 );
                rd1_Q10 = silk_SMULBB( -q1_Q10, Lambda_Q10 );
                rd2_Q10 = silk_SMULBB(  q2_Q10, Lambda_Q10 );
            } else {            /* q1_Q0 < -1 */
                q1_Q10  = silk_ADD32( silk_LSHIFT( q1_Q0, 10 ), QUANT_LEVEL_ADJUST_Q10 );
                q1_Q10  = silk_ADD32( q1_Q10, offset_Q10 );
                q2_Q10  = silk_ADD32( q1_Q10, 1024 );
                rd1_Q10 = silk_SMULBB( -q1_Q10, Lambda_Q10 );
                rd2_Q10 = silk_SMULBB( -q2_Q10, Lambda_Q10 );
            }
            rr_Q10  = silk_SUB32( r_Q10, q1_Q10 );
            rd1_Q10 = silk_RSHIFT( silk_SMLABB( rd1_Q10, rr_Q10, rr_Q10 ), 10 );
            rr_Q10  = silk_SUB32( r_Q10, q2_Q10 );
            rd2_Q10 = silk_RSHIFT( silk_SMLABB( rd2_Q10, rr_Q10, rr_Q10 ), 10 );

            if( rd1_Q10 < rd2_Q10 ) {
                psSS[ 0 ].RD_Q10 = silk_ADD32( psDD->RD_Q10, rd1_Q10 );
                psSS[ 1 ].RD_Q10 = silk_ADD32( psDD->RD_Q10, rd2_Q10 );
                psSS[ 0 ].Q_Q10  = q1_Q10;
                psSS[ 1 ].Q_Q10  = q2_Q10;
            } else {
                psSS[ 0 ].RD_Q10 = silk_ADD32( psDD->RD_Q10, rd2_Q10 );
                psSS[ 1 ].RD_Q10 = silk_ADD32( psDD->RD_Q10, rd1_Q10 );
                psSS[ 0 ].Q_Q10  = q2_Q10;
                psSS[ 1 ].Q_Q10  = q1_Q10;
            }

            /* Update states for best quantization */

            /* Quantized excitation */
            exc_Q14 = silk_LSHIFT32( psSS[ 0 ].Q_Q10, 4 );
            if ( psDD->Seed < 0 ) {
                exc_Q14 = -exc_Q14;
            }

            /* Add predictions */
            LPC_exc_Q14 = silk_ADD32( exc_Q14, LTP_pred_Q14 );
            xq_Q14      = silk_ADD32( LPC_exc_Q14, LPC_pred_Q14 );

            /* Update states */
            sLF_AR_shp_Q14         = silk_SUB32( xq_Q14, n_AR_Q14 );
            psSS[ 0 ].sLTP_shp_Q14 = silk_SUB32( sLF_AR_shp_Q14, n_LF_Q14 );
            psSS[ 0 ].LF_AR_Q14    = sLF_AR_shp_Q14;
            psSS[ 0 ].LPC_exc_Q14  = LPC_exc_Q14;
            psSS[ 0 ].xq_Q14       = xq_Q14;

            /* Update states for second best quantization */

            /* Quantized excitation */
            exc_Q14 = silk_LSHIFT32( psSS[ 1 ].Q_Q10, 4 );
            if ( psDD->Seed < 0 ) {
                exc_Q14 = -exc_Q14;
            }


            /* Add predictions */
            LPC_exc_Q14 = silk_ADD32( exc_Q14, LTP_pred_Q14 );
            xq_Q14      = silk_ADD32( LPC_exc_Q14, LPC_pred_Q14 );

            /* Update states */
            sLF_AR_shp_Q14         = silk_SUB32( xq_Q14, n_AR_Q14 );
            psSS[ 1 ].sLTP_shp_Q14 = silk_SUB32( sLF_AR_shp_Q14, n_LF_Q14 );
            psSS[ 1 ].LF_AR_Q14    = sLF_AR_shp_Q14;
            psSS[ 1 ].LPC_exc_Q14  = LPC_exc_Q14;
            psSS[ 1 ].xq_Q14       = xq_Q14;
        }

        *smpl_buf_idx  = ( *smpl_buf_idx - 1 ) & DECISION_DELAY_MASK;                   /* Index to newest samples              */
        last_smple_idx = ( *smpl_buf_idx + decisionDelay ) & DECISION_DELAY_MASK;       /* Index to decisionDelay old samples   */

        /* Find winner */
        RDmin_Q10 = psSampleState[ 0 ][ 0 ].RD_Q10;
        Winner_ind = 0;
        for( k = 1; k < nStatesDelayedDecision; k++ ) {
            if( psSampleState[ k ][ 0 ].RD_Q10 < RDmin_Q10 ) {
                RDmin_Q10  = psSampleState[ k ][ 0 ].RD_Q10;
                Winner_ind = k;
            }
        }

        /* Increase RD values of expired states */
        Winner_rand_state = psDelDec[ Winner_ind ].RandState[ last_smple_idx ];
        for( k = 0; k < nStatesDelayedDecision; k++ ) {
            if( psDelDec[ k ].RandState[ last_smple_idx ] != Winner_rand_state ) {
                psSampleState[ k ][ 0 ].RD_Q10 = silk_ADD32( psSampleState[ k ][ 0 ].RD_Q10, silk_int32_MAX >> 4 );
                psSampleState[ k ][ 1 ].RD_Q10 = silk_ADD32( psSampleState[ k ][ 1 ].RD_Q10, silk_int32_MAX >> 4 );
                silk_assert( psSampleState[ k ][ 0 ].RD_Q10 >= 0 );
            }
        }

        /* Find worst in first set and best in second set */
        RDmax_Q10  = psSampleState[ 0 ][ 0 ].RD_Q10;
        RDmin_Q10  = psSampleState[ 0 ][ 1 ].RD_Q10;
        RDmax_ind = 0;
        RDmin_ind = 0;
        for( k = 1; k < nStatesDelayedDecision; k++ ) {
            /* find worst in first set */
            if( psSampleState[ k ][ 0 ].RD_Q10 > RDmax_Q10 ) {
                RDmax_Q10  = psSampleState[ k ][ 0 ].RD_Q10;
                RDmax_ind = k;
            }
            /* find best in second set */
            if( psSampleState[ k ][ 1 ].RD_Q10 < RDmin_Q10 ) {
                RDmin_Q10  = psSampleState[ k ][ 1 ].RD_Q10;
                RDmin_ind = k;
            }
        }

        /* Replace a state if best from second set outperforms worst in first set */
        if( RDmin_Q10 < RDmax_Q10 ) {
            silk_memcpy( ( (opus_int32 *)&psDelDec[ RDmax_ind ] ) + i,
                         ( (opus_int32 *)&psDelDec[ RDmin_ind ] ) + i, sizeof( NSQ_del_dec_struct ) - i * sizeof( opus_int32) );
            silk_memcpy( &psSampleState[ RDmax_ind ][ 0 ], &psSampleState[ RDmin_ind ][ 1 ], sizeof( NSQ_sample_struct ) );
        }

        /* Write samples from winner to output and long-term filter states */
        psDD = &psDelDec[ Winner_ind ];
        if( subfr > 0 || i >= decisionDelay ) {
            pulses[  i - decisionDelay ] = (opus_int8)silk_RSHIFT_ROUND( psDD->Q_Q10[ last_smple_idx ], 10 );
            xq[ i - decisionDelay ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND(
                silk_SMULWW( psDD->Xq_Q14[ last_smple_idx ], delayedGain_Q10[ last_smple_idx ] ), 8 ) );
            NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - decisionDelay ] = psDD->Shape_Q14[ last_smple_idx ];
            sLTP_Q15[          NSQ->sLTP_buf_idx     - decisionDelay ] = psDD->Pred_Q15[  last_smple_idx ];
        }
        NSQ->sLTP_shp_buf_idx++;
        NSQ->sLTP_buf_idx++;

        /* Update states */
        for( k = 0; k < nStatesDelayedDecision; k++ ) {
            psDD                                     = &psDelDec[ k ];
            psSS                                     = &psSampleState[ k ][ 0 ];
            psDD->LF_AR_Q14                          = psSS->LF_AR_Q14;
            psDD->sLPC_Q14[ NSQ_LPC_BUF_LENGTH + i ] = psSS->xq_Q14;
            psDD->Xq_Q14[    *smpl_buf_idx ]         = psSS->xq_Q14;
            psDD->Q_Q10[     *smpl_buf_idx ]         = psSS->Q_Q10;
            psDD->Pred_Q15[  *smpl_buf_idx ]         = silk_LSHIFT32( psSS->LPC_exc_Q14, 1 );
            psDD->Shape_Q14[ *smpl_buf_idx ]         = psSS->sLTP_shp_Q14;
            psDD->Seed                               = silk_ADD32_ovflw( psDD->Seed, silk_RSHIFT_ROUND( psSS->Q_Q10, 10 ) );
            psDD->RandState[ *smpl_buf_idx ]         = psDD->Seed;
            psDD->RD_Q10                             = psSS->RD_Q10;
        }
        delayedGain_Q10[     *smpl_buf_idx ]         = Gain_Q10;
    }
    /* Update LPC states */
    for( k = 0; k < nStatesDelayedDecision; k++ ) {
        psDD = &psDelDec[ k ];
        silk_memcpy( psDD->sLPC_Q14, &psDD->sLPC_Q14[ length ], NSQ_LPC_BUF_LENGTH * sizeof( opus_int32 ) );
    }
}

#endif /* __NSQ_DEL_DEC_MIPSR1_H__ */
