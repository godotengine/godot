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
#include "stack_alloc.h"

/**********************************************************/
/* Core decoder. Performs inverse NSQ operation LTP + LPC */
/**********************************************************/
void silk_decode_core(
    silk_decoder_state          *psDec,                         /* I/O  Decoder state                               */
    silk_decoder_control        *psDecCtrl,                     /* I    Decoder control                             */
    opus_int16                  xq[],                           /* O    Decoded speech                              */
    const opus_int16            pulses[ MAX_FRAME_LENGTH ],     /* I    Pulse signal                                */
    int                         arch                            /* I    Run-time architecture                       */
)
{
    opus_int   i, k, lag = 0, start_idx, sLTP_buf_idx, NLSF_interpolation_flag, signalType;
    opus_int16 *A_Q12, *B_Q14, *pxq, A_Q12_tmp[ MAX_LPC_ORDER ];
    VARDECL( opus_int16, sLTP );
    VARDECL( opus_int32, sLTP_Q15 );
    opus_int32 LTP_pred_Q13, LPC_pred_Q10, Gain_Q10, inv_gain_Q31, gain_adj_Q16, rand_seed, offset_Q10;
    opus_int32 *pred_lag_ptr, *pexc_Q14, *pres_Q14;
    VARDECL( opus_int32, res_Q14 );
    VARDECL( opus_int32, sLPC_Q14 );
    SAVE_STACK;

    silk_assert( psDec->prev_gain_Q16 != 0 );

    ALLOC( sLTP, psDec->ltp_mem_length, opus_int16 );
    ALLOC( sLTP_Q15, psDec->ltp_mem_length + psDec->frame_length, opus_int32 );
    ALLOC( res_Q14, psDec->subfr_length, opus_int32 );
    ALLOC( sLPC_Q14, psDec->subfr_length + MAX_LPC_ORDER, opus_int32 );

    offset_Q10 = silk_Quantization_Offsets_Q10[ psDec->indices.signalType >> 1 ][ psDec->indices.quantOffsetType ];

    if( psDec->indices.NLSFInterpCoef_Q2 < 1 << 2 ) {
        NLSF_interpolation_flag = 1;
    } else {
        NLSF_interpolation_flag = 0;
    }

    /* Decode excitation */
    rand_seed = psDec->indices.Seed;
    for( i = 0; i < psDec->frame_length; i++ ) {
        rand_seed = silk_RAND( rand_seed );
        psDec->exc_Q14[ i ] = silk_LSHIFT( (opus_int32)pulses[ i ], 14 );
        if( psDec->exc_Q14[ i ] > 0 ) {
            psDec->exc_Q14[ i ] -= QUANT_LEVEL_ADJUST_Q10 << 4;
        } else
        if( psDec->exc_Q14[ i ] < 0 ) {
            psDec->exc_Q14[ i ] += QUANT_LEVEL_ADJUST_Q10 << 4;
        }
        psDec->exc_Q14[ i ] += offset_Q10 << 4;
        if( rand_seed < 0 ) {
           psDec->exc_Q14[ i ] = -psDec->exc_Q14[ i ];
        }

        rand_seed = silk_ADD32_ovflw( rand_seed, pulses[ i ] );
    }

    /* Copy LPC state */
    silk_memcpy( sLPC_Q14, psDec->sLPC_Q14_buf, MAX_LPC_ORDER * sizeof( opus_int32 ) );

    pexc_Q14 = psDec->exc_Q14;
    pxq      = xq;
    sLTP_buf_idx = psDec->ltp_mem_length;
    /* Loop over subframes */
    for( k = 0; k < psDec->nb_subfr; k++ ) {
        pres_Q14 = res_Q14;
        A_Q12 = psDecCtrl->PredCoef_Q12[ k >> 1 ];

        /* Preload LPC coeficients to array on stack. Gives small performance gain */
        silk_memcpy( A_Q12_tmp, A_Q12, psDec->LPC_order * sizeof( opus_int16 ) );
        B_Q14        = &psDecCtrl->LTPCoef_Q14[ k * LTP_ORDER ];
        signalType   = psDec->indices.signalType;

        Gain_Q10     = silk_RSHIFT( psDecCtrl->Gains_Q16[ k ], 6 );
        inv_gain_Q31 = silk_INVERSE32_varQ( psDecCtrl->Gains_Q16[ k ], 47 );

        /* Calculate gain adjustment factor */
        if( psDecCtrl->Gains_Q16[ k ] != psDec->prev_gain_Q16 ) {
            gain_adj_Q16 =  silk_DIV32_varQ( psDec->prev_gain_Q16, psDecCtrl->Gains_Q16[ k ], 16 );

            /* Scale short term state */
            for( i = 0; i < MAX_LPC_ORDER; i++ ) {
                sLPC_Q14[ i ] = silk_SMULWW( gain_adj_Q16, sLPC_Q14[ i ] );
            }
        } else {
            gain_adj_Q16 = (opus_int32)1 << 16;
        }

        /* Save inv_gain */
        silk_assert( inv_gain_Q31 != 0 );
        psDec->prev_gain_Q16 = psDecCtrl->Gains_Q16[ k ];

        /* Avoid abrupt transition from voiced PLC to unvoiced normal decoding */
        if( psDec->lossCnt && psDec->prevSignalType == TYPE_VOICED &&
            psDec->indices.signalType != TYPE_VOICED && k < MAX_NB_SUBFR/2 ) {

            silk_memset( B_Q14, 0, LTP_ORDER * sizeof( opus_int16 ) );
            B_Q14[ LTP_ORDER/2 ] = SILK_FIX_CONST( 0.25, 14 );

            signalType = TYPE_VOICED;
            psDecCtrl->pitchL[ k ] = psDec->lagPrev;
        }

        if( signalType == TYPE_VOICED ) {
            /* Voiced */
            lag = psDecCtrl->pitchL[ k ];

            /* Re-whitening */
            if( k == 0 || ( k == 2 && NLSF_interpolation_flag ) ) {
                /* Rewhiten with new A coefs */
                start_idx = psDec->ltp_mem_length - lag - psDec->LPC_order - LTP_ORDER / 2;
                celt_assert( start_idx > 0 );

                if( k == 2 ) {
                    silk_memcpy( &psDec->outBuf[ psDec->ltp_mem_length ], xq, 2 * psDec->subfr_length * sizeof( opus_int16 ) );
                }

                silk_LPC_analysis_filter( &sLTP[ start_idx ], &psDec->outBuf[ start_idx + k * psDec->subfr_length ],
                    A_Q12, psDec->ltp_mem_length - start_idx, psDec->LPC_order, arch );

                /* After rewhitening the LTP state is unscaled */
                if( k == 0 ) {
                    /* Do LTP downscaling to reduce inter-packet dependency */
                    inv_gain_Q31 = silk_LSHIFT( silk_SMULWB( inv_gain_Q31, psDecCtrl->LTP_scale_Q14 ), 2 );
                }
                for( i = 0; i < lag + LTP_ORDER/2; i++ ) {
                    sLTP_Q15[ sLTP_buf_idx - i - 1 ] = silk_SMULWB( inv_gain_Q31, sLTP[ psDec->ltp_mem_length - i - 1 ] );
                }
            } else {
                /* Update LTP state when Gain changes */
                if( gain_adj_Q16 != (opus_int32)1 << 16 ) {
                    for( i = 0; i < lag + LTP_ORDER/2; i++ ) {
                        sLTP_Q15[ sLTP_buf_idx - i - 1 ] = silk_SMULWW( gain_adj_Q16, sLTP_Q15[ sLTP_buf_idx - i - 1 ] );
                    }
                }
            }
        }

        /* Long-term prediction */
        if( signalType == TYPE_VOICED ) {
            /* Set up pointer */
            pred_lag_ptr = &sLTP_Q15[ sLTP_buf_idx - lag + LTP_ORDER / 2 ];
            for( i = 0; i < psDec->subfr_length; i++ ) {
                /* Unrolled loop */
                /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
                LTP_pred_Q13 = 2;
                LTP_pred_Q13 = silk_SMLAWB( LTP_pred_Q13, pred_lag_ptr[  0 ], B_Q14[ 0 ] );
                LTP_pred_Q13 = silk_SMLAWB( LTP_pred_Q13, pred_lag_ptr[ -1 ], B_Q14[ 1 ] );
                LTP_pred_Q13 = silk_SMLAWB( LTP_pred_Q13, pred_lag_ptr[ -2 ], B_Q14[ 2 ] );
                LTP_pred_Q13 = silk_SMLAWB( LTP_pred_Q13, pred_lag_ptr[ -3 ], B_Q14[ 3 ] );
                LTP_pred_Q13 = silk_SMLAWB( LTP_pred_Q13, pred_lag_ptr[ -4 ], B_Q14[ 4 ] );
                pred_lag_ptr++;

                /* Generate LPC excitation */
                pres_Q14[ i ] = silk_ADD_LSHIFT32( pexc_Q14[ i ], LTP_pred_Q13, 1 );

                /* Update states */
                sLTP_Q15[ sLTP_buf_idx ] = silk_LSHIFT( pres_Q14[ i ], 1 );
                sLTP_buf_idx++;
            }
        } else {
            pres_Q14 = pexc_Q14;
        }

        for( i = 0; i < psDec->subfr_length; i++ ) {
            /* Short-term prediction */
            celt_assert( psDec->LPC_order == 10 || psDec->LPC_order == 16 );
            /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
            LPC_pred_Q10 = silk_RSHIFT( psDec->LPC_order, 1 );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  1 ], A_Q12_tmp[ 0 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  2 ], A_Q12_tmp[ 1 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  3 ], A_Q12_tmp[ 2 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  4 ], A_Q12_tmp[ 3 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  5 ], A_Q12_tmp[ 4 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  6 ], A_Q12_tmp[ 5 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  7 ], A_Q12_tmp[ 6 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  8 ], A_Q12_tmp[ 7 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  9 ], A_Q12_tmp[ 8 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i - 10 ], A_Q12_tmp[ 9 ] );
            if( psDec->LPC_order == 16 ) {
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i - 11 ], A_Q12_tmp[ 10 ] );
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i - 12 ], A_Q12_tmp[ 11 ] );
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i - 13 ], A_Q12_tmp[ 12 ] );
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i - 14 ], A_Q12_tmp[ 13 ] );
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i - 15 ], A_Q12_tmp[ 14 ] );
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i - 16 ], A_Q12_tmp[ 15 ] );
            }

            /* Add prediction to LPC excitation */
            sLPC_Q14[ MAX_LPC_ORDER + i ] = silk_ADD_SAT32( pres_Q14[ i ], silk_LSHIFT_SAT32( LPC_pred_Q10, 4 ) );

            /* Scale with gain */
            pxq[ i ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( silk_SMULWW( sLPC_Q14[ MAX_LPC_ORDER + i ], Gain_Q10 ), 8 ) );
        }

        /* Update LPC filter state */
        silk_memcpy( sLPC_Q14, &sLPC_Q14[ psDec->subfr_length ], MAX_LPC_ORDER * sizeof( opus_int32 ) );
        pexc_Q14 += psDec->subfr_length;
        pxq      += psDec->subfr_length;
    }

    /* Save LPC state */
    silk_memcpy( psDec->sLPC_Q14_buf, sLPC_Q14, MAX_LPC_ORDER * sizeof( opus_int32 ) );
    RESTORE_STACK;
}
