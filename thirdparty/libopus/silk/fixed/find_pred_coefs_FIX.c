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

#include "main_FIX.h"
#include "stack_alloc.h"

void silk_find_pred_coefs_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  encoder state                                                               */
    silk_encoder_control_FIX        *psEncCtrl,                             /* I/O  encoder control                                                             */
    const opus_int16                res_pitch[],                            /* I    Residual from pitch analysis                                                */
    const opus_int16                x[],                                    /* I    Speech signal                                                               */
    opus_int                        condCoding                              /* I    The type of conditional coding to use                                       */
)
{
    opus_int         i;
    opus_int32       invGains_Q16[ MAX_NB_SUBFR ], local_gains[ MAX_NB_SUBFR ];
    /* Set to NLSF_Q15 to zero so we don't copy junk to the state. */
    opus_int16       NLSF_Q15[ MAX_LPC_ORDER ]={0};
    const opus_int16 *x_ptr;
    opus_int16       *x_pre_ptr;
    VARDECL( opus_int16, LPC_in_pre );
    opus_int32       min_gain_Q16, minInvGain_Q30;
    SAVE_STACK;

    /* weighting for weighted least squares */
    min_gain_Q16 = silk_int32_MAX >> 6;
    for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
        min_gain_Q16 = silk_min( min_gain_Q16, psEncCtrl->Gains_Q16[ i ] );
    }
    for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
        /* Divide to Q16 */
        silk_assert( psEncCtrl->Gains_Q16[ i ] > 0 );
        /* Invert and normalize gains, and ensure that maximum invGains_Q16 is within range of a 16 bit int */
        invGains_Q16[ i ] = silk_DIV32_varQ( min_gain_Q16, psEncCtrl->Gains_Q16[ i ], 16 - 2 );

        /* Limit inverse */
        invGains_Q16[ i ] = silk_max( invGains_Q16[ i ], 100 );

        /* Square the inverted gains */
        silk_assert( invGains_Q16[ i ] == silk_SAT16( invGains_Q16[ i ] ) );

        /* Invert the inverted and normalized gains */
        local_gains[ i ] = silk_DIV32( ( (opus_int32)1 << 16 ), invGains_Q16[ i ] );
    }

    ALLOC( LPC_in_pre,
           psEnc->sCmn.nb_subfr * psEnc->sCmn.predictLPCOrder
               + psEnc->sCmn.frame_length, opus_int16 );
    if( psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
        VARDECL( opus_int32, xXLTP_Q17 );
        VARDECL( opus_int32, XXLTP_Q17 );

        /**********/
        /* VOICED */
        /**********/
        celt_assert( psEnc->sCmn.ltp_mem_length - psEnc->sCmn.predictLPCOrder >= psEncCtrl->pitchL[ 0 ] + LTP_ORDER / 2 );

        ALLOC( xXLTP_Q17, psEnc->sCmn.nb_subfr * LTP_ORDER, opus_int32 );
        ALLOC( XXLTP_Q17, psEnc->sCmn.nb_subfr * LTP_ORDER * LTP_ORDER, opus_int32 );

        /* LTP analysis */
        silk_find_LTP_FIX( XXLTP_Q17, xXLTP_Q17, res_pitch,
            psEncCtrl->pitchL, psEnc->sCmn.subfr_length, psEnc->sCmn.nb_subfr, psEnc->sCmn.arch );

        /* Quantize LTP gain parameters */
        silk_quant_LTP_gains( psEncCtrl->LTPCoef_Q14, psEnc->sCmn.indices.LTPIndex, &psEnc->sCmn.indices.PERIndex,
            &psEnc->sCmn.sum_log_gain_Q7, &psEncCtrl->LTPredCodGain_Q7, XXLTP_Q17, xXLTP_Q17, psEnc->sCmn.subfr_length, psEnc->sCmn.nb_subfr, psEnc->sCmn.arch );

        /* Control LTP scaling */
        silk_LTP_scale_ctrl_FIX( psEnc, psEncCtrl, condCoding );

        /* Create LTP residual */
        silk_LTP_analysis_filter_FIX( LPC_in_pre, x - psEnc->sCmn.predictLPCOrder, psEncCtrl->LTPCoef_Q14,
            psEncCtrl->pitchL, invGains_Q16, psEnc->sCmn.subfr_length, psEnc->sCmn.nb_subfr, psEnc->sCmn.predictLPCOrder );

    } else {
        /************/
        /* UNVOICED */
        /************/
        /* Create signal with prepended subframes, scaled by inverse gains */
        x_ptr     = x - psEnc->sCmn.predictLPCOrder;
        x_pre_ptr = LPC_in_pre;
        for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
            silk_scale_copy_vector16( x_pre_ptr, x_ptr, invGains_Q16[ i ],
                psEnc->sCmn.subfr_length + psEnc->sCmn.predictLPCOrder );
            x_pre_ptr += psEnc->sCmn.subfr_length + psEnc->sCmn.predictLPCOrder;
            x_ptr     += psEnc->sCmn.subfr_length;
        }

        silk_memset( psEncCtrl->LTPCoef_Q14, 0, psEnc->sCmn.nb_subfr * LTP_ORDER * sizeof( opus_int16 ) );
        psEncCtrl->LTPredCodGain_Q7 = 0;
        psEnc->sCmn.sum_log_gain_Q7 = 0;
        psEncCtrl->LTP_scale_Q14 = 0;
    }

    /* Limit on total predictive coding gain */
    if( psEnc->sCmn.first_frame_after_reset ) {
        minInvGain_Q30 = SILK_FIX_CONST( 1.0f / MAX_PREDICTION_POWER_GAIN_AFTER_RESET, 30 );
    } else {
        minInvGain_Q30 = silk_log2lin( silk_SMLAWB( 16 << 7, (opus_int32)psEncCtrl->LTPredCodGain_Q7, SILK_FIX_CONST( 1.0 / 3, 16 ) ) );      /* Q16 */
        minInvGain_Q30 = silk_DIV32_varQ( minInvGain_Q30,
            silk_SMULWW( SILK_FIX_CONST( MAX_PREDICTION_POWER_GAIN, 0 ),
                silk_SMLAWB( SILK_FIX_CONST( 0.25, 18 ), SILK_FIX_CONST( 0.75, 18 ), psEncCtrl->coding_quality_Q14 ) ), 14 );
    }

    /* LPC_in_pre contains the LTP-filtered input for voiced, and the unfiltered input for unvoiced */
    silk_find_LPC_FIX( &psEnc->sCmn, NLSF_Q15, LPC_in_pre, minInvGain_Q30 );

    /* Quantize LSFs */
    silk_process_NLSFs( &psEnc->sCmn, psEncCtrl->PredCoef_Q12, NLSF_Q15, psEnc->sCmn.prev_NLSFq_Q15 );

    /* Calculate residual energy using quantized LPC coefficients */
    silk_residual_energy_FIX( psEncCtrl->ResNrg, psEncCtrl->ResNrgQ, LPC_in_pre, psEncCtrl->PredCoef_Q12, local_gains,
        psEnc->sCmn.subfr_length, psEnc->sCmn.nb_subfr, psEnc->sCmn.predictLPCOrder, psEnc->sCmn.arch );

    /* Copy to prediction struct for use in next frame for interpolation */
    silk_memcpy( psEnc->sCmn.prev_NLSFq_Q15, NLSF_Q15, sizeof( psEnc->sCmn.prev_NLSFq_Q15 ) );
    RESTORE_STACK;
}
