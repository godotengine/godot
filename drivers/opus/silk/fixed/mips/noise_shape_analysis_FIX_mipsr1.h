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


/**************************************************************/
/* Compute noise shaping coefficients and initial gain values */
/**************************************************************/
#define OVERRIDE_silk_noise_shape_analysis_FIX

void silk_noise_shape_analysis_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  Encoder state FIX                                                           */
    silk_encoder_control_FIX        *psEncCtrl,                             /* I/O  Encoder control FIX                                                         */
    const opus_int16                *pitch_res,                             /* I    LPC residual from pitch analysis                                            */
    const opus_int16                *x,                                     /* I    Input signal [ frame_length + la_shape ]                                    */
    int                              arch                                   /* I    Run-time architecture                                                       */
)
{
    silk_shape_state_FIX *psShapeSt = &psEnc->sShape;
    opus_int     k, i, nSamples, Qnrg, b_Q14, warping_Q16, scale = 0;
    opus_int32   SNR_adj_dB_Q7, HarmBoost_Q16, HarmShapeGain_Q16, Tilt_Q16, tmp32;
    opus_int32   nrg, pre_nrg_Q30, log_energy_Q7, log_energy_prev_Q7, energy_variation_Q7;
    opus_int32   delta_Q16, BWExp1_Q16, BWExp2_Q16, gain_mult_Q16, gain_add_Q16, strength_Q16, b_Q8;
    opus_int32   auto_corr[     MAX_SHAPE_LPC_ORDER + 1 ];
    opus_int32   refl_coef_Q16[ MAX_SHAPE_LPC_ORDER ];
    opus_int32   AR1_Q24[       MAX_SHAPE_LPC_ORDER ];
    opus_int32   AR2_Q24[       MAX_SHAPE_LPC_ORDER ];
    VARDECL( opus_int16, x_windowed );
    const opus_int16 *x_ptr, *pitch_res_ptr;
    SAVE_STACK;

    /* Point to start of first LPC analysis block */
    x_ptr = x - psEnc->sCmn.la_shape;

    /****************/
    /* GAIN CONTROL */
    /****************/
    SNR_adj_dB_Q7 = psEnc->sCmn.SNR_dB_Q7;

    /* Input quality is the average of the quality in the lowest two VAD bands */
    psEncCtrl->input_quality_Q14 = ( opus_int )silk_RSHIFT( (opus_int32)psEnc->sCmn.input_quality_bands_Q15[ 0 ]
        + psEnc->sCmn.input_quality_bands_Q15[ 1 ], 2 );

    /* Coding quality level, between 0.0_Q0 and 1.0_Q0, but in Q14 */
    psEncCtrl->coding_quality_Q14 = silk_RSHIFT( silk_sigm_Q15( silk_RSHIFT_ROUND( SNR_adj_dB_Q7 -
        SILK_FIX_CONST( 20.0, 7 ), 4 ) ), 1 );

    /* Reduce coding SNR during low speech activity */
    if( psEnc->sCmn.useCBR == 0 ) {
        b_Q8 = SILK_FIX_CONST( 1.0, 8 ) - psEnc->sCmn.speech_activity_Q8;
        b_Q8 = silk_SMULWB( silk_LSHIFT( b_Q8, 8 ), b_Q8 );
        SNR_adj_dB_Q7 = silk_SMLAWB( SNR_adj_dB_Q7,
            silk_SMULBB( SILK_FIX_CONST( -BG_SNR_DECR_dB, 7 ) >> ( 4 + 1 ), b_Q8 ),                                       /* Q11*/
            silk_SMULWB( SILK_FIX_CONST( 1.0, 14 ) + psEncCtrl->input_quality_Q14, psEncCtrl->coding_quality_Q14 ) );     /* Q12*/
    }

    if( psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
        /* Reduce gains for periodic signals */
        SNR_adj_dB_Q7 = silk_SMLAWB( SNR_adj_dB_Q7, SILK_FIX_CONST( HARM_SNR_INCR_dB, 8 ), psEnc->LTPCorr_Q15 );
    } else {
        /* For unvoiced signals and low-quality input, adjust the quality slower than SNR_dB setting */
        SNR_adj_dB_Q7 = silk_SMLAWB( SNR_adj_dB_Q7,
            silk_SMLAWB( SILK_FIX_CONST( 6.0, 9 ), -SILK_FIX_CONST( 0.4, 18 ), psEnc->sCmn.SNR_dB_Q7 ),
            SILK_FIX_CONST( 1.0, 14 ) - psEncCtrl->input_quality_Q14 );
    }

    /*************************/
    /* SPARSENESS PROCESSING */
    /*************************/
    /* Set quantizer offset */
    if( psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
        /* Initially set to 0; may be overruled in process_gains(..) */
        psEnc->sCmn.indices.quantOffsetType = 0;
        psEncCtrl->sparseness_Q8 = 0;
    } else {
        /* Sparseness measure, based on relative fluctuations of energy per 2 milliseconds */
        nSamples = silk_LSHIFT( psEnc->sCmn.fs_kHz, 1 );
        energy_variation_Q7 = 0;
        log_energy_prev_Q7  = 0;
        pitch_res_ptr = pitch_res;
        for( k = 0; k < silk_SMULBB( SUB_FRAME_LENGTH_MS, psEnc->sCmn.nb_subfr ) / 2; k++ ) {
            silk_sum_sqr_shift( &nrg, &scale, pitch_res_ptr, nSamples );
            nrg += silk_RSHIFT( nSamples, scale );           /* Q(-scale)*/

            log_energy_Q7 = silk_lin2log( nrg );
            if( k > 0 ) {
                energy_variation_Q7 += silk_abs( log_energy_Q7 - log_energy_prev_Q7 );
            }
            log_energy_prev_Q7 = log_energy_Q7;
            pitch_res_ptr += nSamples;
        }

        psEncCtrl->sparseness_Q8 = silk_RSHIFT( silk_sigm_Q15( silk_SMULWB( energy_variation_Q7 -
            SILK_FIX_CONST( 5.0, 7 ), SILK_FIX_CONST( 0.1, 16 ) ) ), 7 );

        /* Set quantization offset depending on sparseness measure */
        if( psEncCtrl->sparseness_Q8 > SILK_FIX_CONST( SPARSENESS_THRESHOLD_QNT_OFFSET, 8 ) ) {
            psEnc->sCmn.indices.quantOffsetType = 0;
        } else {
            psEnc->sCmn.indices.quantOffsetType = 1;
        }

        /* Increase coding SNR for sparse signals */
        SNR_adj_dB_Q7 = silk_SMLAWB( SNR_adj_dB_Q7, SILK_FIX_CONST( SPARSE_SNR_INCR_dB, 15 ), psEncCtrl->sparseness_Q8 - SILK_FIX_CONST( 0.5, 8 ) );
    }

    /*******************************/
    /* Control bandwidth expansion */
    /*******************************/
    /* More BWE for signals with high prediction gain */
    strength_Q16 = silk_SMULWB( psEncCtrl->predGain_Q16, SILK_FIX_CONST( FIND_PITCH_WHITE_NOISE_FRACTION, 16 ) );
    BWExp1_Q16 = BWExp2_Q16 = silk_DIV32_varQ( SILK_FIX_CONST( BANDWIDTH_EXPANSION, 16 ),
        silk_SMLAWW( SILK_FIX_CONST( 1.0, 16 ), strength_Q16, strength_Q16 ), 16 );
    delta_Q16  = silk_SMULWB( SILK_FIX_CONST( 1.0, 16 ) - silk_SMULBB( 3, psEncCtrl->coding_quality_Q14 ),
        SILK_FIX_CONST( LOW_RATE_BANDWIDTH_EXPANSION_DELTA, 16 ) );
    BWExp1_Q16 = silk_SUB32( BWExp1_Q16, delta_Q16 );
    BWExp2_Q16 = silk_ADD32( BWExp2_Q16, delta_Q16 );
    /* BWExp1 will be applied after BWExp2, so make it relative */
    BWExp1_Q16 = silk_DIV32_16( silk_LSHIFT( BWExp1_Q16, 14 ), silk_RSHIFT( BWExp2_Q16, 2 ) );

    if( psEnc->sCmn.warping_Q16 > 0 ) {
        /* Slightly more warping in analysis will move quantization noise up in frequency, where it's better masked */
        warping_Q16 = silk_SMLAWB( psEnc->sCmn.warping_Q16, (opus_int32)psEncCtrl->coding_quality_Q14, SILK_FIX_CONST( 0.01, 18 ) );
    } else {
        warping_Q16 = 0;
    }

    /********************************************/
    /* Compute noise shaping AR coefs and gains */
    /********************************************/
    ALLOC( x_windowed, psEnc->sCmn.shapeWinLength, opus_int16 );
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        /* Apply window: sine slope followed by flat part followed by cosine slope */
        opus_int shift, slope_part, flat_part;
        flat_part = psEnc->sCmn.fs_kHz * 3;
        slope_part = silk_RSHIFT( psEnc->sCmn.shapeWinLength - flat_part, 1 );

        silk_apply_sine_window( x_windowed, x_ptr, 1, slope_part );
        shift = slope_part;
        silk_memcpy( x_windowed + shift, x_ptr + shift, flat_part * sizeof(opus_int16) );
        shift += flat_part;
        silk_apply_sine_window( x_windowed + shift, x_ptr + shift, 2, slope_part );

        /* Update pointer: next LPC analysis block */
        x_ptr += psEnc->sCmn.subfr_length;

        if( psEnc->sCmn.warping_Q16 > 0 ) {
            /* Calculate warped auto correlation */
            silk_warped_autocorrelation_FIX( auto_corr, &scale, x_windowed, warping_Q16, psEnc->sCmn.shapeWinLength, psEnc->sCmn.shapingLPCOrder );
        } else {
            /* Calculate regular auto correlation */
            silk_autocorr( auto_corr, &scale, x_windowed, psEnc->sCmn.shapeWinLength, psEnc->sCmn.shapingLPCOrder + 1, arch );
        }

        /* Add white noise, as a fraction of energy */
        auto_corr[0] = silk_ADD32( auto_corr[0], silk_max_32( silk_SMULWB( silk_RSHIFT( auto_corr[ 0 ], 4 ),
            SILK_FIX_CONST( SHAPE_WHITE_NOISE_FRACTION, 20 ) ), 1 ) );

        /* Calculate the reflection coefficients using schur */
        nrg = silk_schur64( refl_coef_Q16, auto_corr, psEnc->sCmn.shapingLPCOrder );
        silk_assert( nrg >= 0 );

        /* Convert reflection coefficients to prediction coefficients */
        silk_k2a_Q16( AR2_Q24, refl_coef_Q16, psEnc->sCmn.shapingLPCOrder );

        Qnrg = -scale;          /* range: -12...30*/
        silk_assert( Qnrg >= -12 );
        silk_assert( Qnrg <=  30 );

        /* Make sure that Qnrg is an even number */
        if( Qnrg & 1 ) {
            Qnrg -= 1;
            nrg >>= 1;
        }

        tmp32 = silk_SQRT_APPROX( nrg );
        Qnrg >>= 1;             /* range: -6...15*/

        psEncCtrl->Gains_Q16[ k ] = (silk_LSHIFT32( silk_LIMIT( (tmp32), silk_RSHIFT32( silk_int32_MIN, (16 - Qnrg) ), \
                            silk_RSHIFT32( silk_int32_MAX, (16 - Qnrg) ) ), (16 - Qnrg) ));

        if( psEnc->sCmn.warping_Q16 > 0 ) {
            /* Adjust gain for warping */
            gain_mult_Q16 = warped_gain( AR2_Q24, warping_Q16, psEnc->sCmn.shapingLPCOrder );
            silk_assert( psEncCtrl->Gains_Q16[ k ] >= 0 );
            if ( silk_SMULWW( silk_RSHIFT_ROUND( psEncCtrl->Gains_Q16[ k ], 1 ), gain_mult_Q16 ) >= ( silk_int32_MAX >> 1 ) ) {
               psEncCtrl->Gains_Q16[ k ] = silk_int32_MAX;
            } else {
               psEncCtrl->Gains_Q16[ k ] = silk_SMULWW( psEncCtrl->Gains_Q16[ k ], gain_mult_Q16 );
            }
        }

        /* Bandwidth expansion for synthesis filter shaping */
        silk_bwexpander_32( AR2_Q24, psEnc->sCmn.shapingLPCOrder, BWExp2_Q16 );

        /* Compute noise shaping filter coefficients */
        silk_memcpy( AR1_Q24, AR2_Q24, psEnc->sCmn.shapingLPCOrder * sizeof( opus_int32 ) );

        /* Bandwidth expansion for analysis filter shaping */
        silk_assert( BWExp1_Q16 <= SILK_FIX_CONST( 1.0, 16 ) );
        silk_bwexpander_32( AR1_Q24, psEnc->sCmn.shapingLPCOrder, BWExp1_Q16 );

        /* Ratio of prediction gains, in energy domain */
        pre_nrg_Q30 = silk_LPC_inverse_pred_gain_Q24( AR2_Q24, psEnc->sCmn.shapingLPCOrder );
        nrg         = silk_LPC_inverse_pred_gain_Q24( AR1_Q24, psEnc->sCmn.shapingLPCOrder );

        /*psEncCtrl->GainsPre[ k ] = 1.0f - 0.7f * ( 1.0f - pre_nrg / nrg ) = 0.3f + 0.7f * pre_nrg / nrg;*/
        pre_nrg_Q30 = silk_LSHIFT32( silk_SMULWB( pre_nrg_Q30, SILK_FIX_CONST( 0.7, 15 ) ), 1 );
        psEncCtrl->GainsPre_Q14[ k ] = ( opus_int ) SILK_FIX_CONST( 0.3, 14 ) + silk_DIV32_varQ( pre_nrg_Q30, nrg, 14 );

        /* Convert to monic warped prediction coefficients and limit absolute values */
        limit_warped_coefs( AR2_Q24, AR1_Q24, warping_Q16, SILK_FIX_CONST( 3.999, 24 ), psEnc->sCmn.shapingLPCOrder );

        /* Convert from Q24 to Q13 and store in int16 */
        for( i = 0; i < psEnc->sCmn.shapingLPCOrder; i++ ) {
            psEncCtrl->AR1_Q13[ k * MAX_SHAPE_LPC_ORDER + i ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( AR1_Q24[ i ], 11 ) );
            psEncCtrl->AR2_Q13[ k * MAX_SHAPE_LPC_ORDER + i ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( AR2_Q24[ i ], 11 ) );
        }
    }

    /*****************/
    /* Gain tweaking */
    /*****************/
    /* Increase gains during low speech activity and put lower limit on gains */
    gain_mult_Q16 = silk_log2lin( -silk_SMLAWB( -SILK_FIX_CONST( 16.0, 7 ), SNR_adj_dB_Q7, SILK_FIX_CONST( 0.16, 16 ) ) );
    gain_add_Q16  = silk_log2lin(  silk_SMLAWB(  SILK_FIX_CONST( 16.0, 7 ), SILK_FIX_CONST( MIN_QGAIN_DB, 7 ), SILK_FIX_CONST( 0.16, 16 ) ) );
    silk_assert( gain_mult_Q16 > 0 );
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        psEncCtrl->Gains_Q16[ k ] = silk_SMULWW( psEncCtrl->Gains_Q16[ k ], gain_mult_Q16 );
        silk_assert( psEncCtrl->Gains_Q16[ k ] >= 0 );
        psEncCtrl->Gains_Q16[ k ] = silk_ADD_POS_SAT32( psEncCtrl->Gains_Q16[ k ], gain_add_Q16 );
    }

    gain_mult_Q16 = SILK_FIX_CONST( 1.0, 16 ) + silk_RSHIFT_ROUND( silk_MLA( SILK_FIX_CONST( INPUT_TILT, 26 ),
        psEncCtrl->coding_quality_Q14, SILK_FIX_CONST( HIGH_RATE_INPUT_TILT, 12 ) ), 10 );
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        psEncCtrl->GainsPre_Q14[ k ] = silk_SMULWB( gain_mult_Q16, psEncCtrl->GainsPre_Q14[ k ] );
    }

    /************************************************/
    /* Control low-frequency shaping and noise tilt */
    /************************************************/
    /* Less low frequency shaping for noisy inputs */
    strength_Q16 = silk_MUL( SILK_FIX_CONST( LOW_FREQ_SHAPING, 4 ), silk_SMLAWB( SILK_FIX_CONST( 1.0, 12 ),
        SILK_FIX_CONST( LOW_QUALITY_LOW_FREQ_SHAPING_DECR, 13 ), psEnc->sCmn.input_quality_bands_Q15[ 0 ] - SILK_FIX_CONST( 1.0, 15 ) ) );
    strength_Q16 = silk_RSHIFT( silk_MUL( strength_Q16, psEnc->sCmn.speech_activity_Q8 ), 8 );
    if( psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
        /* Reduce low frequencies quantization noise for periodic signals, depending on pitch lag */
        /*f = 400; freqz([1, -0.98 + 2e-4 * f], [1, -0.97 + 7e-4 * f], 2^12, Fs); axis([0, 1000, -10, 1])*/
        opus_int fs_kHz_inv = silk_DIV32_16( SILK_FIX_CONST( 0.2, 14 ), psEnc->sCmn.fs_kHz );
        for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
            b_Q14 = fs_kHz_inv + silk_DIV32_16( SILK_FIX_CONST( 3.0, 14 ), psEncCtrl->pitchL[ k ] );
            /* Pack two coefficients in one int32 */
            psEncCtrl->LF_shp_Q14[ k ]  = silk_LSHIFT( SILK_FIX_CONST( 1.0, 14 ) - b_Q14 - silk_SMULWB( strength_Q16, b_Q14 ), 16 );
            psEncCtrl->LF_shp_Q14[ k ] |= (opus_uint16)( b_Q14 - SILK_FIX_CONST( 1.0, 14 ) );
        }
        silk_assert( SILK_FIX_CONST( HARM_HP_NOISE_COEF, 24 ) < SILK_FIX_CONST( 0.5, 24 ) ); /* Guarantees that second argument to SMULWB() is within range of an opus_int16*/
        Tilt_Q16 = - SILK_FIX_CONST( HP_NOISE_COEF, 16 ) -
            silk_SMULWB( SILK_FIX_CONST( 1.0, 16 ) - SILK_FIX_CONST( HP_NOISE_COEF, 16 ),
                silk_SMULWB( SILK_FIX_CONST( HARM_HP_NOISE_COEF, 24 ), psEnc->sCmn.speech_activity_Q8 ) );
    } else {
        b_Q14 = silk_DIV32_16( 21299, psEnc->sCmn.fs_kHz ); /* 1.3_Q0 = 21299_Q14*/
        /* Pack two coefficients in one int32 */
        psEncCtrl->LF_shp_Q14[ 0 ]  = silk_LSHIFT( SILK_FIX_CONST( 1.0, 14 ) - b_Q14 -
            silk_SMULWB( strength_Q16, silk_SMULWB( SILK_FIX_CONST( 0.6, 16 ), b_Q14 ) ), 16 );
        psEncCtrl->LF_shp_Q14[ 0 ] |= (opus_uint16)( b_Q14 - SILK_FIX_CONST( 1.0, 14 ) );
        for( k = 1; k < psEnc->sCmn.nb_subfr; k++ ) {
            psEncCtrl->LF_shp_Q14[ k ] = psEncCtrl->LF_shp_Q14[ 0 ];
        }
        Tilt_Q16 = -SILK_FIX_CONST( HP_NOISE_COEF, 16 );
    }

    /****************************/
    /* HARMONIC SHAPING CONTROL */
    /****************************/
    /* Control boosting of harmonic frequencies */
    HarmBoost_Q16 = silk_SMULWB( silk_SMULWB( SILK_FIX_CONST( 1.0, 17 ) - silk_LSHIFT( psEncCtrl->coding_quality_Q14, 3 ),
        psEnc->LTPCorr_Q15 ), SILK_FIX_CONST( LOW_RATE_HARMONIC_BOOST, 16 ) );

    /* More harmonic boost for noisy input signals */
    HarmBoost_Q16 = silk_SMLAWB( HarmBoost_Q16,
        SILK_FIX_CONST( 1.0, 16 ) - silk_LSHIFT( psEncCtrl->input_quality_Q14, 2 ), SILK_FIX_CONST( LOW_INPUT_QUALITY_HARMONIC_BOOST, 16 ) );

    if( USE_HARM_SHAPING && psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
        /* More harmonic noise shaping for high bitrates or noisy input */
        HarmShapeGain_Q16 = silk_SMLAWB( SILK_FIX_CONST( HARMONIC_SHAPING, 16 ),
                SILK_FIX_CONST( 1.0, 16 ) - silk_SMULWB( SILK_FIX_CONST( 1.0, 18 ) - silk_LSHIFT( psEncCtrl->coding_quality_Q14, 4 ),
                psEncCtrl->input_quality_Q14 ), SILK_FIX_CONST( HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING, 16 ) );

        /* Less harmonic noise shaping for less periodic signals */
        HarmShapeGain_Q16 = silk_SMULWB( silk_LSHIFT( HarmShapeGain_Q16, 1 ),
            silk_SQRT_APPROX( silk_LSHIFT( psEnc->LTPCorr_Q15, 15 ) ) );
    } else {
        HarmShapeGain_Q16 = 0;
    }

    /*************************/
    /* Smooth over subframes */
    /*************************/
    for( k = 0; k < MAX_NB_SUBFR; k++ ) {
        psShapeSt->HarmBoost_smth_Q16 =
            silk_SMLAWB( psShapeSt->HarmBoost_smth_Q16,     HarmBoost_Q16     - psShapeSt->HarmBoost_smth_Q16,     SILK_FIX_CONST( SUBFR_SMTH_COEF, 16 ) );
        psShapeSt->HarmShapeGain_smth_Q16 =
            silk_SMLAWB( psShapeSt->HarmShapeGain_smth_Q16, HarmShapeGain_Q16 - psShapeSt->HarmShapeGain_smth_Q16, SILK_FIX_CONST( SUBFR_SMTH_COEF, 16 ) );
        psShapeSt->Tilt_smth_Q16 =
            silk_SMLAWB( psShapeSt->Tilt_smth_Q16,          Tilt_Q16          - psShapeSt->Tilt_smth_Q16,          SILK_FIX_CONST( SUBFR_SMTH_COEF, 16 ) );

        psEncCtrl->HarmBoost_Q14[ k ]     = ( opus_int )silk_RSHIFT_ROUND( psShapeSt->HarmBoost_smth_Q16,     2 );
        psEncCtrl->HarmShapeGain_Q14[ k ] = ( opus_int )silk_RSHIFT_ROUND( psShapeSt->HarmShapeGain_smth_Q16, 2 );
        psEncCtrl->Tilt_Q14[ k ]          = ( opus_int )silk_RSHIFT_ROUND( psShapeSt->Tilt_smth_Q16,          2 );
    }
    RESTORE_STACK;
}
