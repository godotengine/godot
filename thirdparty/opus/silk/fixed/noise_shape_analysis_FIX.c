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
#include "tuning_parameters.h"

/* Compute gain to make warped filter coefficients have a zero mean log frequency response on a   */
/* non-warped frequency scale. (So that it can be implemented with a minimum-phase monic filter.) */
/* Note: A monic filter is one with the first coefficient equal to 1.0. In Silk we omit the first */
/* coefficient in an array of coefficients, for monic filters.                                    */
static OPUS_INLINE opus_int32 warped_gain( /* gain in Q16*/
    const opus_int32     *coefs_Q24,
    opus_int             lambda_Q16,
    opus_int             order
) {
    opus_int   i;
    opus_int32 gain_Q24;

    lambda_Q16 = -lambda_Q16;
    gain_Q24 = coefs_Q24[ order - 1 ];
    for( i = order - 2; i >= 0; i-- ) {
        gain_Q24 = silk_SMLAWB( coefs_Q24[ i ], gain_Q24, lambda_Q16 );
    }
    gain_Q24  = silk_SMLAWB( SILK_FIX_CONST( 1.0, 24 ), gain_Q24, -lambda_Q16 );
    return silk_INVERSE32_varQ( gain_Q24, 40 );
}

/* Convert warped filter coefficients to monic pseudo-warped coefficients and limit maximum     */
/* amplitude of monic warped coefficients by using bandwidth expansion on the true coefficients */
static OPUS_INLINE void limit_warped_coefs(
    opus_int32           *coefs_Q24,
    opus_int             lambda_Q16,
    opus_int32           limit_Q24,
    opus_int             order
) {
    opus_int   i, iter, ind = 0;
    opus_int32 tmp, maxabs_Q24, chirp_Q16, gain_Q16;
    opus_int32 nom_Q16, den_Q24;
    opus_int32 limit_Q20, maxabs_Q20;

    /* Convert to monic coefficients */
    lambda_Q16 = -lambda_Q16;
    for( i = order - 1; i > 0; i-- ) {
        coefs_Q24[ i - 1 ] = silk_SMLAWB( coefs_Q24[ i - 1 ], coefs_Q24[ i ], lambda_Q16 );
    }
    lambda_Q16 = -lambda_Q16;
    nom_Q16  = silk_SMLAWB( SILK_FIX_CONST( 1.0, 16 ), -(opus_int32)lambda_Q16, lambda_Q16 );
    den_Q24  = silk_SMLAWB( SILK_FIX_CONST( 1.0, 24 ), coefs_Q24[ 0 ], lambda_Q16 );
    gain_Q16 = silk_DIV32_varQ( nom_Q16, den_Q24, 24 );
    for( i = 0; i < order; i++ ) {
        coefs_Q24[ i ] = silk_SMULWW( gain_Q16, coefs_Q24[ i ] );
    }
    limit_Q20 = silk_RSHIFT(limit_Q24, 4);
    for( iter = 0; iter < 10; iter++ ) {
        /* Find maximum absolute value */
        maxabs_Q24 = -1;
        for( i = 0; i < order; i++ ) {
            tmp = silk_abs_int32( coefs_Q24[ i ] );
            if( tmp > maxabs_Q24 ) {
                maxabs_Q24 = tmp;
                ind = i;
            }
        }
        /* Use Q20 to avoid any overflow when multiplying by (ind + 1) later. */
        maxabs_Q20 = silk_RSHIFT(maxabs_Q24, 4);
        if( maxabs_Q20 <= limit_Q20 ) {
            /* Coefficients are within range - done */
            return;
        }

        /* Convert back to true warped coefficients */
        for( i = 1; i < order; i++ ) {
            coefs_Q24[ i - 1 ] = silk_SMLAWB( coefs_Q24[ i - 1 ], coefs_Q24[ i ], lambda_Q16 );
        }
        gain_Q16 = silk_INVERSE32_varQ( gain_Q16, 32 );
        for( i = 0; i < order; i++ ) {
            coefs_Q24[ i ] = silk_SMULWW( gain_Q16, coefs_Q24[ i ] );
        }

        /* Apply bandwidth expansion */
        chirp_Q16 = SILK_FIX_CONST( 0.99, 16 ) - silk_DIV32_varQ(
            silk_SMULWB( maxabs_Q20 - limit_Q20, silk_SMLABB( SILK_FIX_CONST( 0.8, 10 ), SILK_FIX_CONST( 0.1, 10 ), iter ) ),
            silk_MUL( maxabs_Q20, ind + 1 ), 22 );
        silk_bwexpander_32( coefs_Q24, order, chirp_Q16 );

        /* Convert to monic warped coefficients */
        lambda_Q16 = -lambda_Q16;
        for( i = order - 1; i > 0; i-- ) {
            coefs_Q24[ i - 1 ] = silk_SMLAWB( coefs_Q24[ i - 1 ], coefs_Q24[ i ], lambda_Q16 );
        }
        lambda_Q16 = -lambda_Q16;
        nom_Q16  = silk_SMLAWB( SILK_FIX_CONST( 1.0, 16 ), -(opus_int32)lambda_Q16,        lambda_Q16 );
        den_Q24  = silk_SMLAWB( SILK_FIX_CONST( 1.0, 24 ), coefs_Q24[ 0 ], lambda_Q16 );
        gain_Q16 = silk_DIV32_varQ( nom_Q16, den_Q24, 24 );
        for( i = 0; i < order; i++ ) {
            coefs_Q24[ i ] = silk_SMULWW( gain_Q16, coefs_Q24[ i ] );
        }
    }
    silk_assert( 0 );
}

/* Disable MIPS version until it's updated. */
#if 0 && defined(MIPSr1_ASM)
#include "mips/noise_shape_analysis_FIX_mipsr1.h"
#endif

/**************************************************************/
/* Compute noise shaping coefficients and initial gain values */
/**************************************************************/
#ifndef OVERRIDE_silk_noise_shape_analysis_FIX
void silk_noise_shape_analysis_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  Encoder state FIX                                                           */
    silk_encoder_control_FIX        *psEncCtrl,                             /* I/O  Encoder control FIX                                                         */
    const opus_int16                *pitch_res,                             /* I    LPC residual from pitch analysis                                            */
    const opus_int16                *x,                                     /* I    Input signal [ frame_length + la_shape ]                                    */
    int                              arch                                   /* I    Run-time architecture                                                       */
)
{
    silk_shape_state_FIX *psShapeSt = &psEnc->sShape;
    opus_int     k, i, nSamples, nSegs, Qnrg, b_Q14, warping_Q16, scale = 0;
    opus_int32   SNR_adj_dB_Q7, HarmShapeGain_Q16, Tilt_Q16, tmp32;
    opus_int32   nrg, log_energy_Q7, log_energy_prev_Q7, energy_variation_Q7;
    opus_int32   BWExp_Q16, gain_mult_Q16, gain_add_Q16, strength_Q16, b_Q8;
    opus_int32   auto_corr[     MAX_SHAPE_LPC_ORDER + 1 ];
    opus_int32   refl_coef_Q16[ MAX_SHAPE_LPC_ORDER ];
    opus_int32   AR_Q24[       MAX_SHAPE_LPC_ORDER ];
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
    } else {
        /* Sparseness measure, based on relative fluctuations of energy per 2 milliseconds */
        nSamples = silk_LSHIFT( psEnc->sCmn.fs_kHz, 1 );
        energy_variation_Q7 = 0;
        log_energy_prev_Q7  = 0;
        pitch_res_ptr = pitch_res;
        nSegs = silk_SMULBB( SUB_FRAME_LENGTH_MS, psEnc->sCmn.nb_subfr ) / 2;
        for( k = 0; k < nSegs; k++ ) {
            silk_sum_sqr_shift( &nrg, &scale, pitch_res_ptr, nSamples );
            nrg += silk_RSHIFT( nSamples, scale );           /* Q(-scale)*/

            log_energy_Q7 = silk_lin2log( nrg );
            if( k > 0 ) {
                energy_variation_Q7 += silk_abs( log_energy_Q7 - log_energy_prev_Q7 );
            }
            log_energy_prev_Q7 = log_energy_Q7;
            pitch_res_ptr += nSamples;
        }

        /* Set quantization offset depending on sparseness measure */
        if( energy_variation_Q7 > SILK_FIX_CONST( ENERGY_VARIATION_THRESHOLD_QNT_OFFSET, 7 ) * (nSegs-1) ) {
            psEnc->sCmn.indices.quantOffsetType = 0;
        } else {
            psEnc->sCmn.indices.quantOffsetType = 1;
        }
    }

    /*******************************/
    /* Control bandwidth expansion */
    /*******************************/
    /* More BWE for signals with high prediction gain */
    strength_Q16 = silk_SMULWB( psEncCtrl->predGain_Q16, SILK_FIX_CONST( FIND_PITCH_WHITE_NOISE_FRACTION, 16 ) );
    BWExp_Q16 = silk_DIV32_varQ( SILK_FIX_CONST( BANDWIDTH_EXPANSION, 16 ),
        silk_SMLAWW( SILK_FIX_CONST( 1.0, 16 ), strength_Q16, strength_Q16 ), 16 );

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
            silk_warped_autocorrelation_FIX( auto_corr, &scale, x_windowed, warping_Q16, psEnc->sCmn.shapeWinLength, psEnc->sCmn.shapingLPCOrder, arch );
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
        silk_k2a_Q16( AR_Q24, refl_coef_Q16, psEnc->sCmn.shapingLPCOrder );

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

        psEncCtrl->Gains_Q16[ k ] = silk_LSHIFT_SAT32( tmp32, 16 - Qnrg );

        if( psEnc->sCmn.warping_Q16 > 0 ) {
            /* Adjust gain for warping */
            gain_mult_Q16 = warped_gain( AR_Q24, warping_Q16, psEnc->sCmn.shapingLPCOrder );
            silk_assert( psEncCtrl->Gains_Q16[ k ] > 0 );
            if( psEncCtrl->Gains_Q16[ k ] < SILK_FIX_CONST( 0.25, 16 ) ) {
                psEncCtrl->Gains_Q16[ k ] = silk_SMULWW( psEncCtrl->Gains_Q16[ k ], gain_mult_Q16 );
            } else {
                psEncCtrl->Gains_Q16[ k ] = silk_SMULWW( silk_RSHIFT_ROUND( psEncCtrl->Gains_Q16[ k ], 1 ), gain_mult_Q16 );
                if ( psEncCtrl->Gains_Q16[ k ] >= ( silk_int32_MAX >> 1 ) ) {
                    psEncCtrl->Gains_Q16[ k ] = silk_int32_MAX;
                } else {
                    psEncCtrl->Gains_Q16[ k ] = silk_LSHIFT32( psEncCtrl->Gains_Q16[ k ], 1 );
                }
            }
            silk_assert( psEncCtrl->Gains_Q16[ k ] > 0 );
        }

        /* Bandwidth expansion */
        silk_bwexpander_32( AR_Q24, psEnc->sCmn.shapingLPCOrder, BWExp_Q16 );

        if( psEnc->sCmn.warping_Q16 > 0 ) {
            /* Convert to monic warped prediction coefficients and limit absolute values */
            limit_warped_coefs( AR_Q24, warping_Q16, SILK_FIX_CONST( 3.999, 24 ), psEnc->sCmn.shapingLPCOrder );

            /* Convert from Q24 to Q13 and store in int16 */
            for( i = 0; i < psEnc->sCmn.shapingLPCOrder; i++ ) {
                psEncCtrl->AR_Q13[ k * MAX_SHAPE_LPC_ORDER + i ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( AR_Q24[ i ], 11 ) );
            }
        } else {
            silk_LPC_fit( &psEncCtrl->AR_Q13[ k * MAX_SHAPE_LPC_ORDER ], AR_Q24, 13, 24, psEnc->sCmn.shapingLPCOrder );
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
        psShapeSt->HarmShapeGain_smth_Q16 =
            silk_SMLAWB( psShapeSt->HarmShapeGain_smth_Q16, HarmShapeGain_Q16 - psShapeSt->HarmShapeGain_smth_Q16, SILK_FIX_CONST( SUBFR_SMTH_COEF, 16 ) );
        psShapeSt->Tilt_smth_Q16 =
            silk_SMLAWB( psShapeSt->Tilt_smth_Q16,          Tilt_Q16          - psShapeSt->Tilt_smth_Q16,          SILK_FIX_CONST( SUBFR_SMTH_COEF, 16 ) );

        psEncCtrl->HarmShapeGain_Q14[ k ] = ( opus_int )silk_RSHIFT_ROUND( psShapeSt->HarmShapeGain_smth_Q16, 2 );
        psEncCtrl->Tilt_Q14[ k ]          = ( opus_int )silk_RSHIFT_ROUND( psShapeSt->Tilt_smth_Q16,          2 );
    }
    RESTORE_STACK;
}
#endif /* OVERRIDE_silk_noise_shape_analysis_FIX */
