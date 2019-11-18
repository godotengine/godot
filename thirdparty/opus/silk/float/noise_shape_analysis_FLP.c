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

#include "main_FLP.h"
#include "tuning_parameters.h"

/* Compute gain to make warped filter coefficients have a zero mean log frequency response on a   */
/* non-warped frequency scale. (So that it can be implemented with a minimum-phase monic filter.) */
/* Note: A monic filter is one with the first coefficient equal to 1.0. In Silk we omit the first */
/* coefficient in an array of coefficients, for monic filters.                                    */
static OPUS_INLINE silk_float warped_gain(
    const silk_float     *coefs,
    silk_float           lambda,
    opus_int             order
) {
    opus_int   i;
    silk_float gain;

    lambda = -lambda;
    gain = coefs[ order - 1 ];
    for( i = order - 2; i >= 0; i-- ) {
        gain = lambda * gain + coefs[ i ];
    }
    return (silk_float)( 1.0f / ( 1.0f - lambda * gain ) );
}

/* Convert warped filter coefficients to monic pseudo-warped coefficients and limit maximum     */
/* amplitude of monic warped coefficients by using bandwidth expansion on the true coefficients */
static OPUS_INLINE void warped_true2monic_coefs(
    silk_float           *coefs_syn,
    silk_float           *coefs_ana,
    silk_float           lambda,
    silk_float           limit,
    opus_int             order
) {
    opus_int   i, iter, ind = 0;
    silk_float tmp, maxabs, chirp, gain_syn, gain_ana;

    /* Convert to monic coefficients */
    for( i = order - 1; i > 0; i-- ) {
        coefs_syn[ i - 1 ] -= lambda * coefs_syn[ i ];
        coefs_ana[ i - 1 ] -= lambda * coefs_ana[ i ];
    }
    gain_syn = ( 1.0f - lambda * lambda ) / ( 1.0f + lambda * coefs_syn[ 0 ] );
    gain_ana = ( 1.0f - lambda * lambda ) / ( 1.0f + lambda * coefs_ana[ 0 ] );
    for( i = 0; i < order; i++ ) {
        coefs_syn[ i ] *= gain_syn;
        coefs_ana[ i ] *= gain_ana;
    }

    /* Limit */
    for( iter = 0; iter < 10; iter++ ) {
        /* Find maximum absolute value */
        maxabs = -1.0f;
        for( i = 0; i < order; i++ ) {
            tmp = silk_max( silk_abs_float( coefs_syn[ i ] ), silk_abs_float( coefs_ana[ i ] ) );
            if( tmp > maxabs ) {
                maxabs = tmp;
                ind = i;
            }
        }
        if( maxabs <= limit ) {
            /* Coefficients are within range - done */
            return;
        }

        /* Convert back to true warped coefficients */
        for( i = 1; i < order; i++ ) {
            coefs_syn[ i - 1 ] += lambda * coefs_syn[ i ];
            coefs_ana[ i - 1 ] += lambda * coefs_ana[ i ];
        }
        gain_syn = 1.0f / gain_syn;
        gain_ana = 1.0f / gain_ana;
        for( i = 0; i < order; i++ ) {
            coefs_syn[ i ] *= gain_syn;
            coefs_ana[ i ] *= gain_ana;
        }

        /* Apply bandwidth expansion */
        chirp = 0.99f - ( 0.8f + 0.1f * iter ) * ( maxabs - limit ) / ( maxabs * ( ind + 1 ) );
        silk_bwexpander_FLP( coefs_syn, order, chirp );
        silk_bwexpander_FLP( coefs_ana, order, chirp );

        /* Convert to monic warped coefficients */
        for( i = order - 1; i > 0; i-- ) {
            coefs_syn[ i - 1 ] -= lambda * coefs_syn[ i ];
            coefs_ana[ i - 1 ] -= lambda * coefs_ana[ i ];
        }
        gain_syn = ( 1.0f - lambda * lambda ) / ( 1.0f + lambda * coefs_syn[ 0 ] );
        gain_ana = ( 1.0f - lambda * lambda ) / ( 1.0f + lambda * coefs_ana[ 0 ] );
        for( i = 0; i < order; i++ ) {
            coefs_syn[ i ] *= gain_syn;
            coefs_ana[ i ] *= gain_ana;
        }
    }
    silk_assert( 0 );
}

/* Compute noise shaping coefficients and initial gain values */
void silk_noise_shape_analysis_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,                         /* I/O  Encoder control FLP                         */
    const silk_float                *pitch_res,                         /* I    LPC residual from pitch analysis            */
    const silk_float                *x                                  /* I    Input signal [frame_length + la_shape]      */
)
{
    silk_shape_state_FLP *psShapeSt = &psEnc->sShape;
    opus_int     k, nSamples;
    silk_float   SNR_adj_dB, HarmBoost, HarmShapeGain, Tilt;
    silk_float   nrg, pre_nrg, log_energy, log_energy_prev, energy_variation;
    silk_float   delta, BWExp1, BWExp2, gain_mult, gain_add, strength, b, warping;
    silk_float   x_windowed[ SHAPE_LPC_WIN_MAX ];
    silk_float   auto_corr[ MAX_SHAPE_LPC_ORDER + 1 ];
    const silk_float *x_ptr, *pitch_res_ptr;

    /* Point to start of first LPC analysis block */
    x_ptr = x - psEnc->sCmn.la_shape;

    /****************/
    /* GAIN CONTROL */
    /****************/
    SNR_adj_dB = psEnc->sCmn.SNR_dB_Q7 * ( 1 / 128.0f );

    /* Input quality is the average of the quality in the lowest two VAD bands */
    psEncCtrl->input_quality = 0.5f * ( psEnc->sCmn.input_quality_bands_Q15[ 0 ] + psEnc->sCmn.input_quality_bands_Q15[ 1 ] ) * ( 1.0f / 32768.0f );

    /* Coding quality level, between 0.0 and 1.0 */
    psEncCtrl->coding_quality = silk_sigmoid( 0.25f * ( SNR_adj_dB - 20.0f ) );

    if( psEnc->sCmn.useCBR == 0 ) {
        /* Reduce coding SNR during low speech activity */
        b = 1.0f - psEnc->sCmn.speech_activity_Q8 * ( 1.0f /  256.0f );
        SNR_adj_dB -= BG_SNR_DECR_dB * psEncCtrl->coding_quality * ( 0.5f + 0.5f * psEncCtrl->input_quality ) * b * b;
    }

    if( psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
        /* Reduce gains for periodic signals */
        SNR_adj_dB += HARM_SNR_INCR_dB * psEnc->LTPCorr;
    } else {
        /* For unvoiced signals and low-quality input, adjust the quality slower than SNR_dB setting */
        SNR_adj_dB += ( -0.4f * psEnc->sCmn.SNR_dB_Q7 * ( 1 / 128.0f ) + 6.0f ) * ( 1.0f - psEncCtrl->input_quality );
    }

    /*************************/
    /* SPARSENESS PROCESSING */
    /*************************/
    /* Set quantizer offset */
    if( psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
        /* Initially set to 0; may be overruled in process_gains(..) */
        psEnc->sCmn.indices.quantOffsetType = 0;
        psEncCtrl->sparseness = 0.0f;
    } else {
        /* Sparseness measure, based on relative fluctuations of energy per 2 milliseconds */
        nSamples = 2 * psEnc->sCmn.fs_kHz;
        energy_variation = 0.0f;
        log_energy_prev  = 0.0f;
        pitch_res_ptr = pitch_res;
        for( k = 0; k < silk_SMULBB( SUB_FRAME_LENGTH_MS, psEnc->sCmn.nb_subfr ) / 2; k++ ) {
            nrg = ( silk_float )nSamples + ( silk_float )silk_energy_FLP( pitch_res_ptr, nSamples );
            log_energy = silk_log2( nrg );
            if( k > 0 ) {
                energy_variation += silk_abs_float( log_energy - log_energy_prev );
            }
            log_energy_prev = log_energy;
            pitch_res_ptr += nSamples;
        }
        psEncCtrl->sparseness = silk_sigmoid( 0.4f * ( energy_variation - 5.0f ) );

        /* Set quantization offset depending on sparseness measure */
        if( psEncCtrl->sparseness > SPARSENESS_THRESHOLD_QNT_OFFSET ) {
            psEnc->sCmn.indices.quantOffsetType = 0;
        } else {
            psEnc->sCmn.indices.quantOffsetType = 1;
        }

        /* Increase coding SNR for sparse signals */
        SNR_adj_dB += SPARSE_SNR_INCR_dB * ( psEncCtrl->sparseness - 0.5f );
    }

    /*******************************/
    /* Control bandwidth expansion */
    /*******************************/
    /* More BWE for signals with high prediction gain */
    strength = FIND_PITCH_WHITE_NOISE_FRACTION * psEncCtrl->predGain;           /* between 0.0 and 1.0 */
    BWExp1 = BWExp2 = BANDWIDTH_EXPANSION / ( 1.0f + strength * strength );
    delta  = LOW_RATE_BANDWIDTH_EXPANSION_DELTA * ( 1.0f - 0.75f * psEncCtrl->coding_quality );
    BWExp1 -= delta;
    BWExp2 += delta;
    /* BWExp1 will be applied after BWExp2, so make it relative */
    BWExp1 /= BWExp2;

    if( psEnc->sCmn.warping_Q16 > 0 ) {
        /* Slightly more warping in analysis will move quantization noise up in frequency, where it's better masked */
        warping = (silk_float)psEnc->sCmn.warping_Q16 / 65536.0f + 0.01f * psEncCtrl->coding_quality;
    } else {
        warping = 0.0f;
    }

    /********************************************/
    /* Compute noise shaping AR coefs and gains */
    /********************************************/
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        /* Apply window: sine slope followed by flat part followed by cosine slope */
        opus_int shift, slope_part, flat_part;
        flat_part = psEnc->sCmn.fs_kHz * 3;
        slope_part = ( psEnc->sCmn.shapeWinLength - flat_part ) / 2;

        silk_apply_sine_window_FLP( x_windowed, x_ptr, 1, slope_part );
        shift = slope_part;
        silk_memcpy( x_windowed + shift, x_ptr + shift, flat_part * sizeof(silk_float) );
        shift += flat_part;
        silk_apply_sine_window_FLP( x_windowed + shift, x_ptr + shift, 2, slope_part );

        /* Update pointer: next LPC analysis block */
        x_ptr += psEnc->sCmn.subfr_length;

        if( psEnc->sCmn.warping_Q16 > 0 ) {
            /* Calculate warped auto correlation */
            silk_warped_autocorrelation_FLP( auto_corr, x_windowed, warping,
                psEnc->sCmn.shapeWinLength, psEnc->sCmn.shapingLPCOrder );
        } else {
            /* Calculate regular auto correlation */
            silk_autocorrelation_FLP( auto_corr, x_windowed, psEnc->sCmn.shapeWinLength, psEnc->sCmn.shapingLPCOrder + 1 );
        }

        /* Add white noise, as a fraction of energy */
        auto_corr[ 0 ] += auto_corr[ 0 ] * SHAPE_WHITE_NOISE_FRACTION;

        /* Convert correlations to prediction coefficients, and compute residual energy */
        nrg = silk_levinsondurbin_FLP( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], auto_corr, psEnc->sCmn.shapingLPCOrder );
        psEncCtrl->Gains[ k ] = ( silk_float )sqrt( nrg );

        if( psEnc->sCmn.warping_Q16 > 0 ) {
            /* Adjust gain for warping */
            psEncCtrl->Gains[ k ] *= warped_gain( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], warping, psEnc->sCmn.shapingLPCOrder );
        }

        /* Bandwidth expansion for synthesis filter shaping */
        silk_bwexpander_FLP( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], psEnc->sCmn.shapingLPCOrder, BWExp2 );

        /* Compute noise shaping filter coefficients */
        silk_memcpy(
            &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ],
            &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ],
            psEnc->sCmn.shapingLPCOrder * sizeof( silk_float ) );

        /* Bandwidth expansion for analysis filter shaping */
        silk_bwexpander_FLP( &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ], psEnc->sCmn.shapingLPCOrder, BWExp1 );

        /* Ratio of prediction gains, in energy domain */
        pre_nrg = silk_LPC_inverse_pred_gain_FLP( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], psEnc->sCmn.shapingLPCOrder );
        nrg     = silk_LPC_inverse_pred_gain_FLP( &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ], psEnc->sCmn.shapingLPCOrder );
        psEncCtrl->GainsPre[ k ] = 1.0f - 0.7f * ( 1.0f - pre_nrg / nrg );

        /* Convert to monic warped prediction coefficients and limit absolute values */
        warped_true2monic_coefs( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ],
            warping, 3.999f, psEnc->sCmn.shapingLPCOrder );
    }

    /*****************/
    /* Gain tweaking */
    /*****************/
    /* Increase gains during low speech activity */
    gain_mult = (silk_float)pow( 2.0f, -0.16f * SNR_adj_dB );
    gain_add  = (silk_float)pow( 2.0f,  0.16f * MIN_QGAIN_DB );
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        psEncCtrl->Gains[ k ] *= gain_mult;
        psEncCtrl->Gains[ k ] += gain_add;
    }

    gain_mult = 1.0f + INPUT_TILT + psEncCtrl->coding_quality * HIGH_RATE_INPUT_TILT;
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        psEncCtrl->GainsPre[ k ] *= gain_mult;
    }

    /************************************************/
    /* Control low-frequency shaping and noise tilt */
    /************************************************/
    /* Less low frequency shaping for noisy inputs */
    strength = LOW_FREQ_SHAPING * ( 1.0f + LOW_QUALITY_LOW_FREQ_SHAPING_DECR * ( psEnc->sCmn.input_quality_bands_Q15[ 0 ] * ( 1.0f / 32768.0f ) - 1.0f ) );
    strength *= psEnc->sCmn.speech_activity_Q8 * ( 1.0f /  256.0f );
    if( psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
        /* Reduce low frequencies quantization noise for periodic signals, depending on pitch lag */
        /*f = 400; freqz([1, -0.98 + 2e-4 * f], [1, -0.97 + 7e-4 * f], 2^12, Fs); axis([0, 1000, -10, 1])*/
        for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
            b = 0.2f / psEnc->sCmn.fs_kHz + 3.0f / psEncCtrl->pitchL[ k ];
            psEncCtrl->LF_MA_shp[ k ] = -1.0f + b;
            psEncCtrl->LF_AR_shp[ k ] =  1.0f - b - b * strength;
        }
        Tilt = - HP_NOISE_COEF -
            (1 - HP_NOISE_COEF) * HARM_HP_NOISE_COEF * psEnc->sCmn.speech_activity_Q8 * ( 1.0f /  256.0f );
    } else {
        b = 1.3f / psEnc->sCmn.fs_kHz;
        psEncCtrl->LF_MA_shp[ 0 ] = -1.0f + b;
        psEncCtrl->LF_AR_shp[ 0 ] =  1.0f - b - b * strength * 0.6f;
        for( k = 1; k < psEnc->sCmn.nb_subfr; k++ ) {
            psEncCtrl->LF_MA_shp[ k ] = psEncCtrl->LF_MA_shp[ 0 ];
            psEncCtrl->LF_AR_shp[ k ] = psEncCtrl->LF_AR_shp[ 0 ];
        }
        Tilt = -HP_NOISE_COEF;
    }

    /****************************/
    /* HARMONIC SHAPING CONTROL */
    /****************************/
    /* Control boosting of harmonic frequencies */
    HarmBoost = LOW_RATE_HARMONIC_BOOST * ( 1.0f - psEncCtrl->coding_quality ) * psEnc->LTPCorr;

    /* More harmonic boost for noisy input signals */
    HarmBoost += LOW_INPUT_QUALITY_HARMONIC_BOOST * ( 1.0f - psEncCtrl->input_quality );

    if( USE_HARM_SHAPING && psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
        /* Harmonic noise shaping */
        HarmShapeGain = HARMONIC_SHAPING;

        /* More harmonic noise shaping for high bitrates or noisy input */
        HarmShapeGain += HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING *
            ( 1.0f - ( 1.0f - psEncCtrl->coding_quality ) * psEncCtrl->input_quality );

        /* Less harmonic noise shaping for less periodic signals */
        HarmShapeGain *= ( silk_float )sqrt( psEnc->LTPCorr );
    } else {
        HarmShapeGain = 0.0f;
    }

    /*************************/
    /* Smooth over subframes */
    /*************************/
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        psShapeSt->HarmBoost_smth     += SUBFR_SMTH_COEF * ( HarmBoost - psShapeSt->HarmBoost_smth );
        psEncCtrl->HarmBoost[ k ]      = psShapeSt->HarmBoost_smth;
        psShapeSt->HarmShapeGain_smth += SUBFR_SMTH_COEF * ( HarmShapeGain - psShapeSt->HarmShapeGain_smth );
        psEncCtrl->HarmShapeGain[ k ]  = psShapeSt->HarmShapeGain_smth;
        psShapeSt->Tilt_smth          += SUBFR_SMTH_COEF * ( Tilt - psShapeSt->Tilt_smth );
        psEncCtrl->Tilt[ k ]           = psShapeSt->Tilt_smth;
    }
}
