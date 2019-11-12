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

#include <stdlib.h>
#include "main_FLP.h"
#include "tuning_parameters.h"

void silk_find_pitch_lags_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,                         /* I/O  Encoder control FLP                         */
    silk_float                      res[],                              /* O    Residual                                    */
    const silk_float                x[],                                /* I    Speech signal                               */
    int                             arch                                /* I    Run-time architecture                       */
)
{
    opus_int   buf_len;
    silk_float thrhld, res_nrg;
    const silk_float *x_buf_ptr, *x_buf;
    silk_float auto_corr[ MAX_FIND_PITCH_LPC_ORDER + 1 ];
    silk_float A[         MAX_FIND_PITCH_LPC_ORDER ];
    silk_float refl_coef[ MAX_FIND_PITCH_LPC_ORDER ];
    silk_float Wsig[      FIND_PITCH_LPC_WIN_MAX ];
    silk_float *Wsig_ptr;

    /******************************************/
    /* Set up buffer lengths etc based on Fs  */
    /******************************************/
    buf_len = psEnc->sCmn.la_pitch + psEnc->sCmn.frame_length + psEnc->sCmn.ltp_mem_length;

    /* Safety check */
    celt_assert( buf_len >= psEnc->sCmn.pitch_LPC_win_length );

    x_buf = x - psEnc->sCmn.ltp_mem_length;

    /******************************************/
    /* Estimate LPC AR coeficients            */
    /******************************************/

    /* Calculate windowed signal */

    /* First LA_LTP samples */
    x_buf_ptr = x_buf + buf_len - psEnc->sCmn.pitch_LPC_win_length;
    Wsig_ptr  = Wsig;
    silk_apply_sine_window_FLP( Wsig_ptr, x_buf_ptr, 1, psEnc->sCmn.la_pitch );

    /* Middle non-windowed samples */
    Wsig_ptr  += psEnc->sCmn.la_pitch;
    x_buf_ptr += psEnc->sCmn.la_pitch;
    silk_memcpy( Wsig_ptr, x_buf_ptr, ( psEnc->sCmn.pitch_LPC_win_length - ( psEnc->sCmn.la_pitch << 1 ) ) * sizeof( silk_float ) );

    /* Last LA_LTP samples */
    Wsig_ptr  += psEnc->sCmn.pitch_LPC_win_length - ( psEnc->sCmn.la_pitch << 1 );
    x_buf_ptr += psEnc->sCmn.pitch_LPC_win_length - ( psEnc->sCmn.la_pitch << 1 );
    silk_apply_sine_window_FLP( Wsig_ptr, x_buf_ptr, 2, psEnc->sCmn.la_pitch );

    /* Calculate autocorrelation sequence */
    silk_autocorrelation_FLP( auto_corr, Wsig, psEnc->sCmn.pitch_LPC_win_length, psEnc->sCmn.pitchEstimationLPCOrder + 1 );

    /* Add white noise, as a fraction of the energy */
    auto_corr[ 0 ] += auto_corr[ 0 ] * FIND_PITCH_WHITE_NOISE_FRACTION + 1;

    /* Calculate the reflection coefficients using Schur */
    res_nrg = silk_schur_FLP( refl_coef, auto_corr, psEnc->sCmn.pitchEstimationLPCOrder );

    /* Prediction gain */
    psEncCtrl->predGain = auto_corr[ 0 ] / silk_max_float( res_nrg, 1.0f );

    /* Convert reflection coefficients to prediction coefficients */
    silk_k2a_FLP( A, refl_coef, psEnc->sCmn.pitchEstimationLPCOrder );

    /* Bandwidth expansion */
    silk_bwexpander_FLP( A, psEnc->sCmn.pitchEstimationLPCOrder, FIND_PITCH_BANDWIDTH_EXPANSION );

    /*****************************************/
    /* LPC analysis filtering                */
    /*****************************************/
    silk_LPC_analysis_filter_FLP( res, A, x_buf, buf_len, psEnc->sCmn.pitchEstimationLPCOrder );

    if( psEnc->sCmn.indices.signalType != TYPE_NO_VOICE_ACTIVITY && psEnc->sCmn.first_frame_after_reset == 0 ) {
        /* Threshold for pitch estimator */
        thrhld  = 0.6f;
        thrhld -= 0.004f * psEnc->sCmn.pitchEstimationLPCOrder;
        thrhld -= 0.1f   * psEnc->sCmn.speech_activity_Q8 * ( 1.0f /  256.0f );
        thrhld -= 0.15f  * (psEnc->sCmn.prevSignalType >> 1);
        thrhld -= 0.1f   * psEnc->sCmn.input_tilt_Q15 * ( 1.0f / 32768.0f );

        /*****************************************/
        /* Call Pitch estimator                  */
        /*****************************************/
        if( silk_pitch_analysis_core_FLP( res, psEncCtrl->pitchL, &psEnc->sCmn.indices.lagIndex,
            &psEnc->sCmn.indices.contourIndex, &psEnc->LTPCorr, psEnc->sCmn.prevLag, psEnc->sCmn.pitchEstimationThreshold_Q16 / 65536.0f,
            thrhld, psEnc->sCmn.fs_kHz, psEnc->sCmn.pitchEstimationComplexity, psEnc->sCmn.nb_subfr, arch ) == 0 )
        {
            psEnc->sCmn.indices.signalType = TYPE_VOICED;
        } else {
            psEnc->sCmn.indices.signalType = TYPE_UNVOICED;
        }
    } else {
        silk_memset( psEncCtrl->pitchL, 0, sizeof( psEncCtrl->pitchL ) );
        psEnc->sCmn.indices.lagIndex = 0;
        psEnc->sCmn.indices.contourIndex = 0;
        psEnc->LTPCorr = 0;
    }
}
