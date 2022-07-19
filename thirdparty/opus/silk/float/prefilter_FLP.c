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

/*
* Prefilter for finding Quantizer input signal
*/
static OPUS_INLINE void silk_prefilt_FLP(
    silk_prefilter_state_FLP    *P,                 /* I/O state */
    silk_float                  st_res[],           /* I */
    silk_float                  xw[],               /* O */
    silk_float                  *HarmShapeFIR,      /* I */
    silk_float                  Tilt,               /* I */
    silk_float                  LF_MA_shp,          /* I */
    silk_float                  LF_AR_shp,          /* I */
    opus_int                    lag,                /* I */
    opus_int                    length              /* I */
);

static void silk_warped_LPC_analysis_filter_FLP(
          silk_float                 state[],            /* I/O  State [order + 1]                       */
          silk_float                 res[],              /* O    Residual signal [length]                */
    const silk_float                 coef[],             /* I    Coefficients [order]                    */
    const silk_float                 input[],            /* I    Input signal [length]                   */
    const silk_float                 lambda,             /* I    Warping factor                          */
    const opus_int                   length,             /* I    Length of input signal                  */
    const opus_int                   order               /* I    Filter order (even)                     */
)
{
    opus_int     n, i;
    silk_float   acc, tmp1, tmp2;

    /* Order must be even */
    silk_assert( ( order & 1 ) == 0 );

    for( n = 0; n < length; n++ ) {
        /* Output of lowpass section */
        tmp2 = state[ 0 ] + lambda * state[ 1 ];
        state[ 0 ] = input[ n ];
        /* Output of allpass section */
        tmp1 = state[ 1 ] + lambda * ( state[ 2 ] - tmp2 );
        state[ 1 ] = tmp2;
        acc = coef[ 0 ] * tmp2;
        /* Loop over allpass sections */
        for( i = 2; i < order; i += 2 ) {
            /* Output of allpass section */
            tmp2 = state[ i ] + lambda * ( state[ i + 1 ] - tmp1 );
            state[ i ] = tmp1;
            acc += coef[ i - 1 ] * tmp1;
            /* Output of allpass section */
            tmp1 = state[ i + 1 ] + lambda * ( state[ i + 2 ] - tmp2 );
            state[ i + 1 ] = tmp2;
            acc += coef[ i ] * tmp2;
        }
        state[ order ] = tmp1;
        acc += coef[ order - 1 ] * tmp1;
        res[ n ] = input[ n ] - acc;
    }
}

/*
* silk_prefilter. Main prefilter function
*/
void silk_prefilter_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    const silk_encoder_control_FLP  *psEncCtrl,                         /* I    Encoder control FLP                         */
    silk_float                      xw[],                               /* O    Weighted signal                             */
    const silk_float                x[]                                 /* I    Speech signal                               */
)
{
    silk_prefilter_state_FLP *P = &psEnc->sPrefilt;
    opus_int   j, k, lag;
    silk_float HarmShapeGain, Tilt, LF_MA_shp, LF_AR_shp;
    silk_float B[ 2 ];
    const silk_float *AR1_shp;
    const silk_float *px;
    silk_float *pxw;
    silk_float HarmShapeFIR[ 3 ];
    silk_float st_res[ MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER ];

    /* Set up pointers */
    px  = x;
    pxw = xw;
    lag = P->lagPrev;
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        /* Update Variables that change per sub frame */
        if( psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
            lag = psEncCtrl->pitchL[ k ];
        }

        /* Noise shape parameters */
        HarmShapeGain = psEncCtrl->HarmShapeGain[ k ] * ( 1.0f - psEncCtrl->HarmBoost[ k ] );
        HarmShapeFIR[ 0 ] = 0.25f               * HarmShapeGain;
        HarmShapeFIR[ 1 ] = 32767.0f / 65536.0f * HarmShapeGain;
        HarmShapeFIR[ 2 ] = 0.25f               * HarmShapeGain;
        Tilt      =  psEncCtrl->Tilt[ k ];
        LF_MA_shp =  psEncCtrl->LF_MA_shp[ k ];
        LF_AR_shp =  psEncCtrl->LF_AR_shp[ k ];
        AR1_shp   = &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ];

        /* Short term FIR filtering */
        silk_warped_LPC_analysis_filter_FLP( P->sAR_shp, st_res, AR1_shp, px,
            (silk_float)psEnc->sCmn.warping_Q16 / 65536.0f, psEnc->sCmn.subfr_length, psEnc->sCmn.shapingLPCOrder );

        /* Reduce (mainly) low frequencies during harmonic emphasis */
        B[ 0 ] =  psEncCtrl->GainsPre[ k ];
        B[ 1 ] = -psEncCtrl->GainsPre[ k ] *
            ( psEncCtrl->HarmBoost[ k ] * HarmShapeGain + INPUT_TILT + psEncCtrl->coding_quality * HIGH_RATE_INPUT_TILT );
        pxw[ 0 ] = B[ 0 ] * st_res[ 0 ] + B[ 1 ] * P->sHarmHP;
        for( j = 1; j < psEnc->sCmn.subfr_length; j++ ) {
            pxw[ j ] = B[ 0 ] * st_res[ j ] + B[ 1 ] * st_res[ j - 1 ];
        }
        P->sHarmHP = st_res[ psEnc->sCmn.subfr_length - 1 ];

        silk_prefilt_FLP( P, pxw, pxw, HarmShapeFIR, Tilt, LF_MA_shp, LF_AR_shp, lag, psEnc->sCmn.subfr_length );

        px  += psEnc->sCmn.subfr_length;
        pxw += psEnc->sCmn.subfr_length;
    }
    P->lagPrev = psEncCtrl->pitchL[ psEnc->sCmn.nb_subfr - 1 ];
}

/*
* Prefilter for finding Quantizer input signal
*/
static OPUS_INLINE void silk_prefilt_FLP(
    silk_prefilter_state_FLP    *P,                 /* I/O state */
    silk_float                  st_res[],           /* I */
    silk_float                  xw[],               /* O */
    silk_float                  *HarmShapeFIR,      /* I */
    silk_float                  Tilt,               /* I */
    silk_float                  LF_MA_shp,          /* I */
    silk_float                  LF_AR_shp,          /* I */
    opus_int                    lag,                /* I */
    opus_int                    length              /* I */
)
{
    opus_int   i;
    opus_int   idx, LTP_shp_buf_idx;
    silk_float n_Tilt, n_LF, n_LTP;
    silk_float sLF_AR_shp, sLF_MA_shp;
    silk_float *LTP_shp_buf;

    /* To speed up use temp variables instead of using the struct */
    LTP_shp_buf     = P->sLTP_shp;
    LTP_shp_buf_idx = P->sLTP_shp_buf_idx;
    sLF_AR_shp      = P->sLF_AR_shp;
    sLF_MA_shp      = P->sLF_MA_shp;

    for( i = 0; i < length; i++ ) {
        if( lag > 0 ) {
            silk_assert( HARM_SHAPE_FIR_TAPS == 3 );
            idx = lag + LTP_shp_buf_idx;
            n_LTP  = LTP_shp_buf[ ( idx - HARM_SHAPE_FIR_TAPS / 2 - 1) & LTP_MASK ] * HarmShapeFIR[ 0 ];
            n_LTP += LTP_shp_buf[ ( idx - HARM_SHAPE_FIR_TAPS / 2    ) & LTP_MASK ] * HarmShapeFIR[ 1 ];
            n_LTP += LTP_shp_buf[ ( idx - HARM_SHAPE_FIR_TAPS / 2 + 1) & LTP_MASK ] * HarmShapeFIR[ 2 ];
        } else {
            n_LTP = 0;
        }

        n_Tilt = sLF_AR_shp * Tilt;
        n_LF   = sLF_AR_shp * LF_AR_shp + sLF_MA_shp * LF_MA_shp;

        sLF_AR_shp = st_res[ i ] - n_Tilt;
        sLF_MA_shp = sLF_AR_shp - n_LF;

        LTP_shp_buf_idx = ( LTP_shp_buf_idx - 1 ) & LTP_MASK;
        LTP_shp_buf[ LTP_shp_buf_idx ] = sLF_MA_shp;

        xw[ i ] = sLF_MA_shp - n_LTP;
    }
    /* Copy temp variable back to state */
    P->sLF_AR_shp       = sLF_AR_shp;
    P->sLF_MA_shp       = sLF_MA_shp;
    P->sLTP_shp_buf_idx = LTP_shp_buf_idx;
}
