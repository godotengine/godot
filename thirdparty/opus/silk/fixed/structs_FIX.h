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

#ifndef SILK_STRUCTS_FIX_H
#define SILK_STRUCTS_FIX_H

#include "typedef.h"
#include "main.h"
#include "structs.h"

#ifdef __cplusplus
extern "C"
{
#endif

/********************************/
/* Noise shaping analysis state */
/********************************/
typedef struct {
    opus_int8                   LastGainIndex;
    opus_int32                  HarmBoost_smth_Q16;
    opus_int32                  HarmShapeGain_smth_Q16;
    opus_int32                  Tilt_smth_Q16;
} silk_shape_state_FIX;

/********************************/
/* Encoder state FIX            */
/********************************/
typedef struct {
    silk_encoder_state          sCmn;                                   /* Common struct, shared with floating-point code       */
    silk_shape_state_FIX        sShape;                                 /* Shape state                                          */

    /* Buffer for find pitch and noise shape analysis */
    silk_DWORD_ALIGN opus_int16 x_buf[ 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX ];/* Buffer for find pitch and noise shape analysis  */
    opus_int                    LTPCorr_Q15;                            /* Normalized correlation from pitch lag estimator      */
    opus_int32                    resNrgSmth;
} silk_encoder_state_FIX;

/************************/
/* Encoder control FIX  */
/************************/
typedef struct {
    /* Prediction and coding parameters */
    opus_int32                  Gains_Q16[ MAX_NB_SUBFR ];
    silk_DWORD_ALIGN opus_int16 PredCoef_Q12[ 2 ][ MAX_LPC_ORDER ];
    opus_int16                  LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ];
    opus_int                    LTP_scale_Q14;
    opus_int                    pitchL[ MAX_NB_SUBFR ];

    /* Noise shaping parameters */
    /* Testing */
    silk_DWORD_ALIGN opus_int16 AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ];
    opus_int32                  LF_shp_Q14[        MAX_NB_SUBFR ];      /* Packs two int16 coefficients per int32 value         */
    opus_int                    Tilt_Q14[          MAX_NB_SUBFR ];
    opus_int                    HarmShapeGain_Q14[ MAX_NB_SUBFR ];
    opus_int                    Lambda_Q10;
    opus_int                    input_quality_Q14;
    opus_int                    coding_quality_Q14;

    /* measures */
    opus_int32                  predGain_Q16;
    opus_int                    LTPredCodGain_Q7;
    opus_int32                  ResNrg[ MAX_NB_SUBFR ];                 /* Residual energy per subframe                         */
    opus_int                    ResNrgQ[ MAX_NB_SUBFR ];                /* Q domain for the residual energy > 0                 */

    /* Parameters for CBR mode */
    opus_int32                  GainsUnq_Q16[ MAX_NB_SUBFR ];
    opus_int8                   lastGainIndexPrev;
} silk_encoder_control_FIX;

/************************/
/* Encoder Super Struct */
/************************/
typedef struct {
    silk_encoder_state_FIX      state_Fxx[ ENCODER_NUM_CHANNELS ];
    stereo_enc_state            sStereo;
    opus_int32                  nBitsUsedLBRR;
    opus_int32                  nBitsExceeded;
    opus_int                    nChannelsAPI;
    opus_int                    nChannelsInternal;
    opus_int                    nPrevChannelsInternal;
    opus_int                    timeSinceSwitchAllowed_ms;
    opus_int                    allowBandwidthSwitch;
    opus_int                    prev_decode_only_middle;
} silk_encoder;


#ifdef __cplusplus
}
#endif

#endif
