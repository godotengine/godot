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

#ifndef SILK_STRUCTS_FLP_H
#define SILK_STRUCTS_FLP_H

#include "typedef.h"
#include "main.h"
#include "structs.h"


/********************************/
/* Noise shaping analysis state */
/********************************/
typedef struct {
    opus_int8                   LastGainIndex;
    silk_float                  HarmShapeGain_smth;
    silk_float                  Tilt_smth;
} silk_shape_state_FLP;

/********************************/
/* Encoder state FLP            */
/********************************/
typedef struct {
    silk_encoder_state          sCmn;                               /* Common struct, shared with fixed-point code */
    silk_shape_state_FLP        sShape;                             /* Noise shaping state */

    /* Buffer for find pitch and noise shape analysis */
    silk_float                  x_buf[ 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX ];/* Buffer for find pitch and noise shape analysis */
    silk_float                  LTPCorr;                            /* Normalized correlation from pitch lag estimator */
} silk_encoder_state_FLP;

/************************/
/* Encoder control FLP  */
/************************/
typedef struct {
    /* Prediction and coding parameters */
    silk_float                  Gains[ MAX_NB_SUBFR ];
    silk_float                  PredCoef[ 2 ][ MAX_LPC_ORDER ];     /* holds interpolated and final coefficients */
    silk_float                  LTPCoef[LTP_ORDER * MAX_NB_SUBFR];
    silk_float                  LTP_scale;
    opus_int                    pitchL[ MAX_NB_SUBFR ];

    /* Noise shaping parameters */
    silk_float                  AR[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ];
    silk_float                  LF_MA_shp[     MAX_NB_SUBFR ];
    silk_float                  LF_AR_shp[     MAX_NB_SUBFR ];
    silk_float                  Tilt[          MAX_NB_SUBFR ];
    silk_float                  HarmShapeGain[ MAX_NB_SUBFR ];
    silk_float                  Lambda;
    silk_float                  input_quality;
    silk_float                  coding_quality;

    /* Measures */
    silk_float                  predGain;
    silk_float                  LTPredCodGain;
    silk_float                  ResNrg[ MAX_NB_SUBFR ];             /* Residual energy per subframe */

    /* Parameters for CBR mode */
    opus_int32                  GainsUnq_Q16[ MAX_NB_SUBFR ];
    opus_int8                   lastGainIndexPrev;
} silk_encoder_control_FLP;

/************************/
/* Encoder Super Struct */
/************************/
typedef struct {
    stereo_enc_state            sStereo;
    opus_int32                  nBitsUsedLBRR;
    opus_int32                  nBitsExceeded;
    opus_int                    nChannelsAPI;
    opus_int                    nChannelsInternal;
    opus_int                    nPrevChannelsInternal;
    opus_int                    timeSinceSwitchAllowed_ms;
    opus_int                    allowBandwidthSwitch;
    opus_int                    prev_decode_only_middle;
    /* This needs to be last so we can skip the second state for mono. */
    silk_encoder_state_FLP      state_Fxx[ ENCODER_NUM_CHANNELS ];
} silk_encoder;

#endif
