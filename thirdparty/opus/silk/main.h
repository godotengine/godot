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

#ifndef SILK_MAIN_H
#define SILK_MAIN_H

#include "SigProc_FIX.h"
#include "define.h"
#include "structs.h"
#include "tables.h"
#include "PLC.h"
#include "control.h"
#include "debug.h"
#include "entenc.h"
#include "entdec.h"

#if defined(OPUS_X86_MAY_HAVE_SSE4_1)
#include "x86/main_sse.h"
#endif

/* Convert Left/Right stereo signal to adaptive Mid/Side representation */
void silk_stereo_LR_to_MS(
    stereo_enc_state            *state,                         /* I/O  State                                       */
    opus_int16                  x1[],                           /* I/O  Left input signal, becomes mid signal       */
    opus_int16                  x2[],                           /* I/O  Right input signal, becomes side signal     */
    opus_int8                   ix[ 2 ][ 3 ],                   /* O    Quantization indices                        */
    opus_int8                   *mid_only_flag,                 /* O    Flag: only mid signal coded                 */
    opus_int32                  mid_side_rates_bps[],           /* O    Bitrates for mid and side signals           */
    opus_int32                  total_rate_bps,                 /* I    Total bitrate                               */
    opus_int                    prev_speech_act_Q8,             /* I    Speech activity level in previous frame     */
    opus_int                    toMono,                         /* I    Last frame before a stereo->mono transition */
    opus_int                    fs_kHz,                         /* I    Sample rate (kHz)                           */
    opus_int                    frame_length                    /* I    Number of samples                           */
);

/* Convert adaptive Mid/Side representation to Left/Right stereo signal */
void silk_stereo_MS_to_LR(
    stereo_dec_state            *state,                         /* I/O  State                                       */
    opus_int16                  x1[],                           /* I/O  Left input signal, becomes mid signal       */
    opus_int16                  x2[],                           /* I/O  Right input signal, becomes side signal     */
    const opus_int32            pred_Q13[],                     /* I    Predictors                                  */
    opus_int                    fs_kHz,                         /* I    Samples rate (kHz)                          */
    opus_int                    frame_length                    /* I    Number of samples                           */
);

/* Find least-squares prediction gain for one signal based on another and quantize it */
opus_int32 silk_stereo_find_predictor(                          /* O    Returns predictor in Q13                    */
    opus_int32                  *ratio_Q14,                     /* O    Ratio of residual and mid energies          */
    const opus_int16            x[],                            /* I    Basis signal                                */
    const opus_int16            y[],                            /* I    Target signal                               */
    opus_int32                  mid_res_amp_Q0[],               /* I/O  Smoothed mid, residual norms                */
    opus_int                    length,                         /* I    Number of samples                           */
    opus_int                    smooth_coef_Q16                 /* I    Smoothing coefficient                       */
);

/* Quantize mid/side predictors */
void silk_stereo_quant_pred(
    opus_int32                  pred_Q13[],                     /* I/O  Predictors (out: quantized)                 */
    opus_int8                   ix[ 2 ][ 3 ]                    /* O    Quantization indices                        */
);

/* Entropy code the mid/side quantization indices */
void silk_stereo_encode_pred(
    ec_enc                      *psRangeEnc,                    /* I/O  Compressor data structure                   */
    opus_int8                   ix[ 2 ][ 3 ]                    /* I    Quantization indices                        */
);

/* Entropy code the mid-only flag */
void silk_stereo_encode_mid_only(
    ec_enc                      *psRangeEnc,                    /* I/O  Compressor data structure                   */
    opus_int8                   mid_only_flag
);

/* Decode mid/side predictors */
void silk_stereo_decode_pred(
    ec_dec                      *psRangeDec,                    /* I/O  Compressor data structure                   */
    opus_int32                  pred_Q13[]                      /* O    Predictors                                  */
);

/* Decode mid-only flag */
void silk_stereo_decode_mid_only(
    ec_dec                      *psRangeDec,                    /* I/O  Compressor data structure                   */
    opus_int                    *decode_only_mid                /* O    Flag that only mid channel has been coded   */
);

/* Encodes signs of excitation */
void silk_encode_signs(
    ec_enc                      *psRangeEnc,                        /* I/O  Compressor data structure                   */
    const opus_int8             pulses[],                           /* I    pulse signal                                */
    opus_int                    length,                             /* I    length of input                             */
    const opus_int              signalType,                         /* I    Signal type                                 */
    const opus_int              quantOffsetType,                    /* I    Quantization offset type                    */
    const opus_int              sum_pulses[ MAX_NB_SHELL_BLOCKS ]   /* I    Sum of absolute pulses per block            */
);

/* Decodes signs of excitation */
void silk_decode_signs(
    ec_dec                      *psRangeDec,                        /* I/O  Compressor data structure                   */
    opus_int16                  pulses[],                           /* I/O  pulse signal                                */
    opus_int                    length,                             /* I    length of input                             */
    const opus_int              signalType,                         /* I    Signal type                                 */
    const opus_int              quantOffsetType,                    /* I    Quantization offset type                    */
    const opus_int              sum_pulses[ MAX_NB_SHELL_BLOCKS ]   /* I    Sum of absolute pulses per block            */
);

/* Check encoder control struct */
opus_int check_control_input(
    silk_EncControlStruct        *encControl                    /* I    Control structure                           */
);

/* Control internal sampling rate */
opus_int silk_control_audio_bandwidth(
    silk_encoder_state          *psEncC,                        /* I/O  Pointer to Silk encoder state               */
    silk_EncControlStruct       *encControl                     /* I    Control structure                           */
);

/* Control SNR of redidual quantizer */
opus_int silk_control_SNR(
    silk_encoder_state          *psEncC,                        /* I/O  Pointer to Silk encoder state               */
    opus_int32                  TargetRate_bps                  /* I    Target max bitrate (bps)                    */
);

/***************/
/* Shell coder */
/***************/

/* Encode quantization indices of excitation */
void silk_encode_pulses(
    ec_enc                      *psRangeEnc,                    /* I/O  compressor data structure                   */
    const opus_int              signalType,                     /* I    Signal type                                 */
    const opus_int              quantOffsetType,                /* I    quantOffsetType                             */
    opus_int8                   pulses[],                       /* I    quantization indices                        */
    const opus_int              frame_length                    /* I    Frame length                                */
);

/* Shell encoder, operates on one shell code frame of 16 pulses */
void silk_shell_encoder(
    ec_enc                      *psRangeEnc,                    /* I/O  compressor data structure                   */
    const opus_int              *pulses0                        /* I    data: nonnegative pulse amplitudes          */
);

/* Shell decoder, operates on one shell code frame of 16 pulses */
void silk_shell_decoder(
    opus_int16                  *pulses0,                       /* O    data: nonnegative pulse amplitudes          */
    ec_dec                      *psRangeDec,                    /* I/O  Compressor data structure                   */
    const opus_int              pulses4                         /* I    number of pulses per pulse-subframe         */
);

/* Gain scalar quantization with hysteresis, uniform on log scale */
void silk_gains_quant(
    opus_int8                   ind[ MAX_NB_SUBFR ],            /* O    gain indices                                */
    opus_int32                  gain_Q16[ MAX_NB_SUBFR ],       /* I/O  gains (quantized out)                       */
    opus_int8                   *prev_ind,                      /* I/O  last index in previous frame                */
    const opus_int              conditional,                    /* I    first gain is delta coded if 1              */
    const opus_int              nb_subfr                        /* I    number of subframes                         */
);

/* Gains scalar dequantization, uniform on log scale */
void silk_gains_dequant(
    opus_int32                  gain_Q16[ MAX_NB_SUBFR ],       /* O    quantized gains                             */
    const opus_int8             ind[ MAX_NB_SUBFR ],            /* I    gain indices                                */
    opus_int8                   *prev_ind,                      /* I/O  last index in previous frame                */
    const opus_int              conditional,                    /* I    first gain is delta coded if 1              */
    const opus_int              nb_subfr                        /* I    number of subframes                          */
);

/* Compute unique identifier of gain indices vector */
opus_int32 silk_gains_ID(                                       /* O    returns unique identifier of gains          */
    const opus_int8             ind[ MAX_NB_SUBFR ],            /* I    gain indices                                */
    const opus_int              nb_subfr                        /* I    number of subframes                         */
);

/* Interpolate two vectors */
void silk_interpolate(
    opus_int16                  xi[ MAX_LPC_ORDER ],            /* O    interpolated vector                         */
    const opus_int16            x0[ MAX_LPC_ORDER ],            /* I    first vector                                */
    const opus_int16            x1[ MAX_LPC_ORDER ],            /* I    second vector                               */
    const opus_int              ifact_Q2,                       /* I    interp. factor, weight on 2nd vector        */
    const opus_int              d                               /* I    number of parameters                        */
);

/* LTP tap quantizer */
void silk_quant_LTP_gains(
    opus_int16                  B_Q14[ MAX_NB_SUBFR * LTP_ORDER ],          /* I/O  (un)quantized LTP gains         */
    opus_int8                   cbk_index[ MAX_NB_SUBFR ],                  /* O    Codebook Index                  */
    opus_int8                   *periodicity_index,                         /* O    Periodicity Index               */
    opus_int32                  *sum_gain_dB_Q7,                            /* I/O  Cumulative max prediction gain  */
    const opus_int32            W_Q18[ MAX_NB_SUBFR*LTP_ORDER*LTP_ORDER ],  /* I    Error Weights in Q18            */
    opus_int                    mu_Q9,                                      /* I    Mu value (R/D tradeoff)         */
    opus_int                    lowComplexity,                              /* I    Flag for low complexity         */
    const opus_int              nb_subfr,                                   /* I    number of subframes             */
    int                         arch                                        /* I    Run-time architecture           */
);

/* Entropy constrained matrix-weighted VQ, for a single input data vector */
void silk_VQ_WMat_EC_c(
    opus_int8                   *ind,                           /* O    index of best codebook vector               */
    opus_int32                  *rate_dist_Q14,                 /* O    best weighted quant error + mu * rate       */
    opus_int                    *gain_Q7,                       /* O    sum of absolute LTP coefficients            */
    const opus_int16            *in_Q14,                        /* I    input vector to be quantized                */
    const opus_int32            *W_Q18,                         /* I    weighting matrix                            */
    const opus_int8             *cb_Q7,                         /* I    codebook                                    */
    const opus_uint8            *cb_gain_Q7,                    /* I    codebook effective gain                     */
    const opus_uint8            *cl_Q5,                         /* I    code length for each codebook vector        */
    const opus_int              mu_Q9,                          /* I    tradeoff betw. weighted error and rate      */
    const opus_int32            max_gain_Q7,                    /* I    maximum sum of absolute LTP coefficients    */
    opus_int                    L                               /* I    number of vectors in codebook               */
);

#if !defined(OVERRIDE_silk_VQ_WMat_EC)
#define silk_VQ_WMat_EC(ind, rate_dist_Q14, gain_Q7, in_Q14, W_Q18, cb_Q7, cb_gain_Q7, cl_Q5, \
                          mu_Q9, max_gain_Q7, L, arch) \
    ((void)(arch),silk_VQ_WMat_EC_c(ind, rate_dist_Q14, gain_Q7, in_Q14, W_Q18, cb_Q7, cb_gain_Q7, cl_Q5, \
                          mu_Q9, max_gain_Q7, L))
#endif

/************************************/
/* Noise shaping quantization (NSQ) */
/************************************/

void silk_NSQ_c(
    const silk_encoder_state    *psEncC,                                    /* I/O  Encoder State                   */
    silk_nsq_state              *NSQ,                                       /* I/O  NSQ state                       */
    SideInfoIndices             *psIndices,                                 /* I/O  Quantization Indices            */
    const opus_int32            x_Q3[],                                     /* I    Prefiltered input signal        */
    opus_int8                   pulses[],                                   /* O    Quantized pulse signal          */
    const opus_int16            PredCoef_Q12[ 2 * MAX_LPC_ORDER ],          /* I    Short term prediction coefs     */
    const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],    /* I    Long term prediction coefs      */
    const opus_int16            AR2_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I Noise shaping coefs             */
    const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],          /* I    Long term shaping coefs         */
    const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                   /* I    Spectral tilt                   */
    const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                 /* I    Low frequency shaping coefs     */
    const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                  /* I    Quantization step sizes         */
    const opus_int              pitchL[ MAX_NB_SUBFR ],                     /* I    Pitch lags                      */
    const opus_int              Lambda_Q10,                                 /* I    Rate/distortion tradeoff        */
    const opus_int              LTP_scale_Q14                               /* I    LTP state scaling               */
);

#if !defined(OVERRIDE_silk_NSQ)
#define silk_NSQ(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                   HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((void)(arch),silk_NSQ_c(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                   HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))
#endif

/* Noise shaping using delayed decision */
void silk_NSQ_del_dec_c(
    const silk_encoder_state    *psEncC,                                    /* I/O  Encoder State                   */
    silk_nsq_state              *NSQ,                                       /* I/O  NSQ state                       */
    SideInfoIndices             *psIndices,                                 /* I/O  Quantization Indices            */
    const opus_int32            x_Q3[],                                     /* I    Prefiltered input signal        */
    opus_int8                   pulses[],                                   /* O    Quantized pulse signal          */
    const opus_int16            PredCoef_Q12[ 2 * MAX_LPC_ORDER ],          /* I    Short term prediction coefs     */
    const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],    /* I    Long term prediction coefs      */
    const opus_int16            AR2_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I Noise shaping coefs             */
    const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],          /* I    Long term shaping coefs         */
    const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                   /* I    Spectral tilt                   */
    const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                 /* I    Low frequency shaping coefs     */
    const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                  /* I    Quantization step sizes         */
    const opus_int              pitchL[ MAX_NB_SUBFR ],                     /* I    Pitch lags                      */
    const opus_int              Lambda_Q10,                                 /* I    Rate/distortion tradeoff        */
    const opus_int              LTP_scale_Q14                               /* I    LTP state scaling               */
);

#if !defined(OVERRIDE_silk_NSQ_del_dec)
#define silk_NSQ_del_dec(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                           HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((void)(arch),silk_NSQ_del_dec_c(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                           HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))
#endif

/************/
/* Silk VAD */
/************/
/* Initialize the Silk VAD */
opus_int silk_VAD_Init(                                         /* O    Return value, 0 if success                  */
    silk_VAD_state              *psSilk_VAD                     /* I/O  Pointer to Silk VAD state                   */
);

/* Get speech activity level in Q8 */
opus_int silk_VAD_GetSA_Q8_c(                                   /* O    Return value, 0 if success                  */
    silk_encoder_state          *psEncC,                        /* I/O  Encoder state                               */
    const opus_int16            pIn[]                           /* I    PCM input                                   */
);

#if !defined(OVERRIDE_silk_VAD_GetSA_Q8)
#define silk_VAD_GetSA_Q8(psEnC, pIn, arch) ((void)(arch),silk_VAD_GetSA_Q8_c(psEnC, pIn))
#endif

/* Low-pass filter with variable cutoff frequency based on  */
/* piece-wise linear interpolation between elliptic filters */
/* Start by setting transition_frame_no = 1;                */
void silk_LP_variable_cutoff(
    silk_LP_state               *psLP,                          /* I/O  LP filter state                             */
    opus_int16                  *frame,                         /* I/O  Low-pass filtered output signal             */
    const opus_int              frame_length                    /* I    Frame length                                */
);

/******************/
/* NLSF Quantizer */
/******************/
/* Limit, stabilize, convert and quantize NLSFs */
void silk_process_NLSFs(
    silk_encoder_state          *psEncC,                            /* I/O  Encoder state                               */
    opus_int16                  PredCoef_Q12[ 2 ][ MAX_LPC_ORDER ], /* O    Prediction coefficients                     */
    opus_int16                  pNLSF_Q15[         MAX_LPC_ORDER ], /* I/O  Normalized LSFs (quant out) (0 - (2^15-1))  */
    const opus_int16            prev_NLSFq_Q15[    MAX_LPC_ORDER ]  /* I    Previous Normalized LSFs (0 - (2^15-1))     */
);

opus_int32 silk_NLSF_encode(                                    /* O    Returns RD value in Q25                     */
          opus_int8             *NLSFIndices,                   /* I    Codebook path vector [ LPC_ORDER + 1 ]      */
          opus_int16            *pNLSF_Q15,                     /* I/O  Quantized NLSF vector [ LPC_ORDER ]         */
    const silk_NLSF_CB_struct   *psNLSF_CB,                     /* I    Codebook object                             */
    const opus_int16            *pW_QW,                         /* I    NLSF weight vector [ LPC_ORDER ]            */
    const opus_int              NLSF_mu_Q20,                    /* I    Rate weight for the RD optimization         */
    const opus_int              nSurvivors,                     /* I    Max survivors after first stage             */
    const opus_int              signalType                      /* I    Signal type: 0/1/2                          */
);

/* Compute quantization errors for an LPC_order element input vector for a VQ codebook */
void silk_NLSF_VQ(
    opus_int32                  err_Q26[],                      /* O    Quantization errors [K]                     */
    const opus_int16            in_Q15[],                       /* I    Input vectors to be quantized [LPC_order]   */
    const opus_uint8            pCB_Q8[],                       /* I    Codebook vectors [K*LPC_order]              */
    const opus_int              K,                              /* I    Number of codebook vectors                  */
    const opus_int              LPC_order                       /* I    Number of LPCs                              */
);

/* Delayed-decision quantizer for NLSF residuals */
opus_int32 silk_NLSF_del_dec_quant(                             /* O    Returns RD value in Q25                     */
    opus_int8                   indices[],                      /* O    Quantization indices [ order ]              */
    const opus_int16            x_Q10[],                        /* I    Input [ order ]                             */
    const opus_int16            w_Q5[],                         /* I    Weights [ order ]                           */
    const opus_uint8            pred_coef_Q8[],                 /* I    Backward predictor coefs [ order ]          */
    const opus_int16            ec_ix[],                        /* I    Indices to entropy coding tables [ order ]  */
    const opus_uint8            ec_rates_Q5[],                  /* I    Rates []                                    */
    const opus_int              quant_step_size_Q16,            /* I    Quantization step size                      */
    const opus_int16            inv_quant_step_size_Q6,         /* I    Inverse quantization step size              */
    const opus_int32            mu_Q20,                         /* I    R/D tradeoff                                */
    const opus_int16            order                           /* I    Number of input values                      */
);

/* Unpack predictor values and indices for entropy coding tables */
void silk_NLSF_unpack(
          opus_int16            ec_ix[],                        /* O    Indices to entropy tables [ LPC_ORDER ]     */
          opus_uint8            pred_Q8[],                      /* O    LSF predictor [ LPC_ORDER ]                 */
    const silk_NLSF_CB_struct   *psNLSF_CB,                     /* I    Codebook object                             */
    const opus_int              CB1_index                       /* I    Index of vector in first LSF codebook       */
);

/***********************/
/* NLSF vector decoder */
/***********************/
void silk_NLSF_decode(
          opus_int16            *pNLSF_Q15,                     /* O    Quantized NLSF vector [ LPC_ORDER ]         */
          opus_int8             *NLSFIndices,                   /* I    Codebook path vector [ LPC_ORDER + 1 ]      */
    const silk_NLSF_CB_struct   *psNLSF_CB                      /* I    Codebook object                             */
);

/****************************************************/
/* Decoder Functions                                */
/****************************************************/
opus_int silk_init_decoder(
    silk_decoder_state          *psDec                          /* I/O  Decoder state pointer                       */
);

/* Set decoder sampling rate */
opus_int silk_decoder_set_fs(
    silk_decoder_state          *psDec,                         /* I/O  Decoder state pointer                       */
    opus_int                    fs_kHz,                         /* I    Sampling frequency (kHz)                    */
    opus_int32                  fs_API_Hz                       /* I    API Sampling frequency (Hz)                 */
);

/****************/
/* Decode frame */
/****************/
opus_int silk_decode_frame(
    silk_decoder_state          *psDec,                         /* I/O  Pointer to Silk decoder state               */
    ec_dec                      *psRangeDec,                    /* I/O  Compressor data structure                   */
    opus_int16                  pOut[],                         /* O    Pointer to output speech frame              */
    opus_int32                  *pN,                            /* O    Pointer to size of output frame             */
    opus_int                    lostFlag,                       /* I    0: no loss, 1 loss, 2 decode fec            */
    opus_int                    condCoding,                     /* I    The type of conditional coding to use       */
    int                         arch                            /* I    Run-time architecture                       */
);

/* Decode indices from bitstream */
void silk_decode_indices(
    silk_decoder_state          *psDec,                         /* I/O  State                                       */
    ec_dec                      *psRangeDec,                    /* I/O  Compressor data structure                   */
    opus_int                    FrameIndex,                     /* I    Frame number                                */
    opus_int                    decode_LBRR,                    /* I    Flag indicating LBRR data is being decoded  */
    opus_int                    condCoding                      /* I    The type of conditional coding to use       */
);

/* Decode parameters from payload */
void silk_decode_parameters(
    silk_decoder_state          *psDec,                         /* I/O  State                                       */
    silk_decoder_control        *psDecCtrl,                     /* I/O  Decoder control                             */
    opus_int                    condCoding                      /* I    The type of conditional coding to use       */
);

/* Core decoder. Performs inverse NSQ operation LTP + LPC */
void silk_decode_core(
    silk_decoder_state          *psDec,                         /* I/O  Decoder state                               */
    silk_decoder_control        *psDecCtrl,                     /* I    Decoder control                             */
    opus_int16                  xq[],                           /* O    Decoded speech                              */
    const opus_int16            pulses[ MAX_FRAME_LENGTH ],     /* I    Pulse signal                                */
    int                         arch                            /* I    Run-time architecture                       */
);

/* Decode quantization indices of excitation (Shell coding) */
void silk_decode_pulses(
    ec_dec                      *psRangeDec,                    /* I/O  Compressor data structure                   */
    opus_int16                  pulses[],                       /* O    Excitation signal                           */
    const opus_int              signalType,                     /* I    Sigtype                                     */
    const opus_int              quantOffsetType,                /* I    quantOffsetType                             */
    const opus_int              frame_length                    /* I    Frame length                                */
);

/******************/
/* CNG */
/******************/

/* Reset CNG */
void silk_CNG_Reset(
    silk_decoder_state          *psDec                          /* I/O  Decoder state                               */
);

/* Updates CNG estimate, and applies the CNG when packet was lost */
void silk_CNG(
    silk_decoder_state          *psDec,                         /* I/O  Decoder state                               */
    silk_decoder_control        *psDecCtrl,                     /* I/O  Decoder control                             */
    opus_int16                  frame[],                        /* I/O  Signal                                      */
    opus_int                    length                          /* I    Length of residual                          */
);

/* Encoding of various parameters */
void silk_encode_indices(
    silk_encoder_state          *psEncC,                        /* I/O  Encoder state                               */
    ec_enc                      *psRangeEnc,                    /* I/O  Compressor data structure                   */
    opus_int                    FrameIndex,                     /* I    Frame number                                */
    opus_int                    encode_LBRR,                    /* I    Flag indicating LBRR data is being encoded  */
    opus_int                    condCoding                      /* I    The type of conditional coding to use       */
);

#endif
