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

#ifndef SILK_MAIN_FLP_H
#define SILK_MAIN_FLP_H

#include "SigProc_FLP.h"
#include "SigProc_FIX.h"
#include "structs_FLP.h"
#include "main.h"
#include "define.h"
#include "debug.h"
#include "entenc.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define silk_encoder_state_Fxx      silk_encoder_state_FLP
#define silk_encode_do_VAD_Fxx      silk_encode_do_VAD_FLP
#define silk_encode_frame_Fxx       silk_encode_frame_FLP

/*********************/
/* Encoder Functions */
/*********************/

/* High-pass filter with cutoff frequency adaptation based on pitch lag statistics */
void silk_HP_variable_cutoff(
    silk_encoder_state_Fxx          state_Fxx[]                         /* I/O  Encoder states                              */
);

/* Encoder main function */
void silk_encode_do_VAD_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    opus_int                        activity                            /* I    Decision of Opus voice activity detector    */
);

/* Encoder main function */
opus_int silk_encode_frame_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    opus_int32                      *pnBytesOut,                        /* O    Number of payload bytes;                    */
    ec_enc                          *psRangeEnc,                        /* I/O  compressor data structure                   */
    opus_int                        condCoding,                         /* I    The type of conditional coding to use       */
    opus_int                        maxBits,                            /* I    If > 0: maximum number of output bits       */
    opus_int                        useCBR                              /* I    Flag to force constant-bitrate operation    */
);

/* Initializes the Silk encoder state */
opus_int silk_init_encoder(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    int                              arch                               /* I    Run-tim architecture                        */
);

/* Control the Silk encoder */
opus_int silk_control_encoder(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Pointer to Silk encoder state FLP           */
    silk_EncControlStruct           *encControl,                        /* I    Control structure                           */
    const opus_int                  allow_bw_switch,                    /* I    Flag to allow switching audio bandwidth     */
    const opus_int                  channelNb,                          /* I    Channel number                              */
    const opus_int                  force_fs_kHz
);

/**************************/
/* Noise shaping analysis */
/**************************/
/* Compute noise shaping coefficients and initial gain values */
void silk_noise_shape_analysis_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,                         /* I/O  Encoder control FLP                         */
    const silk_float                *pitch_res,                         /* I    LPC residual from pitch analysis            */
    const silk_float                *x                                  /* I    Input signal [frame_length + la_shape]      */
);

/* Autocorrelations for a warped frequency axis */
void silk_warped_autocorrelation_FLP(
    silk_float                      *corr,                              /* O    Result [order + 1]                          */
    const silk_float                *input,                             /* I    Input data to correlate                     */
    const silk_float                warping,                            /* I    Warping coefficient                         */
    const opus_int                  length,                             /* I    Length of input                             */
    const opus_int                  order                               /* I    Correlation order (even)                    */
);

/* Calculation of LTP state scaling */
void silk_LTP_scale_ctrl_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,                         /* I/O  Encoder control FLP                         */
    opus_int                        condCoding                          /* I    The type of conditional coding to use       */
);

/**********************************************/
/* Prediction Analysis                        */
/**********************************************/
/* Find pitch lags */
void silk_find_pitch_lags_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,                         /* I/O  Encoder control FLP                         */
    silk_float                      res[],                              /* O    Residual                                    */
    const silk_float                x[],                                /* I    Speech signal                               */
    int                             arch                                /* I    Run-time architecture                       */
);

/* Find LPC and LTP coefficients */
void silk_find_pred_coefs_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,                         /* I/O  Encoder control FLP                         */
    const silk_float                res_pitch[],                        /* I    Residual from pitch analysis                */
    const silk_float                x[],                                /* I    Speech signal                               */
    opus_int                        condCoding                          /* I    The type of conditional coding to use       */
);

/* LPC analysis */
void silk_find_LPC_FLP(
    silk_encoder_state              *psEncC,                            /* I/O  Encoder state                               */
    opus_int16                      NLSF_Q15[],                         /* O    NLSFs                                       */
    const silk_float                x[],                                /* I    Input signal                                */
    const silk_float                minInvGain                          /* I    Prediction gain from LTP (dB)               */
);

/* LTP analysis */
void silk_find_LTP_FLP(
    silk_float                      XX[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ], /* O    Weight for LTP quantization         */
    silk_float                      xX[ MAX_NB_SUBFR * LTP_ORDER ],     /* O    Weight for LTP quantization                 */
    const silk_float                r_ptr[],                            /* I    LPC residual                                */
    const opus_int                  lag[  MAX_NB_SUBFR ],               /* I    LTP lags                                    */
    const opus_int                  subfr_length,                       /* I    Subframe length                             */
    const opus_int                  nb_subfr                            /* I    number of subframes                         */
);

void silk_LTP_analysis_filter_FLP(
    silk_float                      *LTP_res,                           /* O    LTP res MAX_NB_SUBFR*(pre_lgth+subfr_lngth) */
    const silk_float                *x,                                 /* I    Input signal, with preceding samples        */
    const silk_float                B[ LTP_ORDER * MAX_NB_SUBFR ],      /* I    LTP coefficients for each subframe          */
    const opus_int                  pitchL[   MAX_NB_SUBFR ],           /* I    Pitch lags                                  */
    const silk_float                invGains[ MAX_NB_SUBFR ],           /* I    Inverse quantization gains                  */
    const opus_int                  subfr_length,                       /* I    Length of each subframe                     */
    const opus_int                  nb_subfr,                           /* I    number of subframes                         */
    const opus_int                  pre_length                          /* I    Preceding samples for each subframe         */
);

/* Calculates residual energies of input subframes where all subframes have LPC_order   */
/* of preceding samples                                                                 */
void silk_residual_energy_FLP(
    silk_float                      nrgs[ MAX_NB_SUBFR ],               /* O    Residual energy per subframe                */
    const silk_float                x[],                                /* I    Input signal                                */
    silk_float                      a[ 2 ][ MAX_LPC_ORDER ],            /* I    AR coefs for each frame half                */
    const silk_float                gains[],                            /* I    Quantization gains                          */
    const opus_int                  subfr_length,                       /* I    Subframe length                             */
    const opus_int                  nb_subfr,                           /* I    number of subframes                         */
    const opus_int                  LPC_order                           /* I    LPC order                                   */
);

/* 16th order LPC analysis filter */
void silk_LPC_analysis_filter_FLP(
    silk_float                      r_LPC[],                            /* O    LPC residual signal                         */
    const silk_float                PredCoef[],                         /* I    LPC coefficients                            */
    const silk_float                s[],                                /* I    Input signal                                */
    const opus_int                  length,                             /* I    Length of input signal                      */
    const opus_int                  Order                               /* I    LPC order                                   */
);

/* LTP tap quantizer */
void silk_quant_LTP_gains_FLP(
    silk_float                      B[ MAX_NB_SUBFR * LTP_ORDER ],      /* O    Quantized LTP gains                         */
    opus_int8                       cbk_index[ MAX_NB_SUBFR ],          /* O    Codebook index                              */
    opus_int8                       *periodicity_index,                 /* O    Periodicity index                           */
    opus_int32                      *sum_log_gain_Q7,                   /* I/O  Cumulative max prediction gain  */
    silk_float                      *pred_gain_dB,                      /* O    LTP prediction gain                         */
    const silk_float                XX[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ], /* I    Correlation matrix                  */
    const silk_float                xX[ MAX_NB_SUBFR * LTP_ORDER ],     /* I    Correlation vector                          */
    const opus_int                  subfr_len,                          /* I    Number of samples per subframe              */
    const opus_int                  nb_subfr,                           /* I    Number of subframes                         */
    int                             arch                                /* I    Run-time architecture                       */
);

/* Residual energy: nrg = wxx - 2 * wXx * c + c' * wXX * c */
silk_float silk_residual_energy_covar_FLP(                              /* O    Weighted residual energy                    */
    const silk_float                *c,                                 /* I    Filter coefficients                         */
    silk_float                      *wXX,                               /* I/O  Weighted correlation matrix, reg. out       */
    const silk_float                *wXx,                               /* I    Weighted correlation vector                 */
    const silk_float                wxx,                                /* I    Weighted correlation value                  */
    const opus_int                  D                                   /* I    Dimension                                   */
);

/* Processing of gains */
void silk_process_gains_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,                         /* I/O  Encoder control FLP                         */
    opus_int                        condCoding                          /* I    The type of conditional coding to use       */
);

/******************/
/* Linear Algebra */
/******************/
/* Calculates correlation matrix X'*X */
void silk_corrMatrix_FLP(
    const silk_float                *x,                                 /* I    x vector [ L+order-1 ] used to create X     */
    const opus_int                  L,                                  /* I    Length of vectors                           */
    const opus_int                  Order,                              /* I    Max lag for correlation                     */
    silk_float                      *XX                                 /* O    X'*X correlation matrix [order x order]     */
);

/* Calculates correlation vector X'*t */
void silk_corrVector_FLP(
    const silk_float                *x,                                 /* I    x vector [L+order-1] used to create X       */
    const silk_float                *t,                                 /* I    Target vector [L]                           */
    const opus_int                  L,                                  /* I    Length of vecors                            */
    const opus_int                  Order,                              /* I    Max lag for correlation                     */
    silk_float                      *Xt                                 /* O    X'*t correlation vector [order]             */
);

/* Apply sine window to signal vector.  */
/* Window types:                        */
/*  1 -> sine window from 0 to pi/2     */
/*  2 -> sine window from pi/2 to pi    */
void silk_apply_sine_window_FLP(
    silk_float                      px_win[],                           /* O    Pointer to windowed signal                  */
    const silk_float                px[],                               /* I    Pointer to input signal                     */
    const opus_int                  win_type,                           /* I    Selects a window type                       */
    const opus_int                  length                              /* I    Window length, multiple of 4                */
);

/* Wrapper functions. Call flp / fix code */

/* Convert AR filter coefficients to NLSF parameters */
void silk_A2NLSF_FLP(
    opus_int16                      *NLSF_Q15,                          /* O    NLSF vector      [ LPC_order ]              */
    const silk_float                *pAR,                               /* I    LPC coefficients [ LPC_order ]              */
    const opus_int                  LPC_order                           /* I    LPC order                                   */
);

/* Convert NLSF parameters to AR prediction filter coefficients */
void silk_NLSF2A_FLP(
    silk_float                      *pAR,                               /* O    LPC coefficients [ LPC_order ]              */
    const opus_int16                *NLSF_Q15,                          /* I    NLSF vector      [ LPC_order ]              */
    const opus_int                  LPC_order,                          /* I    LPC order                                   */
    int                             arch                                /* I    Run-time architecture                       */
);

/* Limit, stabilize, and quantize NLSFs */
void silk_process_NLSFs_FLP(
    silk_encoder_state              *psEncC,                            /* I/O  Encoder state                               */
    silk_float                      PredCoef[ 2 ][ MAX_LPC_ORDER ],     /* O    Prediction coefficients                     */
    opus_int16                      NLSF_Q15[      MAX_LPC_ORDER ],     /* I/O  Normalized LSFs (quant out) (0 - (2^15-1))  */
    const opus_int16                prev_NLSF_Q15[ MAX_LPC_ORDER ]      /* I    Previous Normalized LSFs (0 - (2^15-1))     */
);

/* Floating-point Silk NSQ wrapper      */
void silk_NSQ_wrapper_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,                         /* I/O  Encoder control FLP                         */
    SideInfoIndices                 *psIndices,                         /* I/O  Quantization indices                        */
    silk_nsq_state                  *psNSQ,                             /* I/O  Noise Shaping Quantzation state             */
    opus_int8                       pulses[],                           /* O    Quantized pulse signal                      */
    const silk_float                x[]                                 /* I    Prefiltered input signal                    */
);

#ifdef __cplusplus
}
#endif

#endif
