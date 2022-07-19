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

#ifndef SILK_MAIN_FIX_H
#define SILK_MAIN_FIX_H

#include "SigProc_FIX.h"
#include "structs_FIX.h"
#include "control.h"
#include "main.h"
#include "PLC.h"
#include "debug.h"
#include "entenc.h"

#ifndef FORCE_CPP_BUILD
#ifdef __cplusplus
extern "C"
{
#endif
#endif

#define silk_encoder_state_Fxx      silk_encoder_state_FIX
#define silk_encode_do_VAD_Fxx      silk_encode_do_VAD_FIX
#define silk_encode_frame_Fxx       silk_encode_frame_FIX

/*********************/
/* Encoder Functions */
/*********************/

/* High-pass filter with cutoff frequency adaptation based on pitch lag statistics */
void silk_HP_variable_cutoff(
    silk_encoder_state_Fxx          state_Fxx[]                             /* I/O  Encoder states                                                              */
);

/* Encoder main function */
void silk_encode_do_VAD_FIX(
    silk_encoder_state_FIX          *psEnc                                  /* I/O  Pointer to Silk FIX encoder state                                           */
);

/* Encoder main function */
opus_int silk_encode_frame_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  Pointer to Silk FIX encoder state                                           */
    opus_int32                      *pnBytesOut,                            /* O    Pointer to number of payload bytes;                                         */
    ec_enc                          *psRangeEnc,                            /* I/O  compressor data structure                                                   */
    opus_int                        condCoding,                             /* I    The type of conditional coding to use                                       */
    opus_int                        maxBits,                                /* I    If > 0: maximum number of output bits                                       */
    opus_int                        useCBR                                  /* I    Flag to force constant-bitrate operation                                    */
);

/* Initializes the Silk encoder state */
opus_int silk_init_encoder(
    silk_encoder_state_Fxx          *psEnc,                                 /* I/O  Pointer to Silk FIX encoder state                                           */
    int                              arch                                   /* I    Run-time architecture                                                       */
);

/* Control the Silk encoder */
opus_int silk_control_encoder(
    silk_encoder_state_Fxx          *psEnc,                                 /* I/O  Pointer to Silk encoder state                                               */
    silk_EncControlStruct           *encControl,                            /* I    Control structure                                                           */
    const opus_int32                TargetRate_bps,                         /* I    Target max bitrate (bps)                                                    */
    const opus_int                  allow_bw_switch,                        /* I    Flag to allow switching audio bandwidth                                     */
    const opus_int                  channelNb,                              /* I    Channel number                                                              */
    const opus_int                  force_fs_kHz
);

/****************/
/* Prefiltering */
/****************/
void silk_prefilter_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  Encoder state                                                               */
    const silk_encoder_control_FIX  *psEncCtrl,                             /* I    Encoder control                                                             */
    opus_int32                      xw_Q10[],                               /* O    Weighted signal                                                             */
    const opus_int16                x[]                                     /* I    Speech signal                                                               */
);

void silk_warped_LPC_analysis_filter_FIX_c(
          opus_int32            state[],                    /* I/O  State [order + 1]                   */
          opus_int32            res_Q2[],                   /* O    Residual signal [length]            */
    const opus_int16            coef_Q13[],                 /* I    Coefficients [order]                */
    const opus_int16            input[],                    /* I    Input signal [length]               */
    const opus_int16            lambda_Q16,                 /* I    Warping factor                      */
    const opus_int              length,                     /* I    Length of input signal              */
    const opus_int              order                       /* I    Filter order (even)                 */
);


/**************************/
/* Noise shaping analysis */
/**************************/
/* Compute noise shaping coefficients and initial gain values */
void silk_noise_shape_analysis_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  Encoder state FIX                                                           */
    silk_encoder_control_FIX        *psEncCtrl,                             /* I/O  Encoder control FIX                                                         */
    const opus_int16                *pitch_res,                             /* I    LPC residual from pitch analysis                                            */
    const opus_int16                *x,                                     /* I    Input signal [ frame_length + la_shape ]                                    */
    int                              arch                                   /* I    Run-time architecture                                                       */
);

/* Autocorrelations for a warped frequency axis */
void silk_warped_autocorrelation_FIX(
          opus_int32                *corr,                                  /* O    Result [order + 1]                                                          */
          opus_int                  *scale,                                 /* O    Scaling of the correlation vector                                           */
    const opus_int16                *input,                                 /* I    Input data to correlate                                                     */
    const opus_int                  warping_Q16,                            /* I    Warping coefficient                                                         */
    const opus_int                  length,                                 /* I    Length of input                                                             */
    const opus_int                  order                                   /* I    Correlation order (even)                                                    */
);

/* Calculation of LTP state scaling */
void silk_LTP_scale_ctrl_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  encoder state                                                               */
    silk_encoder_control_FIX        *psEncCtrl,                             /* I/O  encoder control                                                             */
    opus_int                        condCoding                              /* I    The type of conditional coding to use                                       */
);

/**********************************************/
/* Prediction Analysis                        */
/**********************************************/
/* Find pitch lags */
void silk_find_pitch_lags_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  encoder state                                                               */
    silk_encoder_control_FIX        *psEncCtrl,                             /* I/O  encoder control                                                             */
    opus_int16                      res[],                                  /* O    residual                                                                    */
    const opus_int16                x[],                                    /* I    Speech signal                                                               */
    int                             arch                                    /* I    Run-time architecture                                                       */
);

/* Find LPC and LTP coefficients */
void silk_find_pred_coefs_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  encoder state                                                               */
    silk_encoder_control_FIX        *psEncCtrl,                             /* I/O  encoder control                                                             */
    const opus_int16                res_pitch[],                            /* I    Residual from pitch analysis                                                */
    const opus_int16                x[],                                    /* I    Speech signal                                                               */
    opus_int                        condCoding                              /* I    The type of conditional coding to use                                       */
);

/* LPC analysis */
void silk_find_LPC_FIX(
    silk_encoder_state              *psEncC,                                /* I/O  Encoder state                                                               */
    opus_int16                      NLSF_Q15[],                             /* O    NLSFs                                                                       */
    const opus_int16                x[],                                    /* I    Input signal                                                                */
    const opus_int32                minInvGain_Q30                          /* I    Inverse of max prediction gain                                              */
);

/* LTP analysis */
void silk_find_LTP_FIX(
    opus_int16                      b_Q14[ MAX_NB_SUBFR * LTP_ORDER ],      /* O    LTP coefs                                                                   */
    opus_int32                      WLTP[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ], /* O    Weight for LTP quantization                                           */
    opus_int                        *LTPredCodGain_Q7,                      /* O    LTP coding gain                                                             */
    const opus_int16                r_lpc[],                                /* I    residual signal after LPC signal + state for first 10 ms                    */
    const opus_int                  lag[ MAX_NB_SUBFR ],                    /* I    LTP lags                                                                    */
    const opus_int32                Wght_Q15[ MAX_NB_SUBFR ],               /* I    weights                                                                     */
    const opus_int                  subfr_length,                           /* I    subframe length                                                             */
    const opus_int                  nb_subfr,                               /* I    number of subframes                                                         */
    const opus_int                  mem_offset,                             /* I    number of samples in LTP memory                                             */
    opus_int                        corr_rshifts[ MAX_NB_SUBFR ],           /* O    right shifts applied to correlations                                        */
    int                             arch                                    /* I    Run-time architecture                                                       */
);

void silk_LTP_analysis_filter_FIX(
    opus_int16                      *LTP_res,                               /* O    LTP residual signal of length MAX_NB_SUBFR * ( pre_length + subfr_length )  */
    const opus_int16                *x,                                     /* I    Pointer to input signal with at least max( pitchL ) preceding samples       */
    const opus_int16                LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],/* I    LTP_ORDER LTP coefficients for each MAX_NB_SUBFR subframe                   */
    const opus_int                  pitchL[ MAX_NB_SUBFR ],                 /* I    Pitch lag, one for each subframe                                            */
    const opus_int32                invGains_Q16[ MAX_NB_SUBFR ],           /* I    Inverse quantization gains, one for each subframe                           */
    const opus_int                  subfr_length,                           /* I    Length of each subframe                                                     */
    const opus_int                  nb_subfr,                               /* I    Number of subframes                                                         */
    const opus_int                  pre_length                              /* I    Length of the preceding samples starting at &x[0] for each subframe         */
);

/* Calculates residual energies of input subframes where all subframes have LPC_order   */
/* of preceding samples                                                                 */
void silk_residual_energy_FIX(
          opus_int32                nrgs[ MAX_NB_SUBFR ],                   /* O    Residual energy per subframe                                                */
          opus_int                  nrgsQ[ MAX_NB_SUBFR ],                  /* O    Q value per subframe                                                        */
    const opus_int16                x[],                                    /* I    Input signal                                                                */
          opus_int16                a_Q12[ 2 ][ MAX_LPC_ORDER ],            /* I    AR coefs for each frame half                                                */
    const opus_int32                gains[ MAX_NB_SUBFR ],                  /* I    Quantization gains                                                          */
    const opus_int                  subfr_length,                           /* I    Subframe length                                                             */
    const opus_int                  nb_subfr,                               /* I    Number of subframes                                                         */
    const opus_int                  LPC_order,                              /* I    LPC order                                                                   */
    int                             arch                                    /* I    Run-time architecture                                                       */
);

/* Residual energy: nrg = wxx - 2 * wXx * c + c' * wXX * c */
opus_int32 silk_residual_energy16_covar_FIX(
    const opus_int16                *c,                                     /* I    Prediction vector                                                           */
    const opus_int32                *wXX,                                   /* I    Correlation matrix                                                          */
    const opus_int32                *wXx,                                   /* I    Correlation vector                                                          */
    opus_int32                      wxx,                                    /* I    Signal energy                                                               */
    opus_int                        D,                                      /* I    Dimension                                                                   */
    opus_int                        cQ                                      /* I    Q value for c vector 0 - 15                                                 */
);

/* Processing of gains */
void silk_process_gains_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  Encoder state                                                               */
    silk_encoder_control_FIX        *psEncCtrl,                             /* I/O  Encoder control                                                             */
    opus_int                        condCoding                              /* I    The type of conditional coding to use                                       */
);

/******************/
/* Linear Algebra */
/******************/
/* Calculates correlation matrix X'*X */
void silk_corrMatrix_FIX(
    const opus_int16                *x,                                     /* I    x vector [L + order - 1] used to form data matrix X                         */
    const opus_int                  L,                                      /* I    Length of vectors                                                           */
    const opus_int                  order,                                  /* I    Max lag for correlation                                                     */
    const opus_int                  head_room,                              /* I    Desired headroom                                                            */
    opus_int32                      *XX,                                    /* O    Pointer to X'*X correlation matrix [ order x order ]                        */
    opus_int                        *rshifts,                               /* I/O  Right shifts of correlations                                                */
    int                              arch                                   /* I    Run-time architecture                                                       */
);

/* Calculates correlation vector X'*t */
void silk_corrVector_FIX(
    const opus_int16                *x,                                     /* I    x vector [L + order - 1] used to form data matrix X                         */
    const opus_int16                *t,                                     /* I    Target vector [L]                                                           */
    const opus_int                  L,                                      /* I    Length of vectors                                                           */
    const opus_int                  order,                                  /* I    Max lag for correlation                                                     */
    opus_int32                      *Xt,                                    /* O    Pointer to X'*t correlation vector [order]                                  */
    const opus_int                  rshifts,                                /* I    Right shifts of correlations                                                */
    int                             arch                                    /* I    Run-time architecture                                                       */
);

/* Add noise to matrix diagonal */
void silk_regularize_correlations_FIX(
    opus_int32                      *XX,                                    /* I/O  Correlation matrices                                                        */
    opus_int32                      *xx,                                    /* I/O  Correlation values                                                          */
    opus_int32                      noise,                                  /* I    Noise to add                                                                */
    opus_int                        D                                       /* I    Dimension of XX                                                             */
);

/* Solves Ax = b, assuming A is symmetric */
void silk_solve_LDL_FIX(
    opus_int32                      *A,                                     /* I    Pointer to symetric square matrix A                                         */
    opus_int                        M,                                      /* I    Size of matrix                                                              */
    const opus_int32                *b,                                     /* I    Pointer to b vector                                                         */
    opus_int32                      *x_Q16                                  /* O    Pointer to x solution vector                                                */
);

#ifndef FORCE_CPP_BUILD
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* FORCE_CPP_BUILD */
#endif /* SILK_MAIN_FIX_H */
