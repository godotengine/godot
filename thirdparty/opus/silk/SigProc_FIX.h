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

#ifndef SILK_SIGPROC_FIX_H
#define SILK_SIGPROC_FIX_H

#ifdef  __cplusplus
extern "C"
{
#endif

/*#define silk_MACRO_COUNT */          /* Used to enable WMOPS counting */

#define SILK_MAX_ORDER_LPC            24            /* max order of the LPC analysis in schur() and k2a() */

#include <string.h>                                 /* for memset(), memcpy(), memmove() */
#include "typedef.h"
#include "resampler_structs.h"
#include "macros.h"
#include "cpu_support.h"

#if defined(OPUS_X86_MAY_HAVE_SSE4_1)
#include "x86/SigProc_FIX_sse.h"
#endif

#if (defined(OPUS_ARM_ASM) || defined(OPUS_ARM_MAY_HAVE_NEON_INTR))
#include "arm/biquad_alt_arm.h"
#include "arm/LPC_inv_pred_gain_arm.h"
#endif

/********************************************************************/
/*                    SIGNAL PROCESSING FUNCTIONS                   */
/********************************************************************/

/*!
 * Initialize/reset the resampler state for a given pair of input/output sampling rates
*/
opus_int silk_resampler_init(
    silk_resampler_state_struct *S,                 /* I/O  Resampler state                                             */
    opus_int32                  Fs_Hz_in,           /* I    Input sampling rate (Hz)                                    */
    opus_int32                  Fs_Hz_out,          /* I    Output sampling rate (Hz)                                   */
    opus_int                    forEnc              /* I    If 1: encoder; if 0: decoder                                */
);

/*!
 * Resampler: convert from one sampling rate to another
 */
opus_int silk_resampler(
    silk_resampler_state_struct *S,                 /* I/O  Resampler state                                             */
    opus_int16                  out[],              /* O    Output signal                                               */
    const opus_int16            in[],               /* I    Input signal                                                */
    opus_int32                  inLen               /* I    Number of input samples                                     */
);

/*!
* Downsample 2x, mediocre quality
*/
void silk_resampler_down2(
    opus_int32                  *S,                 /* I/O  State vector [ 2 ]                                          */
    opus_int16                  *out,               /* O    Output signal [ len ]                                       */
    const opus_int16            *in,                /* I    Input signal [ floor(len/2) ]                               */
    opus_int32                  inLen               /* I    Number of input samples                                     */
);

/*!
 * Downsample by a factor 2/3, low quality
*/
void silk_resampler_down2_3(
    opus_int32                  *S,                 /* I/O  State vector [ 6 ]                                          */
    opus_int16                  *out,               /* O    Output signal [ floor(2*inLen/3) ]                          */
    const opus_int16            *in,                /* I    Input signal [ inLen ]                                      */
    opus_int32                  inLen               /* I    Number of input samples                                     */
);

/*!
 * second order ARMA filter;
 * slower than biquad() but uses more precise coefficients
 * can handle (slowly) varying coefficients
 */
void silk_biquad_alt_stride1(
    const opus_int16            *in,                /* I     input signal                                               */
    const opus_int32            *B_Q28,             /* I     MA coefficients [3]                                        */
    const opus_int32            *A_Q28,             /* I     AR coefficients [2]                                        */
    opus_int32                  *S,                 /* I/O   State vector [2]                                           */
    opus_int16                  *out,               /* O     output signal                                              */
    const opus_int32            len                 /* I     signal length (must be even)                               */
);

void silk_biquad_alt_stride2_c(
    const opus_int16            *in,                /* I     input signal                                               */
    const opus_int32            *B_Q28,             /* I     MA coefficients [3]                                        */
    const opus_int32            *A_Q28,             /* I     AR coefficients [2]                                        */
    opus_int32                  *S,                 /* I/O   State vector [4]                                           */
    opus_int16                  *out,               /* O     output signal                                              */
    const opus_int32            len                 /* I     signal length (must be even)                               */
);

/* Variable order MA prediction error filter. */
void silk_LPC_analysis_filter(
    opus_int16                  *out,               /* O    Output signal                                               */
    const opus_int16            *in,                /* I    Input signal                                                */
    const opus_int16            *B,                 /* I    MA prediction coefficients, Q12 [order]                     */
    const opus_int32            len,                /* I    Signal length                                               */
    const opus_int32            d,                  /* I    Filter order                                                */
    int                         arch                /* I    Run-time architecture                                       */
);

/* Chirp (bandwidth expand) LP AR filter */
void silk_bwexpander(
    opus_int16                  *ar,                /* I/O  AR filter to be expanded (without leading 1)                */
    const opus_int              d,                  /* I    Length of ar                                                */
    opus_int32                  chirp_Q16           /* I    Chirp factor (typically in the range 0 to 1)                */
);

/* Chirp (bandwidth expand) LP AR filter */
void silk_bwexpander_32(
    opus_int32                  *ar,                /* I/O  AR filter to be expanded (without leading 1)                */
    const opus_int              d,                  /* I    Length of ar                                                */
    opus_int32                  chirp_Q16           /* I    Chirp factor in Q16                                         */
);

/* Compute inverse of LPC prediction gain, and                           */
/* test if LPC coefficients are stable (all poles within unit circle)    */
opus_int32 silk_LPC_inverse_pred_gain_c(            /* O   Returns inverse prediction gain in energy domain, Q30        */
    const opus_int16            *A_Q12,             /* I   Prediction coefficients, Q12 [order]                         */
    const opus_int              order               /* I   Prediction order                                             */
);

/* Split signal in two decimated bands using first-order allpass filters */
void silk_ana_filt_bank_1(
    const opus_int16            *in,                /* I    Input signal [N]                                            */
    opus_int32                  *S,                 /* I/O  State vector [2]                                            */
    opus_int16                  *outL,              /* O    Low band [N/2]                                              */
    opus_int16                  *outH,              /* O    High band [N/2]                                             */
    const opus_int32            N                   /* I    Number of input samples                                     */
);

#if !defined(OVERRIDE_silk_biquad_alt_stride2)
#define silk_biquad_alt_stride2(in, B_Q28, A_Q28, S, out, len, arch) ((void)(arch), silk_biquad_alt_stride2_c(in, B_Q28, A_Q28, S, out, len))
#endif

#if !defined(OVERRIDE_silk_LPC_inverse_pred_gain)
#define silk_LPC_inverse_pred_gain(A_Q12, order, arch)     ((void)(arch), silk_LPC_inverse_pred_gain_c(A_Q12, order))
#endif

/********************************************************************/
/*                        SCALAR FUNCTIONS                          */
/********************************************************************/

/* Approximation of 128 * log2() (exact inverse of approx 2^() below) */
/* Convert input to a log scale    */
opus_int32 silk_lin2log(
    const opus_int32            inLin               /* I  input in linear scale                                         */
);

/* Approximation of a sigmoid function */
opus_int silk_sigm_Q15(
    opus_int                    in_Q5               /* I                                                                */
);

/* Approximation of 2^() (exact inverse of approx log2() above) */
/* Convert input to a linear scale */
opus_int32 silk_log2lin(
    const opus_int32            inLog_Q7            /* I  input on log scale                                            */
);

/* Compute number of bits to right shift the sum of squares of a vector    */
/* of int16s to make it fit in an int32                                    */
void silk_sum_sqr_shift(
    opus_int32                  *energy,            /* O   Energy of x, after shifting to the right                     */
    opus_int                    *shift,             /* O   Number of bits right shift applied to energy                 */
    const opus_int16            *x,                 /* I   Input vector                                                 */
    opus_int                    len                 /* I   Length of input vector                                       */
);

/* Calculates the reflection coefficients from the correlation sequence    */
/* Faster than schur64(), but much less accurate.                          */
/* uses SMLAWB(), requiring armv5E and higher.                             */
opus_int32 silk_schur(                              /* O    Returns residual energy                                     */
    opus_int16                  *rc_Q15,            /* O    reflection coefficients [order] Q15                         */
    const opus_int32            *c,                 /* I    correlations [order+1]                                      */
    const opus_int32            order               /* I    prediction order                                            */
);

/* Calculates the reflection coefficients from the correlation sequence    */
/* Slower than schur(), but more accurate.                                 */
/* Uses SMULL(), available on armv4                                        */
opus_int32 silk_schur64(                            /* O    returns residual energy                                     */
    opus_int32                  rc_Q16[],           /* O    Reflection coefficients [order] Q16                         */
    const opus_int32            c[],                /* I    Correlations [order+1]                                      */
    opus_int32                  order               /* I    Prediction order                                            */
);

/* Step up function, converts reflection coefficients to prediction coefficients */
void silk_k2a(
    opus_int32                  *A_Q24,             /* O    Prediction coefficients [order] Q24                         */
    const opus_int16            *rc_Q15,            /* I    Reflection coefficients [order] Q15                         */
    const opus_int32            order               /* I    Prediction order                                            */
);

/* Step up function, converts reflection coefficients to prediction coefficients */
void silk_k2a_Q16(
    opus_int32                  *A_Q24,             /* O    Prediction coefficients [order] Q24                         */
    const opus_int32            *rc_Q16,            /* I    Reflection coefficients [order] Q16                         */
    const opus_int32            order               /* I    Prediction order                                            */
);

/* Apply sine window to signal vector.                              */
/* Window types:                                                    */
/*    1 -> sine window from 0 to pi/2                               */
/*    2 -> sine window from pi/2 to pi                              */
/* every other sample of window is linearly interpolated, for speed */
void silk_apply_sine_window(
    opus_int16                  px_win[],           /* O    Pointer to windowed signal                                  */
    const opus_int16            px[],               /* I    Pointer to input signal                                     */
    const opus_int              win_type,           /* I    Selects a window type                                       */
    const opus_int              length              /* I    Window length, multiple of 4                                */
);

/* Compute autocorrelation */
void silk_autocorr(
    opus_int32                  *results,           /* O    Result (length correlationCount)                            */
    opus_int                    *scale,             /* O    Scaling of the correlation vector                           */
    const opus_int16            *inputData,         /* I    Input data to correlate                                     */
    const opus_int              inputDataSize,      /* I    Length of input                                             */
    const opus_int              correlationCount,   /* I    Number of correlation taps to compute                       */
    int                         arch                /* I    Run-time architecture                                       */
);

void silk_decode_pitch(
    opus_int16                  lagIndex,           /* I                                                                */
    opus_int8                   contourIndex,       /* O                                                                */
    opus_int                    pitch_lags[],       /* O    4 pitch values                                              */
    const opus_int              Fs_kHz,             /* I    sampling frequency (kHz)                                    */
    const opus_int              nb_subfr            /* I    number of sub frames                                        */
);

opus_int silk_pitch_analysis_core(                  /* O    Voicing estimate: 0 voiced, 1 unvoiced                      */
    const opus_int16            *frame,             /* I    Signal of length PE_FRAME_LENGTH_MS*Fs_kHz                  */
    opus_int                    *pitch_out,         /* O    4 pitch lag values                                          */
    opus_int16                  *lagIndex,          /* O    Lag Index                                                   */
    opus_int8                   *contourIndex,      /* O    Pitch contour Index                                         */
    opus_int                    *LTPCorr_Q15,       /* I/O  Normalized correlation; input: value from previous frame    */
    opus_int                    prevLag,            /* I    Last lag of previous frame; set to zero is unvoiced         */
    const opus_int32            search_thres1_Q16,  /* I    First stage threshold for lag candidates 0 - 1              */
    const opus_int              search_thres2_Q13,  /* I    Final threshold for lag candidates 0 - 1                    */
    const opus_int              Fs_kHz,             /* I    Sample frequency (kHz)                                      */
    const opus_int              complexity,         /* I    Complexity setting, 0-2, where 2 is highest                 */
    const opus_int              nb_subfr,           /* I    number of 5 ms subframes                                    */
    int                         arch                /* I    Run-time architecture                                       */
);

/* Compute Normalized Line Spectral Frequencies (NLSFs) from whitening filter coefficients      */
/* If not all roots are found, the a_Q16 coefficients are bandwidth expanded until convergence. */
void silk_A2NLSF(
    opus_int16                  *NLSF,              /* O    Normalized Line Spectral Frequencies in Q15 (0..2^15-1) [d] */
    opus_int32                  *a_Q16,             /* I/O  Monic whitening filter coefficients in Q16 [d]              */
    const opus_int              d                   /* I    Filter order (must be even)                                 */
);

/* compute whitening filter coefficients from normalized line spectral frequencies */
void silk_NLSF2A(
    opus_int16                  *a_Q12,             /* O    monic whitening filter coefficients in Q12,  [ d ]          */
    const opus_int16            *NLSF,              /* I    normalized line spectral frequencies in Q15, [ d ]          */
    const opus_int              d,                  /* I    filter order (should be even)                               */
    int                         arch                /* I    Run-time architecture                                       */
);

/* Convert int32 coefficients to int16 coefs and make sure there's no wrap-around */
void silk_LPC_fit(
    opus_int16                  *a_QOUT,            /* O    Output signal                                               */
    opus_int32                  *a_QIN,             /* I/O  Input signal                                                */
    const opus_int              QOUT,               /* I    Input Q domain                                              */
    const opus_int              QIN,                /* I    Input Q domain                                              */
    const opus_int              d                   /* I    Filter order                                                */
);

void silk_insertion_sort_increasing(
    opus_int32                  *a,                 /* I/O   Unsorted / Sorted vector                                   */
    opus_int                    *idx,               /* O     Index vector for the sorted elements                       */
    const opus_int              L,                  /* I     Vector length                                              */
    const opus_int              K                   /* I     Number of correctly sorted positions                       */
);

void silk_insertion_sort_decreasing_int16(
    opus_int16                  *a,                 /* I/O   Unsorted / Sorted vector                                   */
    opus_int                    *idx,               /* O     Index vector for the sorted elements                       */
    const opus_int              L,                  /* I     Vector length                                              */
    const opus_int              K                   /* I     Number of correctly sorted positions                       */
);

void silk_insertion_sort_increasing_all_values_int16(
     opus_int16                 *a,                 /* I/O   Unsorted / Sorted vector                                   */
     const opus_int             L                   /* I     Vector length                                              */
);

/* NLSF stabilizer, for a single input data vector */
void silk_NLSF_stabilize(
          opus_int16            *NLSF_Q15,          /* I/O   Unstable/stabilized normalized LSF vector in Q15 [L]       */
    const opus_int16            *NDeltaMin_Q15,     /* I     Min distance vector, NDeltaMin_Q15[L] must be >= 1 [L+1]   */
    const opus_int              L                   /* I     Number of NLSF parameters in the input vector              */
);

/* Laroia low complexity NLSF weights */
void silk_NLSF_VQ_weights_laroia(
    opus_int16                  *pNLSFW_Q_OUT,      /* O     Pointer to input vector weights [D]                        */
    const opus_int16            *pNLSF_Q15,         /* I     Pointer to input vector         [D]                        */
    const opus_int              D                   /* I     Input vector dimension (even)                              */
);

/* Compute reflection coefficients from input signal */
void silk_burg_modified_c(
    opus_int32                  *res_nrg,           /* O    Residual energy                                             */
    opus_int                    *res_nrg_Q,         /* O    Residual energy Q value                                     */
    opus_int32                  A_Q16[],            /* O    Prediction coefficients (length order)                      */
    const opus_int16            x[],                /* I    Input signal, length: nb_subfr * ( D + subfr_length )       */
    const opus_int32            minInvGain_Q30,     /* I    Inverse of max prediction gain                              */
    const opus_int              subfr_length,       /* I    Input signal subframe length (incl. D preceding samples)    */
    const opus_int              nb_subfr,           /* I    Number of subframes stacked in x                            */
    const opus_int              D,                  /* I    Order                                                       */
    int                         arch                /* I    Run-time architecture                                       */
);

/* Copy and multiply a vector by a constant */
void silk_scale_copy_vector16(
    opus_int16                  *data_out,
    const opus_int16            *data_in,
    opus_int32                  gain_Q16,           /* I    Gain in Q16                                                 */
    const opus_int              dataSize            /* I    Length                                                      */
);

/* Some for the LTP related function requires Q26 to work.*/
void silk_scale_vector32_Q26_lshift_18(
    opus_int32                  *data1,             /* I/O  Q0/Q18                                                      */
    opus_int32                  gain_Q26,           /* I    Q26                                                         */
    opus_int                    dataSize            /* I    length                                                      */
);

/********************************************************************/
/*                        INLINE ARM MATH                           */
/********************************************************************/

/*    return sum( inVec1[i] * inVec2[i] ) */

opus_int32 silk_inner_prod_aligned(
    const opus_int16 *const     inVec1,             /*    I input vector 1                                              */
    const opus_int16 *const     inVec2,             /*    I input vector 2                                              */
    const opus_int              len,                /*    I vector lengths                                              */
    int                         arch                /*    I Run-time architecture                                       */
);


opus_int32 silk_inner_prod_aligned_scale(
    const opus_int16 *const     inVec1,             /*    I input vector 1                                              */
    const opus_int16 *const     inVec2,             /*    I input vector 2                                              */
    const opus_int              scale,              /*    I number of bits to shift                                     */
    const opus_int              len                 /*    I vector lengths                                              */
);

opus_int64 silk_inner_prod16_aligned_64_c(
    const opus_int16            *inVec1,            /*    I input vector 1                                              */
    const opus_int16            *inVec2,            /*    I input vector 2                                              */
    const opus_int              len                 /*    I vector lengths                                              */
);

/********************************************************************/
/*                                MACROS                            */
/********************************************************************/

/* Rotate a32 right by 'rot' bits. Negative rot values result in rotating
   left. Output is 32bit int.
   Note: contemporary compilers recognize the C expression below and
   compile it into a 'ror' instruction if available. No need for OPUS_INLINE ASM! */
static OPUS_INLINE opus_int32 silk_ROR32( opus_int32 a32, opus_int rot )
{
    opus_uint32 x = (opus_uint32) a32;
    opus_uint32 r = (opus_uint32) rot;
    opus_uint32 m = (opus_uint32) -rot;
    if( rot == 0 ) {
        return a32;
    } else if( rot < 0 ) {
        return (opus_int32) ((x << m) | (x >> (32 - m)));
    } else {
        return (opus_int32) ((x << (32 - r)) | (x >> r));
    }
}

/* Allocate opus_int16 aligned to 4-byte memory address */
#if EMBEDDED_ARM
#define silk_DWORD_ALIGN __attribute__((aligned(4)))
#else
#define silk_DWORD_ALIGN
#endif

/* Useful Macros that can be adjusted to other platforms */
#define silk_memcpy(dest, src, size)        memcpy((dest), (src), (size))
#define silk_memset(dest, src, size)        memset((dest), (src), (size))
#define silk_memmove(dest, src, size)       memmove((dest), (src), (size))

/* Fixed point macros */

/* (a32 * b32) output have to be 32bit int */
#define silk_MUL(a32, b32)                  ((a32) * (b32))

/* (a32 * b32) output have to be 32bit uint */
#define silk_MUL_uint(a32, b32)             silk_MUL(a32, b32)

/* a32 + (b32 * c32) output have to be 32bit int */
#define silk_MLA(a32, b32, c32)             silk_ADD32((a32),((b32) * (c32)))

/* a32 + (b32 * c32) output have to be 32bit uint */
#define silk_MLA_uint(a32, b32, c32)        silk_MLA(a32, b32, c32)

/* ((a32 >> 16)  * (b32 >> 16)) output have to be 32bit int */
#define silk_SMULTT(a32, b32)               (((a32) >> 16) * ((b32) >> 16))

/* a32 + ((a32 >> 16)  * (b32 >> 16)) output have to be 32bit int */
#define silk_SMLATT(a32, b32, c32)          silk_ADD32((a32),((b32) >> 16) * ((c32) >> 16))

#define silk_SMLALBB(a64, b16, c16)         silk_ADD64((a64),(opus_int64)((opus_int32)(b16) * (opus_int32)(c16)))

/* (a32 * b32) */
#define silk_SMULL(a32, b32)                ((opus_int64)(a32) * /*(opus_int64)*/(b32))

/* Adds two signed 32-bit values in a way that can overflow, while not relying on undefined behaviour
   (just standard two's complement implementation-specific behaviour) */
#define silk_ADD32_ovflw(a, b)              ((opus_int32)((opus_uint32)(a) + (opus_uint32)(b)))
/* Subtractss two signed 32-bit values in a way that can overflow, while not relying on undefined behaviour
   (just standard two's complement implementation-specific behaviour) */
#define silk_SUB32_ovflw(a, b)              ((opus_int32)((opus_uint32)(a) - (opus_uint32)(b)))

/* Multiply-accumulate macros that allow overflow in the addition (ie, no asserts in debug mode) */
#define silk_MLA_ovflw(a32, b32, c32)       silk_ADD32_ovflw((a32), (opus_uint32)(b32) * (opus_uint32)(c32))
#define silk_SMLABB_ovflw(a32, b32, c32)    (silk_ADD32_ovflw((a32) , ((opus_int32)((opus_int16)(b32))) * (opus_int32)((opus_int16)(c32))))

#define silk_DIV32_16(a32, b16)             ((opus_int32)((a32) / (b16)))
#define silk_DIV32(a32, b32)                ((opus_int32)((a32) / (b32)))

/* These macros enables checking for overflow in silk_API_Debug.h*/
#define silk_ADD16(a, b)                    ((a) + (b))
#define silk_ADD32(a, b)                    ((a) + (b))
#define silk_ADD64(a, b)                    ((a) + (b))

#define silk_SUB16(a, b)                    ((a) - (b))
#define silk_SUB32(a, b)                    ((a) - (b))
#define silk_SUB64(a, b)                    ((a) - (b))

#define silk_SAT8(a)                        ((a) > silk_int8_MAX ? silk_int8_MAX  :       \
                                            ((a) < silk_int8_MIN ? silk_int8_MIN  : (a)))
#define silk_SAT16(a)                       ((a) > silk_int16_MAX ? silk_int16_MAX :      \
                                            ((a) < silk_int16_MIN ? silk_int16_MIN : (a)))
#define silk_SAT32(a)                       ((a) > silk_int32_MAX ? silk_int32_MAX :      \
                                            ((a) < silk_int32_MIN ? silk_int32_MIN : (a)))

#define silk_CHECK_FIT8(a)                  (a)
#define silk_CHECK_FIT16(a)                 (a)
#define silk_CHECK_FIT32(a)                 (a)

#define silk_ADD_SAT16(a, b)                (opus_int16)silk_SAT16( silk_ADD32( (opus_int32)(a), (b) ) )
#define silk_ADD_SAT64(a, b)                ((((a) + (b)) & 0x8000000000000000LL) == 0 ?                            \
                                            ((((a) & (b)) & 0x8000000000000000LL) != 0 ? silk_int64_MIN : (a)+(b)) : \
                                            ((((a) | (b)) & 0x8000000000000000LL) == 0 ? silk_int64_MAX : (a)+(b)) )

#define silk_SUB_SAT16(a, b)                (opus_int16)silk_SAT16( silk_SUB32( (opus_int32)(a), (b) ) )
#define silk_SUB_SAT64(a, b)                ((((a)-(b)) & 0x8000000000000000LL) == 0 ?                                               \
                                            (( (a) & ((b)^0x8000000000000000LL) & 0x8000000000000000LL) ? silk_int64_MIN : (a)-(b)) : \
                                            ((((a)^0x8000000000000000LL) & (b)  & 0x8000000000000000LL) ? silk_int64_MAX : (a)-(b)) )

/* Saturation for positive input values */
#define silk_POS_SAT32(a)                   ((a) > silk_int32_MAX ? silk_int32_MAX : (a))

/* Add with saturation for positive input values */
#define silk_ADD_POS_SAT8(a, b)             ((((a)+(b)) & 0x80)                 ? silk_int8_MAX  : ((a)+(b)))
#define silk_ADD_POS_SAT16(a, b)            ((((a)+(b)) & 0x8000)               ? silk_int16_MAX : ((a)+(b)))
#define silk_ADD_POS_SAT32(a, b)            ((((opus_uint32)(a)+(opus_uint32)(b)) & 0x80000000) ? silk_int32_MAX : ((a)+(b)))

#define silk_LSHIFT8(a, shift)              ((opus_int8)((opus_uint8)(a)<<(shift)))         /* shift >= 0, shift < 8  */
#define silk_LSHIFT16(a, shift)             ((opus_int16)((opus_uint16)(a)<<(shift)))       /* shift >= 0, shift < 16 */
#define silk_LSHIFT32(a, shift)             ((opus_int32)((opus_uint32)(a)<<(shift)))       /* shift >= 0, shift < 32 */
#define silk_LSHIFT64(a, shift)             ((opus_int64)((opus_uint64)(a)<<(shift)))       /* shift >= 0, shift < 64 */
#define silk_LSHIFT(a, shift)               silk_LSHIFT32(a, shift)                         /* shift >= 0, shift < 32 */

#define silk_RSHIFT8(a, shift)              ((a)>>(shift))                                  /* shift >= 0, shift < 8  */
#define silk_RSHIFT16(a, shift)             ((a)>>(shift))                                  /* shift >= 0, shift < 16 */
#define silk_RSHIFT32(a, shift)             ((a)>>(shift))                                  /* shift >= 0, shift < 32 */
#define silk_RSHIFT64(a, shift)             ((a)>>(shift))                                  /* shift >= 0, shift < 64 */
#define silk_RSHIFT(a, shift)               silk_RSHIFT32(a, shift)                         /* shift >= 0, shift < 32 */

/* saturates before shifting */
#define silk_LSHIFT_SAT32(a, shift)         (silk_LSHIFT32( silk_LIMIT( (a), silk_RSHIFT32( silk_int32_MIN, (shift) ), \
                                                    silk_RSHIFT32( silk_int32_MAX, (shift) ) ), (shift) ))

#define silk_LSHIFT_ovflw(a, shift)         ((opus_int32)((opus_uint32)(a) << (shift)))     /* shift >= 0, allowed to overflow */
#define silk_LSHIFT_uint(a, shift)          ((a) << (shift))                                /* shift >= 0 */
#define silk_RSHIFT_uint(a, shift)          ((a) >> (shift))                                /* shift >= 0 */

#define silk_ADD_LSHIFT(a, b, shift)        ((a) + silk_LSHIFT((b), (shift)))               /* shift >= 0 */
#define silk_ADD_LSHIFT32(a, b, shift)      silk_ADD32((a), silk_LSHIFT32((b), (shift)))    /* shift >= 0 */
#define silk_ADD_LSHIFT_uint(a, b, shift)   ((a) + silk_LSHIFT_uint((b), (shift)))          /* shift >= 0 */
#define silk_ADD_RSHIFT(a, b, shift)        ((a) + silk_RSHIFT((b), (shift)))               /* shift >= 0 */
#define silk_ADD_RSHIFT32(a, b, shift)      silk_ADD32((a), silk_RSHIFT32((b), (shift)))    /* shift >= 0 */
#define silk_ADD_RSHIFT_uint(a, b, shift)   ((a) + silk_RSHIFT_uint((b), (shift)))          /* shift >= 0 */
#define silk_SUB_LSHIFT32(a, b, shift)      silk_SUB32((a), silk_LSHIFT32((b), (shift)))    /* shift >= 0 */
#define silk_SUB_RSHIFT32(a, b, shift)      silk_SUB32((a), silk_RSHIFT32((b), (shift)))    /* shift >= 0 */

/* Requires that shift > 0 */
#define silk_RSHIFT_ROUND(a, shift)         ((shift) == 1 ? ((a) >> 1) + ((a) & 1) : (((a) >> ((shift) - 1)) + 1) >> 1)
#define silk_RSHIFT_ROUND64(a, shift)       ((shift) == 1 ? ((a) >> 1) + ((a) & 1) : (((a) >> ((shift) - 1)) + 1) >> 1)

/* Number of rightshift required to fit the multiplication */
#define silk_NSHIFT_MUL_32_32(a, b)         ( -(31- (32-silk_CLZ32(silk_abs(a)) + (32-silk_CLZ32(silk_abs(b))))) )
#define silk_NSHIFT_MUL_16_16(a, b)         ( -(15- (16-silk_CLZ16(silk_abs(a)) + (16-silk_CLZ16(silk_abs(b))))) )


#define silk_min(a, b)                      (((a) < (b)) ? (a) : (b))
#define silk_max(a, b)                      (((a) > (b)) ? (a) : (b))

/* Macro to convert floating-point constants to fixed-point */
#define SILK_FIX_CONST( C, Q )              ((opus_int32)((C) * ((opus_int64)1 << (Q)) + 0.5))

/* silk_min() versions with typecast in the function call */
static OPUS_INLINE opus_int silk_min_int(opus_int a, opus_int b)
{
    return (((a) < (b)) ? (a) : (b));
}
static OPUS_INLINE opus_int16 silk_min_16(opus_int16 a, opus_int16 b)
{
    return (((a) < (b)) ? (a) : (b));
}
static OPUS_INLINE opus_int32 silk_min_32(opus_int32 a, opus_int32 b)
{
    return (((a) < (b)) ? (a) : (b));
}
static OPUS_INLINE opus_int64 silk_min_64(opus_int64 a, opus_int64 b)
{
    return (((a) < (b)) ? (a) : (b));
}

/* silk_min() versions with typecast in the function call */
static OPUS_INLINE opus_int silk_max_int(opus_int a, opus_int b)
{
    return (((a) > (b)) ? (a) : (b));
}
static OPUS_INLINE opus_int16 silk_max_16(opus_int16 a, opus_int16 b)
{
    return (((a) > (b)) ? (a) : (b));
}
static OPUS_INLINE opus_int32 silk_max_32(opus_int32 a, opus_int32 b)
{
    return (((a) > (b)) ? (a) : (b));
}
static OPUS_INLINE opus_int64 silk_max_64(opus_int64 a, opus_int64 b)
{
    return (((a) > (b)) ? (a) : (b));
}

#define silk_LIMIT( a, limit1, limit2)      ((limit1) > (limit2) ? ((a) > (limit1) ? (limit1) : ((a) < (limit2) ? (limit2) : (a))) \
                                                                 : ((a) > (limit2) ? (limit2) : ((a) < (limit1) ? (limit1) : (a))))

#define silk_LIMIT_int                      silk_LIMIT
#define silk_LIMIT_16                       silk_LIMIT
#define silk_LIMIT_32                       silk_LIMIT

#define silk_abs(a)                         (((a) >  0)  ? (a) : -(a))            /* Be careful, silk_abs returns wrong when input equals to silk_intXX_MIN */
#define silk_abs_int(a)                     (((a) ^ ((a) >> (8 * sizeof(a) - 1))) - ((a) >> (8 * sizeof(a) - 1)))
#define silk_abs_int32(a)                   (((a) ^ ((a) >> 31)) - ((a) >> 31))
#define silk_abs_int64(a)                   (((a) >  0)  ? (a) : -(a))

#define silk_sign(a)                        ((a) > 0 ? 1 : ( (a) < 0 ? -1 : 0 ))

/* PSEUDO-RANDOM GENERATOR                                                          */
/* Make sure to store the result as the seed for the next call (also in between     */
/* frames), otherwise result won't be random at all. When only using some of the    */
/* bits, take the most significant bits by right-shifting.                          */
#define RAND_MULTIPLIER                     196314165
#define RAND_INCREMENT                      907633515
#define silk_RAND(seed)                     (silk_MLA_ovflw((RAND_INCREMENT), (seed), (RAND_MULTIPLIER)))

/*  Add some multiplication functions that can be easily mapped to ARM. */

/*    silk_SMMUL: Signed top word multiply.
          ARMv6        2 instruction cycles.
          ARMv3M+      3 instruction cycles. use SMULL and ignore LSB registers.(except xM)*/
/*#define silk_SMMUL(a32, b32)                (opus_int32)silk_RSHIFT(silk_SMLAL(silk_SMULWB((a32), (b32)), (a32), silk_RSHIFT_ROUND((b32), 16)), 16)*/
/* the following seems faster on x86 */
#define silk_SMMUL(a32, b32)                (opus_int32)silk_RSHIFT64(silk_SMULL((a32), (b32)), 32)

#if !defined(OPUS_X86_MAY_HAVE_SSE4_1)
#define silk_burg_modified(res_nrg, res_nrg_Q, A_Q16, x, minInvGain_Q30, subfr_length, nb_subfr, D, arch) \
    ((void)(arch), silk_burg_modified_c(res_nrg, res_nrg_Q, A_Q16, x, minInvGain_Q30, subfr_length, nb_subfr, D, arch))

#define silk_inner_prod16_aligned_64(inVec1, inVec2, len, arch) \
    ((void)(arch),silk_inner_prod16_aligned_64_c(inVec1, inVec2, len))
#endif

#include "Inlines.h"
#include "MacroCount.h"
#include "MacroDebug.h"

#ifdef OPUS_ARM_INLINE_ASM
#include "arm/SigProc_FIX_armv4.h"
#endif

#ifdef OPUS_ARM_INLINE_EDSP
#include "arm/SigProc_FIX_armv5e.h"
#endif

#if defined(MIPSr1_ASM)
#include "mips/sigproc_fix_mipsr1.h"
#endif


#ifdef  __cplusplus
}
#endif

#endif /* SILK_SIGPROC_FIX_H */
