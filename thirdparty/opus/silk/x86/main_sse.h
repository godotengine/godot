/* Copyright (c) 2014, Cisco Systems, INC
   Written by XiangMingZhu WeiZhou MinPeng YanWang

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef MAIN_SSE_H
#define MAIN_SSE_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

# if defined(OPUS_X86_MAY_HAVE_SSE4_1)

#if 0 /* FIXME: SSE disabled until silk_VQ_WMat_EC_sse4_1() gets updated. */
#  define OVERRIDE_silk_VQ_WMat_EC

void silk_VQ_WMat_EC_sse4_1(
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

#if defined OPUS_X86_PRESUME_SSE4_1

#define silk_VQ_WMat_EC(ind, rate_dist_Q14, gain_Q7, in_Q14, W_Q18, cb_Q7, cb_gain_Q7, cl_Q5, \
                          mu_Q9, max_gain_Q7, L, arch) \
    ((void)(arch),silk_VQ_WMat_EC_sse4_1(ind, rate_dist_Q14, gain_Q7, in_Q14, W_Q18, cb_Q7, cb_gain_Q7, cl_Q5, \
                          mu_Q9, max_gain_Q7, L))

#else

extern void (*const SILK_VQ_WMAT_EC_IMPL[OPUS_ARCHMASK + 1])(
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

#  define silk_VQ_WMat_EC(ind, rate_dist_Q14, gain_Q7, in_Q14, W_Q18, cb_Q7, cb_gain_Q7, cl_Q5, \
                          mu_Q9, max_gain_Q7, L, arch) \
    ((*SILK_VQ_WMAT_EC_IMPL[(arch) & OPUS_ARCHMASK])(ind, rate_dist_Q14, gain_Q7, in_Q14, W_Q18, cb_Q7, cb_gain_Q7, cl_Q5, \
                          mu_Q9, max_gain_Q7, L))

#endif
#endif

#if 0 /* FIXME: SSE disabled until the NSQ code gets updated. */
#  define OVERRIDE_silk_NSQ

void silk_NSQ_sse4_1(
    const silk_encoder_state    *psEncC,                                    /* I    Encoder State                   */
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

#if defined OPUS_X86_PRESUME_SSE4_1

#define silk_NSQ(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                   HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((void)(arch),silk_NSQ_sse4_1(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                   HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))

#else

extern void (*const SILK_NSQ_IMPL[OPUS_ARCHMASK + 1])(
    const silk_encoder_state    *psEncC,                                    /* I    Encoder State                   */
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

#  define silk_NSQ(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                   HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((*SILK_NSQ_IMPL[(arch) & OPUS_ARCHMASK])(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                   HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))

#endif

#  define OVERRIDE_silk_NSQ_del_dec

void silk_NSQ_del_dec_sse4_1(
    const silk_encoder_state    *psEncC,                                    /* I    Encoder State                   */
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

#if defined OPUS_X86_PRESUME_SSE4_1

#define silk_NSQ_del_dec(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                           HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((void)(arch),silk_NSQ_del_dec_sse4_1(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                           HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))

#else

extern void (*const SILK_NSQ_DEL_DEC_IMPL[OPUS_ARCHMASK + 1])(
    const silk_encoder_state    *psEncC,                                    /* I    Encoder State                   */
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

#  define silk_NSQ_del_dec(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                           HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((*SILK_NSQ_DEL_DEC_IMPL[(arch) & OPUS_ARCHMASK])(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                           HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))

#endif
#endif

void silk_noise_shape_quantizer(
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                       */
    opus_int            signalType,             /* I    Signal type                     */
    const opus_int32    x_sc_Q10[],             /* I                                    */
    opus_int8           pulses[],               /* O                                    */
    opus_int16          xq[],                   /* O                                    */
    opus_int32          sLTP_Q15[],             /* I/O  LTP state                       */
    const opus_int16    a_Q12[],                /* I    Short term prediction coefs     */
    const opus_int16    b_Q14[],                /* I    Long term prediction coefs      */
    const opus_int16    AR_shp_Q13[],           /* I    Noise shaping AR coefs          */
    opus_int            lag,                    /* I    Pitch lag                       */
    opus_int32          HarmShapeFIRPacked_Q14, /* I                                    */
    opus_int            Tilt_Q14,               /* I    Spectral tilt                   */
    opus_int32          LF_shp_Q14,             /* I                                    */
    opus_int32          Gain_Q16,               /* I                                    */
    opus_int            Lambda_Q10,             /* I                                    */
    opus_int            offset_Q10,             /* I                                    */
    opus_int            length,                 /* I    Input length                    */
    opus_int            shapingLPCOrder,        /* I    Noise shaping AR filter order   */
    opus_int            predictLPCOrder,        /* I    Prediction filter order         */
    int                 arch                    /* I    Architecture                    */
);

/**************************/
/* Noise level estimation */
/**************************/
void silk_VAD_GetNoiseLevels(
    const opus_int32            pX[ VAD_N_BANDS ],  /* I    subband energies                            */
    silk_VAD_state              *psSilk_VAD         /* I/O  Pointer to Silk VAD state                   */
);

#  define OVERRIDE_silk_VAD_GetSA_Q8

opus_int silk_VAD_GetSA_Q8_sse4_1(
    silk_encoder_state *psEnC,
    const opus_int16   pIn[]
);

#if defined(OPUS_X86_PRESUME_SSE4_1)
#define silk_VAD_GetSA_Q8(psEnC, pIn, arch) ((void)(arch),silk_VAD_GetSA_Q8_sse4_1(psEnC, pIn))

#else

#  define silk_VAD_GetSA_Q8(psEnC, pIn, arch) \
     ((*SILK_VAD_GETSA_Q8_IMPL[(arch) & OPUS_ARCHMASK])(psEnC, pIn))

extern opus_int (*const SILK_VAD_GETSA_Q8_IMPL[OPUS_ARCHMASK + 1])(
     silk_encoder_state *psEnC,
     const opus_int16   pIn[]);

#endif

# endif
#endif
