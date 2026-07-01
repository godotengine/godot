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
# define MAIN_SSE_H

# ifdef HAVE_CONFIG_H
#  include "config.h"
# endif

# if defined(OPUS_X86_MAY_HAVE_SSE4_1)

void silk_VQ_WMat_EC_sse4_1(
    opus_int8                   *ind,                           /* O    index of best codebook vector               */
    opus_int32                  *res_nrg_Q15,                   /* O    best residual energy                        */
    opus_int32                  *rate_dist_Q8,                  /* O    best total bitrate                          */
    opus_int                    *gain_Q7,                       /* O    sum of absolute LTP coefficients            */
    const opus_int32            *XX_Q17,                        /* I    correlation matrix                          */
    const opus_int32            *xX_Q17,                        /* I    correlation vector                          */
    const opus_int8             *cb_Q7,                         /* I    codebook                                    */
    const opus_uint8            *cb_gain_Q7,                    /* I    codebook effective gain                     */
    const opus_uint8            *cl_Q5,                         /* I    code length for each codebook vector        */
    const opus_int              subfr_len,                      /* I    number of samples per subframe              */
    const opus_int32            max_gain_Q7,                    /* I    maximum sum of absolute LTP coefficients    */
    const opus_int              L                               /* I    number of vectors in codebook               */
);

#  if defined OPUS_X86_PRESUME_SSE4_1

#   define OVERRIDE_silk_VQ_WMat_EC
#   define silk_VQ_WMat_EC(ind, res_nrg_Q15, rate_dist_Q8, gain_Q7, XX_Q17, xX_Q17, cb_Q7, cb_gain_Q7, cl_Q5, \
                           subfr_len, max_gain_Q7, L, arch) \
    ((void)(arch),silk_VQ_WMat_EC_sse4_1(ind, res_nrg_Q15, rate_dist_Q8, gain_Q7, XX_Q17, xX_Q17, cb_Q7, cb_gain_Q7, cl_Q5, \
                          subfr_len, max_gain_Q7, L))

#  elif defined(OPUS_HAVE_RTCD)

extern void (*const SILK_VQ_WMAT_EC_IMPL[OPUS_ARCHMASK + 1])(
    opus_int8                   *ind,                           /* O    index of best codebook vector               */
    opus_int32                  *res_nrg_Q15,                   /* O    best residual energy                        */
    opus_int32                  *rate_dist_Q8,                  /* O    best total bitrate                          */
    opus_int                    *gain_Q7,                       /* O    sum of absolute LTP coefficients            */
    const opus_int32            *XX_Q17,                        /* I    correlation matrix                          */
    const opus_int32            *xX_Q17,                        /* I    correlation vector                          */
    const opus_int8             *cb_Q7,                         /* I    codebook                                    */
    const opus_uint8            *cb_gain_Q7,                    /* I    codebook effective gain                     */
    const opus_uint8            *cl_Q5,                         /* I    code length for each codebook vector        */
    const opus_int              subfr_len,                      /* I    number of samples per subframe              */
    const opus_int32            max_gain_Q7,                    /* I    maximum sum of absolute LTP coefficients    */
    const opus_int              L                               /* I    number of vectors in codebook               */
);

#   define OVERRIDE_silk_VQ_WMat_EC
#   define silk_VQ_WMat_EC(ind, res_nrg_Q15, rate_dist_Q8, gain_Q7, XX_Q17, xX_Q17, cb_Q7, cb_gain_Q7, cl_Q5, \
                           subfr_len, max_gain_Q7, L, arch) \
    ((*SILK_VQ_WMAT_EC_IMPL[(arch) & OPUS_ARCHMASK])(ind, res_nrg_Q15, rate_dist_Q8, gain_Q7, XX_Q17, xX_Q17, cb_Q7, cb_gain_Q7, cl_Q5, \
                          subfr_len, max_gain_Q7, L))

#  endif

void silk_NSQ_sse4_1(
    const silk_encoder_state    *psEncC,                                      /* I    Encoder State                   */
    silk_nsq_state              *NSQ,                                         /* I/O  NSQ state                       */
    SideInfoIndices             *psIndices,                                   /* I/O  Quantization Indices            */
    const opus_int16            x16[],                                        /* I    Input                           */
    opus_int8                   pulses[],                                     /* O    Quantized pulse signal          */
    const opus_int16            *PredCoef_Q12,                                /* I    Short term prediction coefs     */
    const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],      /* I    Long term prediction coefs      */
    const opus_int16            AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I    Noise shaping coefs             */
    const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],            /* I    Long term shaping coefs         */
    const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                     /* I    Spectral tilt                   */
    const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                   /* I    Low frequency shaping coefs     */
    const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                    /* I    Quantization step sizes         */
    const opus_int              pitchL[ MAX_NB_SUBFR ],                       /* I    Pitch lags                      */
    const opus_int              Lambda_Q10,                                   /* I    Rate/distortion tradeoff        */
    const opus_int              LTP_scale_Q14                                 /* I    LTP state scaling               */
);

#  if defined OPUS_X86_PRESUME_SSE4_1

#   define OVERRIDE_silk_NSQ
#   define silk_NSQ(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                    HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((void)(arch),silk_NSQ_sse4_1(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                   HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))

#  elif defined(OPUS_HAVE_RTCD)

extern void (*const SILK_NSQ_IMPL[OPUS_ARCHMASK + 1])(
    const silk_encoder_state    *psEncC,                                      /* I    Encoder State                   */
    silk_nsq_state              *NSQ,                                         /* I/O  NSQ state                       */
    SideInfoIndices             *psIndices,                                   /* I/O  Quantization Indices            */
    const opus_int16            x16[],                                        /* I    Input                           */
    opus_int8                   pulses[],                                     /* O    Quantized pulse signal          */
    const opus_int16            *PredCoef_Q12,                                /* I    Short term prediction coefs     */
    const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],      /* I    Long term prediction coefs      */
    const opus_int16            AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I    Noise shaping coefs             */
    const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],            /* I    Long term shaping coefs         */
    const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                     /* I    Spectral tilt                   */
    const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                   /* I    Low frequency shaping coefs     */
    const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                    /* I    Quantization step sizes         */
    const opus_int              pitchL[ MAX_NB_SUBFR ],                       /* I    Pitch lags                      */
    const opus_int              Lambda_Q10,                                   /* I    Rate/distortion tradeoff        */
    const opus_int              LTP_scale_Q14                                 /* I    LTP state scaling               */
);

#   define OVERRIDE_silk_NSQ
#   define silk_NSQ(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                    HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((*SILK_NSQ_IMPL[(arch) & OPUS_ARCHMASK])(psEncC, NSQ, psIndices, x_Q3, pulses, PredCoef_Q12, LTPCoef_Q14, AR2_Q13, \
                   HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))

#  endif

void silk_NSQ_del_dec_sse4_1(
    const silk_encoder_state    *psEncC,                                      /* I    Encoder State                   */
    silk_nsq_state              *NSQ,                                         /* I/O  NSQ state                       */
    SideInfoIndices             *psIndices,                                   /* I/O  Quantization Indices            */
    const opus_int16            x16[],                                        /* I    Input                           */
    opus_int8                   pulses[],                                     /* O    Quantized pulse signal          */
    const opus_int16            *PredCoef_Q12,                                /* I    Short term prediction coefs     */
    const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],      /* I    Long term prediction coefs      */
    const opus_int16            AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I    Noise shaping coefs             */
    const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],            /* I    Long term shaping coefs         */
    const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                     /* I    Spectral tilt                   */
    const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                   /* I    Low frequency shaping coefs     */
    const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                    /* I    Quantization step sizes         */
    const opus_int              pitchL[ MAX_NB_SUBFR ],                       /* I    Pitch lags                      */
    const opus_int              Lambda_Q10,                                   /* I    Rate/distortion tradeoff        */
    const opus_int              LTP_scale_Q14                                 /* I    LTP state scaling               */
);

void silk_NSQ_del_dec_avx2(
    const silk_encoder_state *psEncC,                            /* I    Encoder State               */
    silk_nsq_state *NSQ,                                         /* I/O  NSQ state                   */
    SideInfoIndices *psIndices,                                  /* I/O  Quantization Indices        */
    const opus_int16 x16[],                                      /* I    Input                       */
    opus_int8 pulses[],                                          /* O    Quantized pulse signal      */
    const opus_int16 *PredCoef_Q12,                              /* I    Short term prediction coefs */
    const opus_int16 LTPCoef_Q14[LTP_ORDER * MAX_NB_SUBFR],      /* I    Long term prediction coefs  */
    const opus_int16 AR_Q13[MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER], /* I    Noise shaping coefs         */
    const opus_int HarmShapeGain_Q14[MAX_NB_SUBFR],              /* I    Long term shaping coefs     */
    const opus_int Tilt_Q14[MAX_NB_SUBFR],                       /* I    Spectral tilt               */
    const opus_int32 LF_shp_Q14[MAX_NB_SUBFR],                   /* I    Low frequency shaping coefs */
    const opus_int32 Gains_Q16[MAX_NB_SUBFR],                    /* I    Quantization step sizes     */
    const opus_int32 pitchL[MAX_NB_SUBFR],                       /* I    Pitch lags                  */
    const opus_int Lambda_Q10,                                   /* I    Rate/distortion tradeoff    */
    const opus_int LTP_scale_Q14                                 /* I    LTP state scaling           */
);

#  if defined (OPUS_X86_PRESUME_AVX2)

#   define OVERRIDE_silk_NSQ_del_dec
#   define silk_NSQ_del_dec(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, \
                            HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((void)(arch),silk_NSQ_del_dec_avx2(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, \
                           HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))

#  elif defined (OPUS_X86_PRESUME_SSE4_1) && !defined(OPUS_X86_MAY_HAVE_AVX2)

#   define OVERRIDE_silk_NSQ_del_dec
#   define silk_NSQ_del_dec(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, \
                            HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((void)(arch),silk_NSQ_del_dec_sse4_1(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, \
                           HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))

#  elif defined(OPUS_HAVE_RTCD)

extern void (*const SILK_NSQ_DEL_DEC_IMPL[OPUS_ARCHMASK + 1])(
    const silk_encoder_state    *psEncC,                                      /* I    Encoder State                   */
    silk_nsq_state              *NSQ,                                         /* I/O  NSQ state                       */
    SideInfoIndices             *psIndices,                                   /* I/O  Quantization Indices            */
    const opus_int16            x16[],                                        /* I    Input                           */
    opus_int8                   pulses[],                                     /* O    Quantized pulse signal          */
    const opus_int16            *PredCoef_Q12,                                /* I    Short term prediction coefs     */
    const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],      /* I    Long term prediction coefs      */
    const opus_int16            AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I    Noise shaping coefs             */
    const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],            /* I    Long term shaping coefs         */
    const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                     /* I    Spectral tilt                   */
    const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                   /* I    Low frequency shaping coefs     */
    const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                    /* I    Quantization step sizes         */
    const opus_int              pitchL[ MAX_NB_SUBFR ],                       /* I    Pitch lags                      */
    const opus_int              Lambda_Q10,                                   /* I    Rate/distortion tradeoff        */
    const opus_int              LTP_scale_Q14                                 /* I    LTP state scaling               */
);

#   define OVERRIDE_silk_NSQ_del_dec
#   define silk_NSQ_del_dec(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, \
                            HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((*SILK_NSQ_DEL_DEC_IMPL[(arch) & OPUS_ARCHMASK])(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, \
                           HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))

#  endif

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

opus_int silk_VAD_GetSA_Q8_sse4_1(
    silk_encoder_state *psEnC,
    const opus_int16   pIn[]
);

#  if defined(OPUS_X86_PRESUME_SSE4_1)

#   define OVERRIDE_silk_VAD_GetSA_Q8
#   define silk_VAD_GetSA_Q8(psEnC, pIn, arch) ((void)(arch),silk_VAD_GetSA_Q8_sse4_1(psEnC, pIn))

#  elif defined(OPUS_HAVE_RTCD)

extern opus_int (*const SILK_VAD_GETSA_Q8_IMPL[OPUS_ARCHMASK + 1])(
     silk_encoder_state *psEnC,
     const opus_int16   pIn[]);

#   define OVERRIDE_silk_VAD_GetSA_Q8
#   define silk_VAD_GetSA_Q8(psEnC, pIn, arch) \
      ((*SILK_VAD_GETSA_Q8_IMPL[(arch) & OPUS_ARCHMASK])(psEnC, pIn))

#  endif

#ifndef FIXED_POINT
double silk_inner_product_FLP_avx2(
    const silk_float    *data1,
    const silk_float    *data2,
    opus_int            dataSize
);

#if defined (OPUS_X86_PRESUME_AVX2)

#define OVERRIDE_inner_product_FLP
#define silk_inner_product_FLP(data1, data2, dataSize, arch) ((void)arch,silk_inner_product_FLP_avx2(data1, data2, dataSize))

#elif defined(OPUS_HAVE_RTCD) && defined(OPUS_X86_MAY_HAVE_AVX2)

#define OVERRIDE_inner_product_FLP
extern double (*const SILK_INNER_PRODUCT_FLP_IMPL[OPUS_ARCHMASK + 1])(
    const silk_float    *data1,
    const silk_float    *data2,
    opus_int            dataSize
);

#define silk_inner_product_FLP(data1, data2, dataSize, arch) ((void)arch,(*SILK_INNER_PRODUCT_FLP_IMPL[(arch) & OPUS_ARCHMASK])(data1, data2, dataSize))

#endif
#endif

# endif
#endif
