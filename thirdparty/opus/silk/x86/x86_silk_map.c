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

#if defined(HAVE_CONFIG_H)
#include "config.h"
#endif

#include "celt/x86/x86cpu.h"
#include "structs.h"
#include "SigProc_FIX.h"
#ifndef FIXED_POINT
#include "SigProc_FLP.h"
#endif
#include "pitch.h"
#include "main.h"

#if defined(OPUS_HAVE_RTCD) && !defined(OPUS_X86_PRESUME_AVX2)

#if defined(FIXED_POINT)

#include "fixed/main_FIX.h"

opus_int64 (*const SILK_INNER_PROD16_IMPL[ OPUS_ARCHMASK + 1 ] )(
    const opus_int16 *inVec1,
    const opus_int16 *inVec2,
    const opus_int   len
) = {
  silk_inner_prod16_c,                  /* non-sse */
  silk_inner_prod16_c,
  silk_inner_prod16_c,
  MAY_HAVE_SSE4_1( silk_inner_prod16 ), /* sse4.1 */
  MAY_HAVE_SSE4_1( silk_inner_prod16 )  /* avx */
};

#endif

opus_int (*const SILK_VAD_GETSA_Q8_IMPL[ OPUS_ARCHMASK + 1 ] )(
    silk_encoder_state *psEncC,
    const opus_int16   pIn[]
) = {
  silk_VAD_GetSA_Q8_c,                  /* non-sse */
  silk_VAD_GetSA_Q8_c,
  silk_VAD_GetSA_Q8_c,
  MAY_HAVE_SSE4_1( silk_VAD_GetSA_Q8 ), /* sse4.1 */
  MAY_HAVE_SSE4_1( silk_VAD_GetSA_Q8 )  /* avx */
};

void (*const SILK_NSQ_IMPL[ OPUS_ARCHMASK + 1 ] )(
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
) = {
  silk_NSQ_c,                  /* non-sse */
  silk_NSQ_c,
  silk_NSQ_c,
  MAY_HAVE_SSE4_1( silk_NSQ ), /* sse4.1 */
  MAY_HAVE_SSE4_1( silk_NSQ )  /* avx */
};

void (*const SILK_VQ_WMAT_EC_IMPL[ OPUS_ARCHMASK + 1 ] )(
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
) = {
  silk_VQ_WMat_EC_c,                  /* non-sse */
  silk_VQ_WMat_EC_c,
  silk_VQ_WMat_EC_c,
  MAY_HAVE_SSE4_1( silk_VQ_WMat_EC ), /* sse4.1 */
  MAY_HAVE_SSE4_1( silk_VQ_WMat_EC )  /* avx */
};

void (*const SILK_NSQ_DEL_DEC_IMPL[ OPUS_ARCHMASK + 1 ] )(
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
) = {
  silk_NSQ_del_dec_c,                  /* non-sse */
  silk_NSQ_del_dec_c,
  silk_NSQ_del_dec_c,
  MAY_HAVE_SSE4_1( silk_NSQ_del_dec ), /* sse4.1 */
  MAY_HAVE_AVX2( silk_NSQ_del_dec )  /* avx */
};

#if defined(FIXED_POINT)

void (*const SILK_BURG_MODIFIED_IMPL[ OPUS_ARCHMASK + 1 ] )(
    opus_int32                  *res_nrg,           /* O    Residual energy                                             */
    opus_int                    *res_nrg_Q,         /* O    Residual energy Q value                                     */
    opus_int32                  A_Q16[],            /* O    Prediction coefficients (length order)                      */
    const opus_int16            x[],                /* I    Input signal, length: nb_subfr * ( D + subfr_length )       */
    const opus_int32            minInvGain_Q30,     /* I    Inverse of max prediction gain                              */
    const opus_int              subfr_length,       /* I    Input signal subframe length (incl. D preceding samples)    */
    const opus_int              nb_subfr,           /* I    Number of subframes stacked in x                            */
    const opus_int              D,                  /* I    Order                                                       */
    int                         arch                /* I    Run-time architecture                                       */
) = {
  silk_burg_modified_c,                  /* non-sse */
  silk_burg_modified_c,
  silk_burg_modified_c,
  MAY_HAVE_SSE4_1( silk_burg_modified ), /* sse4.1 */
  MAY_HAVE_SSE4_1( silk_burg_modified )  /* avx */
};

#endif

#ifndef FIXED_POINT

double (*const SILK_INNER_PRODUCT_FLP_IMPL[ OPUS_ARCHMASK + 1 ] )(
    const silk_float    *data1,
    const silk_float    *data2,
    opus_int            dataSize
) = {
  silk_inner_product_FLP_c,                  /* non-sse */
  silk_inner_product_FLP_c,
  silk_inner_product_FLP_c,
  silk_inner_product_FLP_c, /* sse4.1 */
  MAY_HAVE_AVX2( silk_inner_product_FLP )  /* avx */
};

#endif

#endif
