/***********************************************************************
Copyright (C) 2014 Vidyo
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
# include "config.h"
#endif

#include "main_FIX.h"
#include "NSQ.h"
#include "SigProc_FIX.h"

#if defined(OPUS_HAVE_RTCD)

# if (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && \
 !defined(OPUS_ARM_PRESUME_NEON_INTR))

void (*const SILK_BIQUAD_ALT_STRIDE2_IMPL[OPUS_ARCHMASK + 1])(
        const opus_int16            *in,                /* I     input signal                                               */
        const opus_int32            *B_Q28,             /* I     MA coefficients [3]                                        */
        const opus_int32            *A_Q28,             /* I     AR coefficients [2]                                        */
        opus_int32                  *S,                 /* I/O   State vector [4]                                           */
        opus_int16                  *out,               /* O     output signal                                              */
        const opus_int32            len                 /* I     signal length (must be even)                               */
) = {
      silk_biquad_alt_stride2_c,    /* ARMv4 */
      silk_biquad_alt_stride2_c,    /* EDSP */
      silk_biquad_alt_stride2_c,    /* Media */
      silk_biquad_alt_stride2_neon, /* Neon */
      silk_biquad_alt_stride2_neon, /* dotprod */
};

opus_int32 (*const SILK_LPC_INVERSE_PRED_GAIN_IMPL[OPUS_ARCHMASK + 1])( /* O   Returns inverse prediction gain in energy domain, Q30        */
        const opus_int16            *A_Q12,                             /* I   Prediction coefficients, Q12 [order]                         */
        const opus_int              order                               /* I   Prediction order                                             */
) = {
      silk_LPC_inverse_pred_gain_c,    /* ARMv4 */
      silk_LPC_inverse_pred_gain_c,    /* EDSP */
      silk_LPC_inverse_pred_gain_c,    /* Media */
      silk_LPC_inverse_pred_gain_neon, /* Neon */
      silk_LPC_inverse_pred_gain_neon, /* dotprod */
};

void  (*const SILK_NSQ_DEL_DEC_IMPL[OPUS_ARCHMASK + 1])(
        const silk_encoder_state    *psEncC,                                    /* I    Encoder State                   */
        silk_nsq_state              *NSQ,                                       /* I/O  NSQ state                       */
        SideInfoIndices             *psIndices,                                 /* I/O  Quantization Indices            */
        const opus_int16            x16[],                                      /* I    Input                           */
        opus_int8                   pulses[],                                   /* O    Quantized pulse signal          */
        const opus_int16            *PredCoef_Q12,                              /* I    Short term prediction coefs     */
        const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],    /* I    Long term prediction coefs      */
        const opus_int16            AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I Noise shaping coefs              */
        const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],          /* I    Long term shaping coefs         */
        const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                   /* I    Spectral tilt                   */
        const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                 /* I    Low frequency shaping coefs     */
        const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                  /* I    Quantization step sizes         */
        const opus_int              pitchL[ MAX_NB_SUBFR ],                     /* I    Pitch lags                      */
        const opus_int              Lambda_Q10,                                 /* I    Rate/distortion tradeoff        */
        const opus_int              LTP_scale_Q14                               /* I    LTP state scaling               */
) = {
      silk_NSQ_del_dec_c,    /* ARMv4 */
      silk_NSQ_del_dec_c,    /* EDSP */
      silk_NSQ_del_dec_c,    /* Media */
      silk_NSQ_del_dec_neon, /* Neon */
      silk_NSQ_del_dec_neon, /* dotprod */
};

/*There is no table for silk_noise_shape_quantizer_short_prediction because the
   NEON version takes different parameters than the C version.
  Instead RTCD is done via if statements at the call sites.
  See NSQ_neon.h for details.*/

opus_int32
 (*const SILK_NSQ_NOISE_SHAPE_FEEDBACK_LOOP_IMPL[OPUS_ARCHMASK+1])(
 const opus_int32 *data0, opus_int32 *data1, const opus_int16 *coef,
 opus_int order) = {
  silk_NSQ_noise_shape_feedback_loop_c,    /* ARMv4 */
  silk_NSQ_noise_shape_feedback_loop_c,    /* EDSP */
  silk_NSQ_noise_shape_feedback_loop_c,    /* Media */
  silk_NSQ_noise_shape_feedback_loop_neon, /* NEON */
  silk_NSQ_noise_shape_feedback_loop_neon, /* dotprod */
};

# endif

# if defined(FIXED_POINT) && \
 defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR)

void (*const SILK_WARPED_AUTOCORRELATION_FIX_IMPL[OPUS_ARCHMASK + 1])(
          opus_int32                *corr,                                  /* O    Result [order + 1]                                                          */
          opus_int                  *scale,                                 /* O    Scaling of the correlation vector                                           */
    const opus_int16                *input,                                 /* I    Input data to correlate                                                     */
    const opus_int                  warping_Q16,                            /* I    Warping coefficient                                                         */
    const opus_int                  length,                                 /* I    Length of input                                                             */
    const opus_int                  order                                   /* I    Correlation order (even)                                                    */
) = {
      silk_warped_autocorrelation_FIX_c,    /* ARMv4 */
      silk_warped_autocorrelation_FIX_c,    /* EDSP */
      silk_warped_autocorrelation_FIX_c,    /* Media */
      silk_warped_autocorrelation_FIX_neon, /* Neon */
      silk_warped_autocorrelation_FIX_neon, /* dotprod */
};

# endif

#endif /* OPUS_HAVE_RTCD */
