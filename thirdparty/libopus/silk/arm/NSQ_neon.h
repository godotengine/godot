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
#ifndef SILK_NSQ_NEON_H
#define SILK_NSQ_NEON_H

#include "cpu_support.h"
#include "SigProc_FIX.h"

#undef silk_short_prediction_create_arch_coef
/* For vectorized calc, reverse a_Q12 coefs, convert to 32-bit, and shift for vqdmulhq_s32. */
static OPUS_INLINE void silk_short_prediction_create_arch_coef_neon(opus_int32 *out, const opus_int16 *in, opus_int order)
{
    out[15] = silk_LSHIFT32(in[0], 15);
    out[14] = silk_LSHIFT32(in[1], 15);
    out[13] = silk_LSHIFT32(in[2], 15);
    out[12] = silk_LSHIFT32(in[3], 15);
    out[11] = silk_LSHIFT32(in[4], 15);
    out[10] = silk_LSHIFT32(in[5], 15);
    out[9]  = silk_LSHIFT32(in[6], 15);
    out[8]  = silk_LSHIFT32(in[7], 15);
    out[7]  = silk_LSHIFT32(in[8], 15);
    out[6]  = silk_LSHIFT32(in[9], 15);

    if (order == 16)
    {
        out[5] = silk_LSHIFT32(in[10], 15);
        out[4] = silk_LSHIFT32(in[11], 15);
        out[3] = silk_LSHIFT32(in[12], 15);
        out[2] = silk_LSHIFT32(in[13], 15);
        out[1] = silk_LSHIFT32(in[14], 15);
        out[0] = silk_LSHIFT32(in[15], 15);
    }
    else
    {
        out[5] = 0;
        out[4] = 0;
        out[3] = 0;
        out[2] = 0;
        out[1] = 0;
        out[0] = 0;
    }
}

#if defined(OPUS_ARM_PRESUME_NEON_INTR)

#define silk_short_prediction_create_arch_coef(out, in, order) \
    (silk_short_prediction_create_arch_coef_neon(out, in, order))

#elif defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_MAY_HAVE_NEON_INTR)

#define silk_short_prediction_create_arch_coef(out, in, order) \
    do { if (arch >= OPUS_ARCH_ARM_NEON) { silk_short_prediction_create_arch_coef_neon(out, in, order); } } while (0)

#endif

opus_int32 silk_noise_shape_quantizer_short_prediction_neon(const opus_int32 *buf32, const opus_int32 *coef32, opus_int order);

opus_int32 silk_NSQ_noise_shape_feedback_loop_neon(const opus_int32 *data0, opus_int32 *data1, const opus_int16 *coef, opus_int order);

#if defined(OPUS_ARM_PRESUME_NEON_INTR)
#undef silk_noise_shape_quantizer_short_prediction
#define silk_noise_shape_quantizer_short_prediction(in, coef, coefRev, order, arch) \
    ((void)arch,silk_noise_shape_quantizer_short_prediction_neon(in, coefRev, order))

#undef silk_NSQ_noise_shape_feedback_loop
#define silk_NSQ_noise_shape_feedback_loop(data0, data1, coef, order, arch)  ((void)arch,silk_NSQ_noise_shape_feedback_loop_neon(data0, data1, coef, order))

#elif defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_MAY_HAVE_NEON_INTR)

/* silk_noise_shape_quantizer_short_prediction implementations take different parameters based on arch
   (coef vs. coefRev) so can't use the usual IMPL table implementation */
#undef silk_noise_shape_quantizer_short_prediction
#define silk_noise_shape_quantizer_short_prediction(in, coef, coefRev, order, arch)  \
    (arch >= OPUS_ARCH_ARM_NEON ? \
        silk_noise_shape_quantizer_short_prediction_neon(in, coefRev, order) : \
        silk_noise_shape_quantizer_short_prediction_c(in, coef, order))

extern opus_int32
 (*const SILK_NSQ_NOISE_SHAPE_FEEDBACK_LOOP_IMPL[OPUS_ARCHMASK+1])(
 const opus_int32 *data0, opus_int32 *data1, const opus_int16 *coef,
 opus_int order);

#undef silk_NSQ_noise_shape_feedback_loop
#define silk_NSQ_noise_shape_feedback_loop(data0, data1, coef, order, arch) \
 (SILK_NSQ_NOISE_SHAPE_FEEDBACK_LOOP_IMPL[(arch)&OPUS_ARCHMASK](data0, data1, \
 coef, order))

#endif

#endif /* SILK_NSQ_NEON_H */
