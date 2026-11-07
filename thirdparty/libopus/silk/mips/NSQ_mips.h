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

#ifndef NSQ_MIPS_H__
#define NSQ_MIPS_H__

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "main.h"
#include "macros.h"

#if defined (__mips_dsp) && __mips == 32

#define MIPS_MULT __builtin_mips_mult
#define MIPS_MADD __builtin_mips_madd
#define MIPS_EXTR_R __builtin_mips_extr_r_w

#define OVERRIDE_silk_noise_shape_quantizer_short_prediction
/* suddenly performance is worse */
#define dont_OVERRIDE_silk_NSQ_noise_shape_feedback_loop

/* gets worst performance result */
#elif defined(__mips_isa_rev) && __mips == 32

static inline long long MIPS_MULT(int a, int b) {
    return (long long)a * b;
}

static inline long long MIPS_MADD(long long acc, int a, int b) {
    return acc + (long long)a * b;
}

static inline opus_val32 MIPS_EXTR_R(long long acc, int shift) {
    return (opus_val32)((acc + (1 << shift) / 2) >> shift);
}

#define OVERRIDE_silk_noise_shape_quantizer_short_prediction
#define OVERRIDE_silk_NSQ_noise_shape_feedback_loop

#endif

#if defined(OVERRIDE_silk_noise_shape_quantizer_short_prediction)

static OPUS_INLINE opus_int32 silk_noise_shape_quantizer_short_prediction_mips(const opus_int32 *buf32, const opus_int16 *coef16, opus_int order)
{
    opus_int64 out;
    silk_assert( order == 10 || order == 16 );

    out = MIPS_MULT(      buf32[  0 ], coef16[ 0 ] );
    out = MIPS_MADD( out, buf32[ -1 ], coef16[ 1 ] );
    out = MIPS_MADD( out, buf32[ -2 ], coef16[ 2 ] );
    out = MIPS_MADD( out, buf32[ -3 ], coef16[ 3 ] );
    out = MIPS_MADD( out, buf32[ -4 ], coef16[ 4 ] );
    out = MIPS_MADD( out, buf32[ -5 ], coef16[ 5 ] );
    out = MIPS_MADD( out, buf32[ -6 ], coef16[ 6 ] );
    out = MIPS_MADD( out, buf32[ -7 ], coef16[ 7 ] );
    out = MIPS_MADD( out, buf32[ -8 ], coef16[ 8 ] );
    out = MIPS_MADD( out, buf32[ -9 ], coef16[ 9 ] );

    if( order == 16 )
    {
        out = MIPS_MADD( out, buf32[ -10 ], coef16[ 10 ] );
        out = MIPS_MADD( out, buf32[ -11 ], coef16[ 11 ] );
        out = MIPS_MADD( out, buf32[ -12 ], coef16[ 12 ] );
        out = MIPS_MADD( out, buf32[ -13 ], coef16[ 13 ] );
        out = MIPS_MADD( out, buf32[ -14 ], coef16[ 14 ] );
        out = MIPS_MADD( out, buf32[ -15 ], coef16[ 15 ] );
    }
    return MIPS_EXTR_R(out, 16);
}

#undef  silk_noise_shape_quantizer_short_prediction
#define silk_noise_shape_quantizer_short_prediction(in, coef, coefRev, order, arch)  ((void)arch,silk_noise_shape_quantizer_short_prediction_mips(in, coef, order))

#endif /* OVERRIDE_silk_noise_shape_quantizer_short_prediction */


#if defined(OVERRIDE_silk_NSQ_noise_shape_feedback_loop)

static OPUS_INLINE opus_int32 silk_NSQ_noise_shape_feedback_loop_mips(const opus_int32 *data0, opus_int32 *data1, const opus_int16 *coef, opus_int order)
{
    opus_int32 out;
    opus_int32 tmp1, tmp2;
    opus_int j;

    tmp2 = data0[0];
    tmp1 = data1[0];
    data1[0] = tmp2;

    out = MIPS_MULT(tmp2, coef[0]);

    for (j = 2; j < order; j += 2) {
        tmp2 = data1[j - 1];
        data1[j - 1] = tmp1;
        out = MIPS_MADD(out, tmp1, coef[j - 1]);
        tmp1 = data1[j + 0];
        data1[j + 0] = tmp2;
        out = MIPS_MADD(out, tmp2, coef[j]);
    }
    data1[order - 1] = tmp1;
    out = MIPS_MADD(out, tmp1, coef[order - 1]);
    /* silk_SMLAWB: shift right by 16  &&  Q11 -> Q12: shift left by 1 */
    return MIPS_EXTR_R( out, (16 - 1) );
}

#undef  silk_NSQ_noise_shape_feedback_loop
#define silk_NSQ_noise_shape_feedback_loop(data0, data1, coef, order, arch)  ((void)arch,silk_NSQ_noise_shape_feedback_loop_mips(data0, data1, coef, order))

#endif /* OVERRIDE_silk_NSQ_noise_shape_feedback_loop */

#endif /* NSQ_DEL_DEC_MIPSR1_H__ */
