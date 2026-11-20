/***********************************************************************
Copyright (c) 2014 Vidyo.
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
#ifndef SILK_NSQ_H
#define SILK_NSQ_H

#include "SigProc_FIX.h"

#undef silk_short_prediction_create_arch_coef

static OPUS_INLINE opus_int32 silk_noise_shape_quantizer_short_prediction_c(const opus_int32 *buf32, const opus_int16 *coef16, opus_int order)
{
    opus_int32 out;
    silk_assert( order == 10 || order == 16 );

    /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
    out = silk_RSHIFT( order, 1 );
    out = silk_SMLAWB( out, buf32[  0 ], coef16[ 0 ] );
    out = silk_SMLAWB( out, buf32[ -1 ], coef16[ 1 ] );
    out = silk_SMLAWB( out, buf32[ -2 ], coef16[ 2 ] );
    out = silk_SMLAWB( out, buf32[ -3 ], coef16[ 3 ] );
    out = silk_SMLAWB( out, buf32[ -4 ], coef16[ 4 ] );
    out = silk_SMLAWB( out, buf32[ -5 ], coef16[ 5 ] );
    out = silk_SMLAWB( out, buf32[ -6 ], coef16[ 6 ] );
    out = silk_SMLAWB( out, buf32[ -7 ], coef16[ 7 ] );
    out = silk_SMLAWB( out, buf32[ -8 ], coef16[ 8 ] );
    out = silk_SMLAWB( out, buf32[ -9 ], coef16[ 9 ] );

    if( order == 16 )
    {
        out = silk_SMLAWB( out, buf32[ -10 ], coef16[ 10 ] );
        out = silk_SMLAWB( out, buf32[ -11 ], coef16[ 11 ] );
        out = silk_SMLAWB( out, buf32[ -12 ], coef16[ 12 ] );
        out = silk_SMLAWB( out, buf32[ -13 ], coef16[ 13 ] );
        out = silk_SMLAWB( out, buf32[ -14 ], coef16[ 14 ] );
        out = silk_SMLAWB( out, buf32[ -15 ], coef16[ 15 ] );
    }
    return out;
}

#define silk_noise_shape_quantizer_short_prediction(in, coef, coefRev, order, arch)  ((void)arch,silk_noise_shape_quantizer_short_prediction_c(in, coef, order))

static OPUS_INLINE opus_int32 silk_NSQ_noise_shape_feedback_loop_c(const opus_int32 *data0, opus_int32 *data1, const opus_int16 *coef, opus_int order)
{
    opus_int32 out;
    opus_int32 tmp1, tmp2;
    opus_int j;

    tmp2 = data0[0];
    tmp1 = data1[0];
    data1[0] = tmp2;

    out = silk_RSHIFT(order, 1);
    out = silk_SMLAWB(out, tmp2, coef[0]);

    for (j = 2; j < order; j += 2) {
        tmp2 = data1[j - 1];
        data1[j - 1] = tmp1;
        out = silk_SMLAWB(out, tmp1, coef[j - 1]);
        tmp1 = data1[j + 0];
        data1[j + 0] = tmp2;
        out = silk_SMLAWB(out, tmp2, coef[j]);
    }
    data1[order - 1] = tmp1;
    out = silk_SMLAWB(out, tmp1, coef[order - 1]);
    /* Q11 -> Q12 */
    out = silk_LSHIFT32( out, 1 );
    return out;
}

#define silk_NSQ_noise_shape_feedback_loop(data0, data1, coef, order, arch)  ((void)arch,silk_NSQ_noise_shape_feedback_loop_c(data0, data1, coef, order))

#if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
#include "arm/NSQ_neon.h"
#endif

#if defined(__mips)
#include "mips/NSQ_mips.h"
#endif

#endif /* SILK_NSQ_H */
