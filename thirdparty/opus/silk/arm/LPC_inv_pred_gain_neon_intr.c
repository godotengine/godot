/***********************************************************************
Copyright (c) 2017 Google Inc.
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
#include "config.h"
#endif

#include <arm_neon.h>
#include "SigProc_FIX.h"
#include "define.h"

#define QA                          24
#define A_LIMIT                     SILK_FIX_CONST( 0.99975, QA )

#define MUL32_FRAC_Q(a32, b32, Q)   ((opus_int32)(silk_RSHIFT_ROUND64(silk_SMULL(a32, b32), Q)))

/* The difficulty is how to judge a 64-bit signed integer tmp64 is 32-bit overflowed,
 * since NEON has no 64-bit min, max or comparison instructions.
 * A failed idea is to compare the results of vmovn(tmp64) and vqmovn(tmp64) whether they are equal or not.
 * However, this idea fails when the tmp64 is something like 0xFFFFFFF980000000.
 * Here we know that mult2Q >= 1, so the highest bit (bit 63, sign bit) of tmp64 must equal to bit 62.
 * tmp64 was shifted left by 1 and we got tmp64'. If high_half(tmp64') != 0 and high_half(tmp64') != -1,
 * then we know that bit 31 to bit 63 of tmp64 can not all be the sign bit, and therefore tmp64 is 32-bit overflowed.
 * That is, we judge if tmp64' > 0x00000000FFFFFFFF, or tmp64' <= 0xFFFFFFFF00000000.
 * We use narrowing shift right 31 bits to tmp32' to save data bandwidth and instructions.
 * That is, we judge if tmp32' > 0x00000000, or tmp32' <= 0xFFFFFFFF.
 */

/* Compute inverse of LPC prediction gain, and                          */
/* test if LPC coefficients are stable (all poles within unit circle)   */
static OPUS_INLINE opus_int32 LPC_inverse_pred_gain_QA_neon( /* O   Returns inverse prediction gain in energy domain, Q30    */
    opus_int32           A_QA[ SILK_MAX_ORDER_LPC ],         /* I   Prediction coefficients                                  */
    const opus_int       order                               /* I   Prediction order                                         */
)
{
    opus_int   k, n, mult2Q;
    opus_int32 invGain_Q30, rc_Q31, rc_mult1_Q30, rc_mult2, tmp1, tmp2;
    opus_int32 max, min;
    int32x4_t  max_s32x4, min_s32x4;
    int32x2_t  max_s32x2, min_s32x2;

    max_s32x4 = vdupq_n_s32( silk_int32_MIN );
    min_s32x4 = vdupq_n_s32( silk_int32_MAX );
    invGain_Q30 = SILK_FIX_CONST( 1, 30 );
    for( k = order - 1; k > 0; k-- ) {
        int32x2_t rc_Q31_s32x2, rc_mult2_s32x2;
        int64x2_t mult2Q_s64x2;

        /* Check for stability */
        if( ( A_QA[ k ] > A_LIMIT ) || ( A_QA[ k ] < -A_LIMIT ) ) {
            return 0;
        }

        /* Set RC equal to negated AR coef */
        rc_Q31 = -silk_LSHIFT( A_QA[ k ], 31 - QA );

        /* rc_mult1_Q30 range: [ 1 : 2^30 ] */
        rc_mult1_Q30 = silk_SUB32( SILK_FIX_CONST( 1, 30 ), silk_SMMUL( rc_Q31, rc_Q31 ) );
        silk_assert( rc_mult1_Q30 > ( 1 << 15 ) );                   /* reduce A_LIMIT if fails */
        silk_assert( rc_mult1_Q30 <= ( 1 << 30 ) );

        /* Update inverse gain */
        /* invGain_Q30 range: [ 0 : 2^30 ] */
        invGain_Q30 = silk_LSHIFT( silk_SMMUL( invGain_Q30, rc_mult1_Q30 ), 2 );
        silk_assert( invGain_Q30 >= 0           );
        silk_assert( invGain_Q30 <= ( 1 << 30 ) );
        if( invGain_Q30 < SILK_FIX_CONST( 1.0f / MAX_PREDICTION_POWER_GAIN, 30 ) ) {
            return 0;
        }

        /* rc_mult2 range: [ 2^30 : silk_int32_MAX ] */
        mult2Q = 32 - silk_CLZ32( silk_abs( rc_mult1_Q30 ) );
        rc_mult2 = silk_INVERSE32_varQ( rc_mult1_Q30, mult2Q + 30 );

        /* Update AR coefficient */
        rc_Q31_s32x2   = vdup_n_s32( rc_Q31 );
        mult2Q_s64x2   = vdupq_n_s64( -mult2Q );
        rc_mult2_s32x2 = vdup_n_s32( rc_mult2 );

        for( n = 0; n < ( ( k + 1 ) >> 1 ) - 3; n += 4 ) {
            /* We always calculate extra elements of A_QA buffer when ( k % 4 ) != 0, to take the advantage of SIMD parallelization. */
            int32x4_t tmp1_s32x4, tmp2_s32x4, t0_s32x4, t1_s32x4, s0_s32x4, s1_s32x4, t_QA0_s32x4, t_QA1_s32x4;
            int64x2_t t0_s64x2, t1_s64x2, t2_s64x2, t3_s64x2;
            tmp1_s32x4  = vld1q_s32( A_QA + n );
            tmp2_s32x4  = vld1q_s32( A_QA + k - n - 4 );
            tmp2_s32x4  = vrev64q_s32( tmp2_s32x4 );
            tmp2_s32x4  = vcombine_s32( vget_high_s32( tmp2_s32x4 ), vget_low_s32( tmp2_s32x4 ) );
            t0_s32x4    = vqrdmulhq_lane_s32( tmp2_s32x4, rc_Q31_s32x2, 0 );
            t1_s32x4    = vqrdmulhq_lane_s32( tmp1_s32x4, rc_Q31_s32x2, 0 );
            t_QA0_s32x4 = vqsubq_s32( tmp1_s32x4, t0_s32x4 );
            t_QA1_s32x4 = vqsubq_s32( tmp2_s32x4, t1_s32x4 );
            t0_s64x2    = vmull_s32( vget_low_s32 ( t_QA0_s32x4 ), rc_mult2_s32x2 );
            t1_s64x2    = vmull_s32( vget_high_s32( t_QA0_s32x4 ), rc_mult2_s32x2 );
            t2_s64x2    = vmull_s32( vget_low_s32 ( t_QA1_s32x4 ), rc_mult2_s32x2 );
            t3_s64x2    = vmull_s32( vget_high_s32( t_QA1_s32x4 ), rc_mult2_s32x2 );
            t0_s64x2    = vrshlq_s64( t0_s64x2, mult2Q_s64x2 );
            t1_s64x2    = vrshlq_s64( t1_s64x2, mult2Q_s64x2 );
            t2_s64x2    = vrshlq_s64( t2_s64x2, mult2Q_s64x2 );
            t3_s64x2    = vrshlq_s64( t3_s64x2, mult2Q_s64x2 );
            t0_s32x4    = vcombine_s32( vmovn_s64( t0_s64x2 ), vmovn_s64( t1_s64x2 ) );
            t1_s32x4    = vcombine_s32( vmovn_s64( t2_s64x2 ), vmovn_s64( t3_s64x2 ) );
            s0_s32x4    = vcombine_s32( vshrn_n_s64( t0_s64x2, 31 ), vshrn_n_s64( t1_s64x2, 31 ) );
            s1_s32x4    = vcombine_s32( vshrn_n_s64( t2_s64x2, 31 ), vshrn_n_s64( t3_s64x2, 31 ) );
            max_s32x4   = vmaxq_s32( max_s32x4, s0_s32x4 );
            min_s32x4   = vminq_s32( min_s32x4, s0_s32x4 );
            max_s32x4   = vmaxq_s32( max_s32x4, s1_s32x4 );
            min_s32x4   = vminq_s32( min_s32x4, s1_s32x4 );
            t1_s32x4    = vrev64q_s32( t1_s32x4 );
            t1_s32x4    = vcombine_s32( vget_high_s32( t1_s32x4 ), vget_low_s32( t1_s32x4 ) );
            vst1q_s32( A_QA + n,         t0_s32x4 );
            vst1q_s32( A_QA + k - n - 4, t1_s32x4 );
        }
        for( ; n < (k + 1) >> 1; n++ ) {
            opus_int64 tmp64;
            tmp1 = A_QA[ n ];
            tmp2 = A_QA[ k - n - 1 ];
            tmp64 = silk_RSHIFT_ROUND64( silk_SMULL( silk_SUB_SAT32(tmp1,
                  MUL32_FRAC_Q( tmp2, rc_Q31, 31 ) ), rc_mult2 ), mult2Q);
            if( tmp64 > silk_int32_MAX || tmp64 < silk_int32_MIN ) {
               return 0;
            }
            A_QA[ n ] = ( opus_int32 )tmp64;
            tmp64 = silk_RSHIFT_ROUND64( silk_SMULL( silk_SUB_SAT32(tmp2,
                  MUL32_FRAC_Q( tmp1, rc_Q31, 31 ) ), rc_mult2), mult2Q);
            if( tmp64 > silk_int32_MAX || tmp64 < silk_int32_MIN ) {
               return 0;
            }
            A_QA[ k - n - 1 ] = ( opus_int32 )tmp64;
        }
    }

    /* Check for stability */
    if( ( A_QA[ k ] > A_LIMIT ) || ( A_QA[ k ] < -A_LIMIT ) ) {
        return 0;
    }

    max_s32x2 = vmax_s32( vget_low_s32( max_s32x4 ), vget_high_s32( max_s32x4 ) );
    min_s32x2 = vmin_s32( vget_low_s32( min_s32x4 ), vget_high_s32( min_s32x4 ) );
    max_s32x2 = vmax_s32( max_s32x2, vreinterpret_s32_s64( vshr_n_s64( vreinterpret_s64_s32( max_s32x2 ), 32 ) ) );
    min_s32x2 = vmin_s32( min_s32x2, vreinterpret_s32_s64( vshr_n_s64( vreinterpret_s64_s32( min_s32x2 ), 32 ) ) );
    max = vget_lane_s32( max_s32x2, 0 );
    min = vget_lane_s32( min_s32x2, 0 );
    if( ( max > 0 ) || ( min < -1 ) ) {
        return 0;
    }

    /* Set RC equal to negated AR coef */
    rc_Q31 = -silk_LSHIFT( A_QA[ 0 ], 31 - QA );

    /* Range: [ 1 : 2^30 ] */
    rc_mult1_Q30 = silk_SUB32( SILK_FIX_CONST( 1, 30 ), silk_SMMUL( rc_Q31, rc_Q31 ) );

    /* Update inverse gain */
    /* Range: [ 0 : 2^30 ] */
    invGain_Q30 = silk_LSHIFT( silk_SMMUL( invGain_Q30, rc_mult1_Q30 ), 2 );
    silk_assert( invGain_Q30 >= 0           );
    silk_assert( invGain_Q30 <= ( 1 << 30 ) );
    if( invGain_Q30 < SILK_FIX_CONST( 1.0f / MAX_PREDICTION_POWER_GAIN, 30 ) ) {
        return 0;
    }

    return invGain_Q30;
}

/* For input in Q12 domain */
opus_int32 silk_LPC_inverse_pred_gain_neon(         /* O   Returns inverse prediction gain in energy domain, Q30        */
    const opus_int16            *A_Q12,             /* I   Prediction coefficients, Q12 [order]                         */
    const opus_int              order               /* I   Prediction order                                             */
)
{
#ifdef OPUS_CHECK_ASM
    const opus_int32 invGain_Q30_c = silk_LPC_inverse_pred_gain_c( A_Q12, order );
#endif

    opus_int32 invGain_Q30;
    if( ( SILK_MAX_ORDER_LPC != 24 ) || ( order & 1 )) {
        invGain_Q30 = silk_LPC_inverse_pred_gain_c( A_Q12, order );
    }
    else {
        opus_int32 Atmp_QA[ SILK_MAX_ORDER_LPC ];
        opus_int32 DC_resp;
        int16x8_t  t0_s16x8, t1_s16x8, t2_s16x8;
        int32x4_t  t0_s32x4;
        const opus_int leftover = order & 7;

        /* Increase Q domain of the AR coefficients */
        t0_s16x8 = vld1q_s16( A_Q12 +  0 );
        t1_s16x8 = vld1q_s16( A_Q12 +  8 );
        t2_s16x8 = vld1q_s16( A_Q12 + 16 );
        t0_s32x4 = vpaddlq_s16( t0_s16x8 );

        switch( order - leftover )
        {
        case 24:
            t0_s32x4 = vpadalq_s16( t0_s32x4, t2_s16x8 );
            /* FALLTHROUGH */

        case 16:
            t0_s32x4 = vpadalq_s16( t0_s32x4, t1_s16x8 );
            vst1q_s32( Atmp_QA + 16, vshll_n_s16( vget_low_s16 ( t2_s16x8 ), QA - 12 ) );
            vst1q_s32( Atmp_QA + 20, vshll_n_s16( vget_high_s16( t2_s16x8 ), QA - 12 ) );
            /* FALLTHROUGH */

        case 8:
        {
            const int32x2_t t_s32x2 = vpadd_s32( vget_low_s32( t0_s32x4 ), vget_high_s32( t0_s32x4 ) );
            const int64x1_t t_s64x1 = vpaddl_s32( t_s32x2 );
            DC_resp = vget_lane_s32( vreinterpret_s32_s64( t_s64x1 ), 0 );
            vst1q_s32( Atmp_QA +  8, vshll_n_s16( vget_low_s16 ( t1_s16x8 ), QA - 12 ) );
            vst1q_s32( Atmp_QA + 12, vshll_n_s16( vget_high_s16( t1_s16x8 ), QA - 12 ) );
        }
        break;

        default:
            DC_resp = 0;
            break;
        }
        A_Q12 += order - leftover;

        switch( leftover )
        {
        case 6:
            DC_resp += (opus_int32)A_Q12[ 5 ];
            DC_resp += (opus_int32)A_Q12[ 4 ];
            /* FALLTHROUGH */

        case 4:
            DC_resp += (opus_int32)A_Q12[ 3 ];
            DC_resp += (opus_int32)A_Q12[ 2 ];
            /* FALLTHROUGH */

        case 2:
            DC_resp += (opus_int32)A_Q12[ 1 ];
            DC_resp += (opus_int32)A_Q12[ 0 ];
            /* FALLTHROUGH */

        default:
            break;
        }

        /* If the DC is unstable, we don't even need to do the full calculations */
        if( DC_resp >= 4096 ) {
            invGain_Q30 = 0;
        } else {
            vst1q_s32( Atmp_QA + 0, vshll_n_s16( vget_low_s16 ( t0_s16x8 ), QA - 12 ) );
            vst1q_s32( Atmp_QA + 4, vshll_n_s16( vget_high_s16( t0_s16x8 ), QA - 12 ) );
            invGain_Q30 = LPC_inverse_pred_gain_QA_neon( Atmp_QA, order );
        }
    }

#ifdef OPUS_CHECK_ASM
    silk_assert( invGain_Q30_c == invGain_Q30 );
#endif

    return invGain_Q30;
}
