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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "SigProc_FIX.h"
#include "define.h"

#define QA                          24
#define A_LIMIT                     SILK_FIX_CONST( 0.99975, QA )

#define MUL32_FRAC_Q(a32, b32, Q)   ((opus_int32)(silk_RSHIFT_ROUND64(silk_SMULL(a32, b32), Q)))

/* Compute inverse of LPC prediction gain, and                          */
/* test if LPC coefficients are stable (all poles within unit circle)   */
static opus_int32 LPC_inverse_pred_gain_QA_c(               /* O   Returns inverse prediction gain in energy domain, Q30    */
    opus_int32           A_QA[ SILK_MAX_ORDER_LPC ],        /* I   Prediction coefficients                                  */
    const opus_int       order                              /* I   Prediction order                                         */
)
{
    opus_int   k, n, mult2Q;
    opus_int32 invGain_Q30, rc_Q31, rc_mult1_Q30, rc_mult2, tmp1, tmp2;

    invGain_Q30 = SILK_FIX_CONST( 1, 30 );
    for( k = order - 1; k > 0; k-- ) {
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
        for( n = 0; n < (k + 1) >> 1; n++ ) {
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
opus_int32 silk_LPC_inverse_pred_gain_c(            /* O   Returns inverse prediction gain in energy domain, Q30        */
    const opus_int16            *A_Q12,             /* I   Prediction coefficients, Q12 [order]                         */
    const opus_int              order               /* I   Prediction order                                             */
)
{
    opus_int   k;
    opus_int32 Atmp_QA[ SILK_MAX_ORDER_LPC ];
    opus_int32 DC_resp = 0;

    /* Increase Q domain of the AR coefficients */
    for( k = 0; k < order; k++ ) {
        DC_resp += (opus_int32)A_Q12[ k ];
        Atmp_QA[ k ] = silk_LSHIFT32( (opus_int32)A_Q12[ k ], QA - 12 );
    }
    /* If the DC is unstable, we don't even need to do the full calculations */
    if( DC_resp >= 4096 ) {
        return 0;
    }
    return LPC_inverse_pred_gain_QA_c( Atmp_QA, order );
}
