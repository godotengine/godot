/***********************************************************************
Copyright (c) 2017 Google Inc., Jean-Marc Valin
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
#ifdef OPUS_CHECK_ASM
# include <string.h>
#endif
#include "stack_alloc.h"
#include "main_FIX.h"

static OPUS_INLINE void calc_corr( const opus_int32 *const input_QS, opus_int64 *const corr_QC, const opus_int offset, const int32x4_t state_QS_s32x4 )
{
    int64x2_t corr_QC_s64x2[ 2 ], t_s64x2[ 2 ];
    const int32x4_t input_QS_s32x4 = vld1q_s32( input_QS + offset );
    corr_QC_s64x2[ 0 ] = vld1q_s64( corr_QC + offset + 0 );
    corr_QC_s64x2[ 1 ] = vld1q_s64( corr_QC + offset + 2 );
    t_s64x2[ 0 ] = vmull_s32( vget_low_s32( state_QS_s32x4 ), vget_low_s32( input_QS_s32x4 ) );
    t_s64x2[ 1 ] = vmull_s32( vget_high_s32( state_QS_s32x4 ), vget_high_s32( input_QS_s32x4 ) );
    corr_QC_s64x2[ 0 ] = vsraq_n_s64( corr_QC_s64x2[ 0 ], t_s64x2[ 0 ], 2 * QS - QC );
    corr_QC_s64x2[ 1 ] = vsraq_n_s64( corr_QC_s64x2[ 1 ], t_s64x2[ 1 ], 2 * QS - QC );
    vst1q_s64( corr_QC + offset + 0, corr_QC_s64x2[ 0 ] );
    vst1q_s64( corr_QC + offset + 2, corr_QC_s64x2[ 1 ] );
}

static OPUS_INLINE int32x4_t calc_state( const int32x4_t state_QS0_s32x4, const int32x4_t state_QS0_1_s32x4, const int32x4_t state_QS1_1_s32x4, const int32x4_t warping_Q16_s32x4 )
{
    int32x4_t t_s32x4 = vsubq_s32( state_QS0_s32x4, state_QS0_1_s32x4 );
    t_s32x4 = vqdmulhq_s32( t_s32x4, warping_Q16_s32x4 );
    return vaddq_s32( state_QS1_1_s32x4, t_s32x4 );
}

void silk_warped_autocorrelation_FIX_neon(
          opus_int32                *corr,                                  /* O    Result [order + 1]                                                          */
          opus_int                  *scale,                                 /* O    Scaling of the correlation vector                                           */
    const opus_int16                *input,                                 /* I    Input data to correlate                                                     */
    const opus_int                  warping_Q16,                            /* I    Warping coefficient                                                         */
    const opus_int                  length,                                 /* I    Length of input                                                             */
    const opus_int                  order                                   /* I    Correlation order (even)                                                    */
)
{
    if( ( MAX_SHAPE_LPC_ORDER > 24 ) || ( order < 6 ) ) {
        silk_warped_autocorrelation_FIX_c( corr, scale, input, warping_Q16, length, order );
    } else {
        opus_int       n, i, lsh;
        opus_int64     corr_QC[ MAX_SHAPE_LPC_ORDER + 1 ] = { 0 }; /* In reverse order */
        opus_int64     corr_QC_orderT;
        int64x2_t      lsh_s64x2;
        const opus_int orderT = ( order + 3 ) & ~3;
        opus_int64     *corr_QCT;
        opus_int32     *input_QS;
        VARDECL( opus_int32, input_QST );
        VARDECL( opus_int32, state );
        SAVE_STACK;

        /* Order must be even */
        silk_assert( ( order & 1 ) == 0 );
        silk_assert( 2 * QS - QC >= 0 );

        ALLOC( input_QST, length + 2 * MAX_SHAPE_LPC_ORDER, opus_int32 );

        input_QS = input_QST;
        /* input_QS has zero paddings in the beginning and end. */
        vst1q_s32( input_QS, vdupq_n_s32( 0 ) );
        input_QS += 4;
        vst1q_s32( input_QS, vdupq_n_s32( 0 ) );
        input_QS += 4;
        vst1q_s32( input_QS, vdupq_n_s32( 0 ) );
        input_QS += 4;
        vst1q_s32( input_QS, vdupq_n_s32( 0 ) );
        input_QS += 4;
        vst1q_s32( input_QS, vdupq_n_s32( 0 ) );
        input_QS += 4;
        vst1q_s32( input_QS, vdupq_n_s32( 0 ) );
        input_QS += 4;

        /* Loop over samples */
        for( n = 0; n < length - 7; n += 8, input_QS += 8 ) {
            const int16x8_t t0_s16x4 = vld1q_s16( input + n );
            vst1q_s32( input_QS + 0, vshll_n_s16( vget_low_s16( t0_s16x4 ), QS ) );
            vst1q_s32( input_QS + 4, vshll_n_s16( vget_high_s16( t0_s16x4 ), QS ) );
        }
        for( ; n < length; n++, input_QS++ ) {
            input_QS[ 0 ] = silk_LSHIFT32( (opus_int32)input[ n ], QS );
        }
        vst1q_s32( input_QS, vdupq_n_s32( 0 ) );
        input_QS += 4;
        vst1q_s32( input_QS, vdupq_n_s32( 0 ) );
        input_QS += 4;
        vst1q_s32( input_QS, vdupq_n_s32( 0 ) );
        input_QS += 4;
        vst1q_s32( input_QS, vdupq_n_s32( 0 ) );
        input_QS += 4;
        vst1q_s32( input_QS, vdupq_n_s32( 0 ) );
        input_QS += 4;
        vst1q_s32( input_QS, vdupq_n_s32( 0 ) );
        input_QS = input_QST + MAX_SHAPE_LPC_ORDER - orderT;

        /* The following loop runs ( length + order ) times, with ( order ) extra epilogues.                  */
        /* The zero paddings in input_QS guarantee corr_QC's correctness even with the extra epilogues.       */
        /* The values of state_QS will be polluted by the extra epilogues, however they are temporary values. */

        /* Keep the C code here to help understand the intrinsics optimization. */
        /*
        {
            opus_int32 state_QS[ 2 ][ MAX_SHAPE_LPC_ORDER + 1 ] = { 0 };
            opus_int32 *state_QST[ 3 ];
            state_QST[ 0 ] = state_QS[ 0 ];
            state_QST[ 1 ] = state_QS[ 1 ];
            for( n = 0; n < length + order; n++, input_QS++ ) {
                state_QST[ 0 ][ orderT ] = input_QS[ orderT ];
                for( i = 0; i < orderT; i++ ) {
                    corr_QC[ i ] += silk_RSHIFT64( silk_SMULL( state_QST[ 0 ][ i ], input_QS[ i ] ), 2 * QS - QC );
                    state_QST[ 1 ][ i ] = silk_SMLAWB( state_QST[ 1 ][ i + 1 ], state_QST[ 0 ][ i ] - state_QST[ 0 ][ i + 1 ], warping_Q16 );
                }
                state_QST[ 2 ] = state_QST[ 0 ];
                state_QST[ 0 ] = state_QST[ 1 ];
                state_QST[ 1 ] = state_QST[ 2 ];
            }
        }
        */

        {
            const int32x4_t warping_Q16_s32x4 = vdupq_n_s32( warping_Q16 << 15 );
            const opus_int32 *in = input_QS + orderT;
            opus_int o = orderT;
            int32x4_t state_QS_s32x4[ 3 ][ 2 ];

            ALLOC( state, length + orderT, opus_int32 );
            state_QS_s32x4[ 2 ][ 1 ] = vdupq_n_s32( 0 );

            /* Calculate 8 taps of all inputs in each loop. */
            do {
                state_QS_s32x4[ 0 ][ 0 ] = state_QS_s32x4[ 0 ][ 1 ] =
                state_QS_s32x4[ 1 ][ 0 ] = state_QS_s32x4[ 1 ][ 1 ] = vdupq_n_s32( 0 );
                n = 0;
                do {
                    calc_corr( input_QS + n, corr_QC, o - 8, state_QS_s32x4[ 0 ][ 0 ] );
                    calc_corr( input_QS + n, corr_QC, o - 4, state_QS_s32x4[ 0 ][ 1 ] );
                    state_QS_s32x4[ 2 ][ 1 ] = vld1q_s32( in + n );
                    vst1q_lane_s32( state + n, state_QS_s32x4[ 0 ][ 0 ], 0 );
                    state_QS_s32x4[ 2 ][ 0 ] = vextq_s32( state_QS_s32x4[ 0 ][ 0 ], state_QS_s32x4[ 0 ][ 1 ], 1 );
                    state_QS_s32x4[ 2 ][ 1 ] = vextq_s32( state_QS_s32x4[ 0 ][ 1 ], state_QS_s32x4[ 2 ][ 1 ], 1 );
                    state_QS_s32x4[ 0 ][ 0 ] = calc_state( state_QS_s32x4[ 0 ][ 0 ], state_QS_s32x4[ 2 ][ 0 ], state_QS_s32x4[ 1 ][ 0 ], warping_Q16_s32x4 );
                    state_QS_s32x4[ 0 ][ 1 ] = calc_state( state_QS_s32x4[ 0 ][ 1 ], state_QS_s32x4[ 2 ][ 1 ], state_QS_s32x4[ 1 ][ 1 ], warping_Q16_s32x4 );
                    state_QS_s32x4[ 1 ][ 0 ] = state_QS_s32x4[ 2 ][ 0 ];
                    state_QS_s32x4[ 1 ][ 1 ] = state_QS_s32x4[ 2 ][ 1 ];
                } while( ++n < ( length + order ) );
                in = state;
                o -= 8;
            } while( o > 4 );

            if( o ) {
                /* Calculate the last 4 taps of all inputs. */
                opus_int32 *stateT = state;
                silk_assert( o == 4 );
                state_QS_s32x4[ 0 ][ 0 ] = state_QS_s32x4[ 1 ][ 0 ] = vdupq_n_s32( 0 );
                n = length + order;
                do {
                    calc_corr( input_QS, corr_QC, 0, state_QS_s32x4[ 0 ][ 0 ] );
                    state_QS_s32x4[ 2 ][ 0 ] = vld1q_s32( stateT );
                    vst1q_lane_s32( stateT, state_QS_s32x4[ 0 ][ 0 ], 0 );
                    state_QS_s32x4[ 2 ][ 0 ] = vextq_s32( state_QS_s32x4[ 0 ][ 0 ], state_QS_s32x4[ 2 ][ 0 ], 1 );
                    state_QS_s32x4[ 0 ][ 0 ] = calc_state( state_QS_s32x4[ 0 ][ 0 ], state_QS_s32x4[ 2 ][ 0 ], state_QS_s32x4[ 1 ][ 0 ], warping_Q16_s32x4 );
                    state_QS_s32x4[ 1 ][ 0 ] = state_QS_s32x4[ 2 ][ 0 ];
                    input_QS++;
                    stateT++;
                } while( --n );
            }
        }

        {
            const opus_int16 *inputT = input;
            int32x4_t t_s32x4;
            int64x1_t t_s64x1;
            int64x2_t t_s64x2 = vdupq_n_s64( 0 );
            for( n = 0; n <= length - 8; n += 8 ) {
                int16x8_t input_s16x8 = vld1q_s16( inputT );
                t_s32x4 = vmull_s16( vget_low_s16( input_s16x8 ), vget_low_s16( input_s16x8 ) );
                t_s32x4 = vmlal_s16( t_s32x4, vget_high_s16( input_s16x8 ), vget_high_s16( input_s16x8 ) );
                t_s64x2 = vaddw_s32( t_s64x2, vget_low_s32( t_s32x4 ) );
                t_s64x2 = vaddw_s32( t_s64x2, vget_high_s32( t_s32x4 ) );
                inputT += 8;
            }
            t_s64x1 = vadd_s64( vget_low_s64( t_s64x2 ), vget_high_s64( t_s64x2 ) );
            corr_QC_orderT = vget_lane_s64( t_s64x1, 0 );
            for( ; n < length; n++ ) {
                corr_QC_orderT += silk_SMULL( input[ n ], input[ n ] );
            }
            corr_QC_orderT = silk_LSHIFT64( corr_QC_orderT, QC );
            corr_QC[ orderT ] = corr_QC_orderT;
        }

        corr_QCT = corr_QC + orderT - order;
        lsh = silk_CLZ64( corr_QC_orderT ) - 35;
        lsh = silk_LIMIT( lsh, -12 - QC, 30 - QC );
        *scale = -( QC + lsh );
        silk_assert( *scale >= -30 && *scale <= 12 );
        lsh_s64x2 = vdupq_n_s64( lsh );
        for( i = 0; i <= order - 3; i += 4 ) {
            int32x4_t corr_s32x4;
            int64x2_t corr_QC0_s64x2, corr_QC1_s64x2;
            corr_QC0_s64x2 = vld1q_s64( corr_QCT + i );
            corr_QC1_s64x2 = vld1q_s64( corr_QCT + i + 2 );
            corr_QC0_s64x2 = vshlq_s64( corr_QC0_s64x2, lsh_s64x2 );
            corr_QC1_s64x2 = vshlq_s64( corr_QC1_s64x2, lsh_s64x2 );
            corr_s32x4     = vcombine_s32( vmovn_s64( corr_QC1_s64x2 ), vmovn_s64( corr_QC0_s64x2 ) );
            corr_s32x4     = vrev64q_s32( corr_s32x4 );
            vst1q_s32( corr + order - i - 3, corr_s32x4 );
        }
        if( lsh >= 0 ) {
            for( ; i < order + 1; i++ ) {
                corr[ order - i ] = (opus_int32)silk_CHECK_FIT32( silk_LSHIFT64( corr_QCT[ i ], lsh ) );
            }
        } else {
            for( ; i < order + 1; i++ ) {
                corr[ order - i ] = (opus_int32)silk_CHECK_FIT32( silk_RSHIFT64( corr_QCT[ i ], -lsh ) );
            }
        }
        silk_assert( corr_QCT[ order ] >= 0 ); /* If breaking, decrease QC*/
        RESTORE_STACK;
    }

#ifdef OPUS_CHECK_ASM
    {
        opus_int32 corr_c[ MAX_SHAPE_LPC_ORDER + 1 ];
        opus_int   scale_c;
        silk_warped_autocorrelation_FIX_c( corr_c, &scale_c, input, warping_Q16, length, order );
        silk_assert( !memcmp( corr_c, corr, sizeof( corr_c[ 0 ] ) * ( order + 1 ) ) );
        silk_assert( scale_c == *scale );
    }
#endif
}
