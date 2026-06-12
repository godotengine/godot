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
#ifdef OPUS_CHECK_ASM
# include <string.h>
# include "stack_alloc.h"
#endif
#include "SigProc_FIX.h"

static inline void silk_biquad_alt_stride2_kernel( const int32x4_t A_L_s32x4, const int32x4_t A_U_s32x4, const int32x4_t B_Q28_s32x4, const int32x2_t t_s32x2, const int32x4_t in_s32x4, int32x4_t *S_s32x4, int32x2_t *out32_Q14_s32x2 )
{
    int32x4_t t_s32x4, out32_Q14_s32x4;

    *out32_Q14_s32x2 = vadd_s32( vget_low_s32( *S_s32x4 ), t_s32x2 );              /* silk_SMLAWB( S{0,1}, B_Q28[ 0 ], in{0,1} )                                      */
    *S_s32x4         = vcombine_s32( vget_high_s32( *S_s32x4 ), vdup_n_s32( 0 ) ); /* S{0,1} = S{2,3}; S{2,3} = 0;                                                    */
    *out32_Q14_s32x2 = vshl_n_s32( *out32_Q14_s32x2, 2 );                          /* out32_Q14_{0,1} = silk_LSHIFT( silk_SMLAWB( S{0,1}, B_Q28[ 0 ], in{0,1} ), 2 ); */
    out32_Q14_s32x4  = vcombine_s32( *out32_Q14_s32x2, *out32_Q14_s32x2 );         /* out32_Q14_{0,1,0,1}                                                             */
    t_s32x4          = vqdmulhq_s32( out32_Q14_s32x4, A_L_s32x4 );                 /* silk_SMULWB( out32_Q14_{0,1,0,1}, A{0,0,1,1}_L_Q28 )                            */
    *S_s32x4         = vrsraq_n_s32( *S_s32x4, t_s32x4, 14 );                      /* S{0,1} = S{2,3} + silk_RSHIFT_ROUND();  S{2,3} = silk_RSHIFT_ROUND();           */
    t_s32x4          = vqdmulhq_s32( out32_Q14_s32x4, A_U_s32x4 );                 /* silk_SMULWB( out32_Q14_{0,1,0,1}, A{0,0,1,1}_U_Q28 )                            */
    *S_s32x4         = vaddq_s32( *S_s32x4, t_s32x4 );                             /* S0 = silk_SMLAWB( S{0,1,2,3}, out32_Q14_{0,1,0,1}, A{0,0,1,1}_U_Q28 );          */
    t_s32x4          = vqdmulhq_s32( in_s32x4, B_Q28_s32x4 );                      /* silk_SMULWB( B_Q28[ {1,1,2,2} ], in{0,1,0,1} )                                  */
    *S_s32x4         = vaddq_s32( *S_s32x4, t_s32x4 );                             /* S0 = silk_SMLAWB( S0, B_Q28[ {1,1,2,2} ], in{0,1,0,1} );                        */
}

void silk_biquad_alt_stride2_neon(
    const opus_int16            *in,                /* I     input signal                                               */
    const opus_int32            *B_Q28,             /* I     MA coefficients [3]                                        */
    const opus_int32            *A_Q28,             /* I     AR coefficients [2]                                        */
    opus_int32                  *S,                 /* I/O   State vector [4]                                           */
    opus_int16                  *out,               /* O     output signal                                              */
    const opus_int32            len                 /* I     signal length (must be even)                               */
)
{
    /* DIRECT FORM II TRANSPOSED (uses 2 element state vector) */
    opus_int        k            = 0;
    const int32x2_t offset_s32x2 = vdup_n_s32( (1<<14) - 1 );
    const int32x4_t offset_s32x4 = vcombine_s32( offset_s32x2, offset_s32x2 );
    int16x4_t       in_s16x4  = vdup_n_s16( 0 );
    int16x4_t       out_s16x4;
    int32x2_t       A_Q28_s32x2, A_L_s32x2, A_U_s32x2, B_Q28_s32x2, t_s32x2;
    int32x4_t       A_L_s32x4, A_U_s32x4, B_Q28_s32x4, S_s32x4, out32_Q14_s32x4;
    int32x2x2_t     t0_s32x2x2, t1_s32x2x2, t2_s32x2x2, S_s32x2x2;

#ifdef OPUS_CHECK_ASM
    opus_int32 S_c[ 4 ];
    VARDECL( opus_int16, out_c );
    SAVE_STACK;
    ALLOC( out_c, 2 * len, opus_int16 );

    silk_memcpy( &S_c, S, sizeof( S_c ) );
    silk_biquad_alt_stride2_c( in, B_Q28, A_Q28, S_c, out_c, len );
#endif

    /* Negate A_Q28 values and split in two parts */
    A_Q28_s32x2 = vld1_s32( A_Q28 );
    A_Q28_s32x2 = vneg_s32( A_Q28_s32x2 );
    A_L_s32x2   = vshl_n_s32( A_Q28_s32x2, 18 );                                                        /* ( -A_Q28[] & 0x00003FFF ) << 18                                                     */
    A_L_s32x2   = vreinterpret_s32_u32( vshr_n_u32( vreinterpret_u32_s32( A_L_s32x2 ), 3 ) );           /* ( -A_Q28[] & 0x00003FFF ) << 15                                                     */
    A_U_s32x2   = vshr_n_s32( A_Q28_s32x2, 14 );                                                        /* silk_RSHIFT( -A_Q28[], 14 )                                                         */
    A_U_s32x2   = vshl_n_s32( A_U_s32x2, 16 );                                                          /* silk_RSHIFT( -A_Q28[], 14 ) << 16 (Clip two leading bits to conform to C function.) */
    A_U_s32x2   = vshr_n_s32( A_U_s32x2, 1 );                                                           /* silk_RSHIFT( -A_Q28[], 14 ) << 15                                                   */

    B_Q28_s32x2  = vld1_s32( B_Q28 );
    t_s32x2      = vld1_s32( B_Q28 + 1 );
    t0_s32x2x2   = vzip_s32( A_L_s32x2, A_L_s32x2 );
    t1_s32x2x2   = vzip_s32( A_U_s32x2, A_U_s32x2 );
    t2_s32x2x2   = vzip_s32( t_s32x2, t_s32x2 );
    A_L_s32x4    = vcombine_s32( t0_s32x2x2.val[ 0 ], t0_s32x2x2.val[ 1 ] );                            /* A{0,0,1,1}_L_Q28          */
    A_U_s32x4    = vcombine_s32( t1_s32x2x2.val[ 0 ], t1_s32x2x2.val[ 1 ] );                            /* A{0,0,1,1}_U_Q28          */
    B_Q28_s32x4  = vcombine_s32( t2_s32x2x2.val[ 0 ], t2_s32x2x2.val[ 1 ] );                            /* B_Q28[ {1,1,2,2} ]        */
    S_s32x4      = vld1q_s32( S );                                                                      /* S0 = S[ 0 ]; S3 = S[ 3 ]; */
    S_s32x2x2    = vtrn_s32( vget_low_s32( S_s32x4 ), vget_high_s32( S_s32x4 ) );                       /* S2 = S[ 1 ]; S1 = S[ 2 ]; */
    S_s32x4      = vcombine_s32( S_s32x2x2.val[ 0 ], S_s32x2x2.val[ 1 ] );

    for( ; k < len - 1; k += 2 ) {
        int32x4_t in_s32x4[ 2 ], t_s32x4;
        int32x2_t out32_Q14_s32x2[ 2 ];

        /* S[ 2 * i + 0 ], S[ 2 * i + 1 ], S[ 2 * i + 2 ], S[ 2 * i + 3 ]: Q12 */
        in_s16x4      = vld1_s16( &in[ 2 * k ] );                                                       /* in{0,1,2,3} = in[ 2 * k + {0,1,2,3} ]; */
        in_s32x4[ 0 ] = vshll_n_s16( in_s16x4, 15 );                                                    /* in{0,1,2,3} << 15                      */
        t_s32x4       = vqdmulhq_lane_s32( in_s32x4[ 0 ], B_Q28_s32x2, 0 );                             /* silk_SMULWB( B_Q28[ 0 ], in{0,1,2,3} ) */
        in_s32x4[ 1 ] = vcombine_s32( vget_high_s32( in_s32x4[ 0 ] ), vget_high_s32( in_s32x4[ 0 ] ) ); /* in{2,3,2,3} << 15                      */
        in_s32x4[ 0 ] = vcombine_s32( vget_low_s32 ( in_s32x4[ 0 ] ), vget_low_s32 ( in_s32x4[ 0 ] ) ); /* in{0,1,0,1} << 15                      */
        silk_biquad_alt_stride2_kernel( A_L_s32x4, A_U_s32x4, B_Q28_s32x4, vget_low_s32 ( t_s32x4 ), in_s32x4[ 0 ], &S_s32x4, &out32_Q14_s32x2[ 0 ] );
        silk_biquad_alt_stride2_kernel( A_L_s32x4, A_U_s32x4, B_Q28_s32x4, vget_high_s32( t_s32x4 ), in_s32x4[ 1 ], &S_s32x4, &out32_Q14_s32x2[ 1 ] );

        /* Scale back to Q0 and saturate */
        out32_Q14_s32x4 = vcombine_s32( out32_Q14_s32x2[ 0 ], out32_Q14_s32x2[ 1 ] );                   /* out32_Q14_{0,1,2,3}                                                                                        */
        out32_Q14_s32x4 = vaddq_s32( out32_Q14_s32x4, offset_s32x4 );                                   /* out32_Q14_{0,1,2,3} + (1<<14) - 1                                                                          */
        out_s16x4       = vqshrn_n_s32( out32_Q14_s32x4, 14 );                                          /* (opus_int16)silk_SAT16( silk_RSHIFT( out32_Q14_{0,1,2,3} + (1<<14) - 1, 14 ) )                             */
        vst1_s16( &out[ 2 * k ], out_s16x4 );                                                           /* out[ 2 * k + {0,1,2,3} ] = (opus_int16)silk_SAT16( silk_RSHIFT( out32_Q14_{0,1,2,3} + (1<<14) - 1, 14 ) ); */
    }

    /* Process leftover. */
    if( k < len ) {
        int32x4_t in_s32x4;
        int32x2_t out32_Q14_s32x2;

        /* S[ 2 * i + 0 ], S[ 2 * i + 1 ]: Q12 */
        in_s16x4     = vld1_lane_s16( &in[ 2 * k + 0 ], in_s16x4, 0 );                                  /* in{0,1} = in[ 2 * k + {0,1} ];     */
        in_s16x4     = vld1_lane_s16( &in[ 2 * k + 1 ], in_s16x4, 1 );                                  /* in{0,1} = in[ 2 * k + {0,1} ];     */
        in_s32x4     = vshll_n_s16( in_s16x4, 15 );                                                     /* in{0,1} << 15                      */
        t_s32x2      = vqdmulh_lane_s32( vget_low_s32( in_s32x4 ), B_Q28_s32x2, 0 );                    /* silk_SMULWB( B_Q28[ 0 ], in{0,1} ) */
        in_s32x4     = vcombine_s32( vget_low_s32( in_s32x4 ), vget_low_s32( in_s32x4 ) );              /* in{0,1,0,1} << 15                  */
        silk_biquad_alt_stride2_kernel( A_L_s32x4, A_U_s32x4, B_Q28_s32x4, t_s32x2, in_s32x4, &S_s32x4, &out32_Q14_s32x2 );

        /* Scale back to Q0 and saturate */
        out32_Q14_s32x2 = vadd_s32( out32_Q14_s32x2, offset_s32x2 );                                    /* out32_Q14_{0,1} + (1<<14) - 1                                                              */
        out32_Q14_s32x4 = vcombine_s32( out32_Q14_s32x2, out32_Q14_s32x2 );                             /* out32_Q14_{0,1,0,1} + (1<<14) - 1                                                          */
        out_s16x4       = vqshrn_n_s32( out32_Q14_s32x4, 14 );                                          /* (opus_int16)silk_SAT16( silk_RSHIFT( out32_Q14_{0,1,0,1} + (1<<14) - 1, 14 ) )             */
        vst1_lane_s16( &out[ 2 * k + 0 ], out_s16x4, 0 );                                               /* out[ 2 * k + 0 ] = (opus_int16)silk_SAT16( silk_RSHIFT( out32_Q14_0 + (1<<14) - 1, 14 ) ); */
        vst1_lane_s16( &out[ 2 * k + 1 ], out_s16x4, 1 );                                               /* out[ 2 * k + 1 ] = (opus_int16)silk_SAT16( silk_RSHIFT( out32_Q14_1 + (1<<14) - 1, 14 ) ); */
    }

    vst1q_lane_s32( &S[ 0 ], S_s32x4, 0 );                                                              /* S[ 0 ] = S0; */
    vst1q_lane_s32( &S[ 1 ], S_s32x4, 2 );                                                              /* S[ 1 ] = S2; */
    vst1q_lane_s32( &S[ 2 ], S_s32x4, 1 );                                                              /* S[ 2 ] = S1; */
    vst1q_lane_s32( &S[ 3 ], S_s32x4, 3 );                                                              /* S[ 3 ] = S3; */

#ifdef OPUS_CHECK_ASM
    silk_assert( !memcmp( S_c, S, sizeof( S_c ) ) );
    silk_assert( !memcmp( out_c, out, 2 * len * sizeof( opus_int16 ) ) );
    RESTORE_STACK;
#endif
}
