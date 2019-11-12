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

#ifndef __WARPED_AUTOCORRELATION_FIX_MIPSR1_H__
#define __WARPED_AUTOCORRELATION_FIX_MIPSR1_H__

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "main_FIX.h"

#undef QC
#define QC  10

#undef QS
#define QS  14

/* Autocorrelations for a warped frequency axis */
#define OVERRIDE_silk_warped_autocorrelation_FIX_c
void silk_warped_autocorrelation_FIX_c(
          opus_int32                *corr,                                  /* O    Result [order + 1]                                                          */
          opus_int                  *scale,                                 /* O    Scaling of the correlation vector                                           */
    const opus_int16                *input,                                 /* I    Input data to correlate                                                     */
    const opus_int                  warping_Q16,                            /* I    Warping coefficient                                                         */
    const opus_int                  length,                                 /* I    Length of input                                                             */
    const opus_int                  order                                   /* I    Correlation order (even)                                                    */
)
{
    opus_int   n, i, lsh;
    opus_int32 tmp1_QS=0, tmp2_QS=0, tmp3_QS=0, tmp4_QS=0, tmp5_QS=0, tmp6_QS=0, tmp7_QS=0, tmp8_QS=0, start_1=0, start_2=0, start_3=0;
    opus_int32 state_QS[ MAX_SHAPE_LPC_ORDER + 1 ] = { 0 };
    opus_int64 corr_QC[  MAX_SHAPE_LPC_ORDER + 1 ] = { 0 };
    opus_int64 temp64;

    opus_int32 val;
    val = 2 * QS - QC;

    /* Order must be even */
    silk_assert( ( order & 1 ) == 0 );
    silk_assert( 2 * QS - QC >= 0 );

    /* Loop over samples */
    for( n = 0; n < length; n=n+4 ) {

        tmp1_QS = silk_LSHIFT32( (opus_int32)input[ n ], QS );
        start_1 = tmp1_QS;
        tmp3_QS = silk_LSHIFT32( (opus_int32)input[ n+1], QS );
        start_2 = tmp3_QS;
        tmp5_QS = silk_LSHIFT32( (opus_int32)input[ n+2], QS );
        start_3 = tmp5_QS;
        tmp7_QS = silk_LSHIFT32( (opus_int32)input[ n+3], QS );

        /* Loop over allpass sections */
        for( i = 0; i < order; i += 2 ) {
            /* Output of allpass section */
            tmp2_QS = silk_SMLAWB( state_QS[ i ], state_QS[ i + 1 ] - tmp1_QS, warping_Q16 );
            corr_QC[  i ] = __builtin_mips_madd( corr_QC[  i ], tmp1_QS,  start_1);

            tmp4_QS = silk_SMLAWB( tmp1_QS, tmp2_QS - tmp3_QS, warping_Q16 );
            corr_QC[  i ] = __builtin_mips_madd( corr_QC[  i ], tmp3_QS,  start_2);

            tmp6_QS = silk_SMLAWB( tmp3_QS, tmp4_QS - tmp5_QS, warping_Q16 );
            corr_QC[  i ] = __builtin_mips_madd( corr_QC[  i ], tmp5_QS,  start_3);

            tmp8_QS = silk_SMLAWB( tmp5_QS, tmp6_QS - tmp7_QS, warping_Q16 );
            state_QS[ i ]  = tmp7_QS;
            corr_QC[  i ] = __builtin_mips_madd( corr_QC[  i ], tmp7_QS, state_QS[0]);

            /* Output of allpass section */
            tmp1_QS = silk_SMLAWB( state_QS[ i + 1 ], state_QS[ i + 2 ] - tmp2_QS, warping_Q16 );
            corr_QC[  i+1 ] = __builtin_mips_madd( corr_QC[  i+1 ], tmp2_QS,  start_1);

            tmp3_QS = silk_SMLAWB( tmp2_QS, tmp1_QS - tmp4_QS, warping_Q16 );
            corr_QC[  i+1 ] = __builtin_mips_madd( corr_QC[  i+1 ], tmp4_QS,  start_2);

            tmp5_QS = silk_SMLAWB( tmp4_QS, tmp3_QS - tmp6_QS, warping_Q16 );
            corr_QC[  i+1 ] = __builtin_mips_madd( corr_QC[  i+1 ], tmp6_QS,  start_3);

            tmp7_QS = silk_SMLAWB( tmp6_QS, tmp5_QS - tmp8_QS, warping_Q16 );
            state_QS[ i + 1 ]  = tmp8_QS;
            corr_QC[  i+1 ] = __builtin_mips_madd( corr_QC[  i+1 ], tmp8_QS,  state_QS[ 0 ]);

        }
        state_QS[ order ] = tmp7_QS;

        corr_QC[  order ] = __builtin_mips_madd( corr_QC[  order ], tmp1_QS,  start_1);
        corr_QC[  order ] = __builtin_mips_madd( corr_QC[  order ], tmp3_QS,  start_2);
        corr_QC[  order ] = __builtin_mips_madd( corr_QC[  order ], tmp5_QS,  start_3);
        corr_QC[  order ] = __builtin_mips_madd( corr_QC[  order ], tmp7_QS,  state_QS[ 0 ]);
    }

    for(;n< length; n++ ) {

        tmp1_QS = silk_LSHIFT32( (opus_int32)input[ n ], QS );

        /* Loop over allpass sections */
        for( i = 0; i < order; i += 2 ) {

            /* Output of allpass section */
            tmp2_QS = silk_SMLAWB( state_QS[ i ], state_QS[ i + 1 ] - tmp1_QS, warping_Q16 );
            state_QS[ i ] = tmp1_QS;
            corr_QC[  i ] = __builtin_mips_madd( corr_QC[  i ], tmp1_QS,   state_QS[ 0 ]);

            /* Output of allpass section */
            tmp1_QS = silk_SMLAWB( state_QS[ i + 1 ], state_QS[ i + 2 ] - tmp2_QS, warping_Q16 );
            state_QS[ i + 1 ]  = tmp2_QS;
            corr_QC[  i+1 ] = __builtin_mips_madd( corr_QC[  i+1 ], tmp2_QS,   state_QS[ 0 ]);
        }
        state_QS[ order ] = tmp1_QS;
        corr_QC[  order ] = __builtin_mips_madd( corr_QC[  order ], tmp1_QS,   state_QS[ 0 ]);
    }

    temp64 =  corr_QC[ 0 ];
    temp64 = __builtin_mips_shilo(temp64, val);

    lsh = silk_CLZ64( temp64 ) - 35;
    lsh = silk_LIMIT( lsh, -12 - QC, 30 - QC );
    *scale = -( QC + lsh );
    silk_assert( *scale >= -30 && *scale <= 12 );
    if( lsh >= 0 ) {
        for( i = 0; i < order + 1; i++ ) {
            temp64 = corr_QC[ i ];
            //temp64 = __builtin_mips_shilo(temp64, val);
            temp64 = (val >= 0) ? (temp64 >> val) : (temp64 << -val);
            corr[ i ] = (opus_int32)silk_CHECK_FIT32( __builtin_mips_shilo( temp64, -lsh ) );
        }
    } else {
        for( i = 0; i < order + 1; i++ ) {
            temp64 = corr_QC[ i ];
            //temp64 = __builtin_mips_shilo(temp64, val);
            temp64 = (val >= 0) ? (temp64 >> val) : (temp64 << -val);
            corr[ i ] = (opus_int32)silk_CHECK_FIT32( __builtin_mips_shilo( temp64, -lsh ) );
        }
    }

     corr_QC[ 0 ] = __builtin_mips_shilo(corr_QC[ 0 ], val);

     silk_assert( corr_QC[ 0 ] >= 0 ); /* If breaking, decrease QC*/
}
#endif /* __WARPED_AUTOCORRELATION_FIX_MIPSR1_H__ */
