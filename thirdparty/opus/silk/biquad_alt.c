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

/*                                                                      *
 * silk_biquad_alt.c                                              *
 *                                                                      *
 * Second order ARMA filter                                             *
 * Can handle slowly varying filter coefficients                        *
 *                                                                      */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "SigProc_FIX.h"

/* Second order ARMA filter, alternative implementation */
void silk_biquad_alt(
    const opus_int16            *in,                /* I     input signal                                               */
    const opus_int32            *B_Q28,             /* I     MA coefficients [3]                                        */
    const opus_int32            *A_Q28,             /* I     AR coefficients [2]                                        */
    opus_int32                  *S,                 /* I/O   State vector [2]                                           */
    opus_int16                  *out,               /* O     output signal                                              */
    const opus_int32            len,                /* I     signal length (must be even)                               */
    opus_int                    stride              /* I     Operate on interleaved signal if > 1                       */
)
{
    /* DIRECT FORM II TRANSPOSED (uses 2 element state vector) */
    opus_int   k;
    opus_int32 inval, A0_U_Q28, A0_L_Q28, A1_U_Q28, A1_L_Q28, out32_Q14;

    /* Negate A_Q28 values and split in two parts */
    A0_L_Q28 = ( -A_Q28[ 0 ] ) & 0x00003FFF;        /* lower part */
    A0_U_Q28 = silk_RSHIFT( -A_Q28[ 0 ], 14 );      /* upper part */
    A1_L_Q28 = ( -A_Q28[ 1 ] ) & 0x00003FFF;        /* lower part */
    A1_U_Q28 = silk_RSHIFT( -A_Q28[ 1 ], 14 );      /* upper part */

    for( k = 0; k < len; k++ ) {
        /* S[ 0 ], S[ 1 ]: Q12 */
        inval = in[ k * stride ];
        out32_Q14 = silk_LSHIFT( silk_SMLAWB( S[ 0 ], B_Q28[ 0 ], inval ), 2 );

        S[ 0 ] = S[1] + silk_RSHIFT_ROUND( silk_SMULWB( out32_Q14, A0_L_Q28 ), 14 );
        S[ 0 ] = silk_SMLAWB( S[ 0 ], out32_Q14, A0_U_Q28 );
        S[ 0 ] = silk_SMLAWB( S[ 0 ], B_Q28[ 1 ], inval);

        S[ 1 ] = silk_RSHIFT_ROUND( silk_SMULWB( out32_Q14, A1_L_Q28 ), 14 );
        S[ 1 ] = silk_SMLAWB( S[ 1 ], out32_Q14, A1_U_Q28 );
        S[ 1 ] = silk_SMLAWB( S[ 1 ], B_Q28[ 2 ], inval );

        /* Scale back to Q0 and saturate */
        out[ k * stride ] = (opus_int16)silk_SAT16( silk_RSHIFT( out32_Q14 + (1<<14) - 1, 14 ) );
    }
}
