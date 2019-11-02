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
#include "resampler_rom.h"

/* Downsample by a factor 2 */
void silk_resampler_down2(
    opus_int32                  *S,                 /* I/O  State vector [ 2 ]                                          */
    opus_int16                  *out,               /* O    Output signal [ floor(len/2) ]                              */
    const opus_int16            *in,                /* I    Input signal [ len ]                                        */
    opus_int32                  inLen               /* I    Number of input samples                                     */
)
{
    opus_int32 k, len2 = silk_RSHIFT32( inLen, 1 );
    opus_int32 in32, out32, Y, X;

    celt_assert( silk_resampler_down2_0 > 0 );
    celt_assert( silk_resampler_down2_1 < 0 );

    /* Internal variables and state are in Q10 format */
    for( k = 0; k < len2; k++ ) {
        /* Convert to Q10 */
        in32 = silk_LSHIFT( (opus_int32)in[ 2 * k ], 10 );

        /* All-pass section for even input sample */
        Y      = silk_SUB32( in32, S[ 0 ] );
        X      = silk_SMLAWB( Y, Y, silk_resampler_down2_1 );
        out32  = silk_ADD32( S[ 0 ], X );
        S[ 0 ] = silk_ADD32( in32, X );

        /* Convert to Q10 */
        in32 = silk_LSHIFT( (opus_int32)in[ 2 * k + 1 ], 10 );

        /* All-pass section for odd input sample, and add to output of previous section */
        Y      = silk_SUB32( in32, S[ 1 ] );
        X      = silk_SMULWB( Y, silk_resampler_down2_0 );
        out32  = silk_ADD32( out32, S[ 1 ] );
        out32  = silk_ADD32( out32, X );
        S[ 1 ] = silk_ADD32( in32, X );

        /* Add, convert back to int16 and store to output */
        out[ k ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( out32, 11 ) );
    }
}

