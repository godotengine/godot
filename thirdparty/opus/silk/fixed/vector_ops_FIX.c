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
#include "pitch.h"

/* Copy and multiply a vector by a constant */
void silk_scale_copy_vector16(
    opus_int16                  *data_out,
    const opus_int16            *data_in,
    opus_int32                  gain_Q16,           /* I    Gain in Q16                                                 */
    const opus_int              dataSize            /* I    Length                                                      */
)
{
    opus_int  i;
    opus_int32 tmp32;

    for( i = 0; i < dataSize; i++ ) {
        tmp32 = silk_SMULWB( gain_Q16, data_in[ i ] );
        data_out[ i ] = (opus_int16)silk_CHECK_FIT16( tmp32 );
    }
}

/* Multiply a vector by a constant */
void silk_scale_vector32_Q26_lshift_18(
    opus_int32                  *data1,             /* I/O  Q0/Q18                                                      */
    opus_int32                  gain_Q26,           /* I    Q26                                                         */
    opus_int                    dataSize            /* I    length                                                      */
)
{
    opus_int  i;

    for( i = 0; i < dataSize; i++ ) {
        data1[ i ] = (opus_int32)silk_CHECK_FIT32( silk_RSHIFT64( silk_SMULL( data1[ i ], gain_Q26 ), 8 ) );    /* OUTPUT: Q18 */
    }
}

/* sum = for(i=0;i<len;i++)inVec1[i]*inVec2[i];      ---        inner product   */
/* Note for ARM asm:                                                            */
/*        * inVec1 and inVec2 should be at least 2 byte aligned.                */
/*        * len should be positive 16bit integer.                               */
/*        * only when len>6, memory access can be reduced by half.              */
opus_int32 silk_inner_prod_aligned(
    const opus_int16 *const     inVec1,             /*    I input vector 1                                              */
    const opus_int16 *const     inVec2,             /*    I input vector 2                                              */
    const opus_int              len,                /*    I vector lengths                                              */
    int                         arch                /*    I Run-time architecture                                       */
)
{
#ifdef FIXED_POINT
   return celt_inner_prod(inVec1, inVec2, len, arch);
#else
    opus_int   i;
    opus_int32 sum = 0;
    for( i = 0; i < len; i++ ) {
        sum = silk_SMLABB( sum, inVec1[ i ], inVec2[ i ] );
    }
    return sum;
#endif
}

opus_int64 silk_inner_prod16_aligned_64_c(
    const opus_int16            *inVec1,            /*    I input vector 1                                              */
    const opus_int16            *inVec2,            /*    I input vector 2                                              */
    const opus_int              len                 /*    I vector lengths                                              */
)
{
    opus_int   i;
    opus_int64 sum = 0;
    for( i = 0; i < len; i++ ) {
        sum = silk_SMLALBB( sum, inVec1[ i ], inVec2[ i ] );
    }
    return sum;
}
