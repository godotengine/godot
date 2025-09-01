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

/* Compute number of bits to right shift the sum of squares of a vector */
/* of int16s to make it fit in an int32                                 */
void silk_sum_sqr_shift(
    opus_int32                  *energy,            /* O   Energy of x, after shifting to the right                     */
    opus_int                    *shift,             /* O   Number of bits right shift applied to energy                 */
    const opus_int16            *x,                 /* I   Input vector                                                 */
    opus_int                    len                 /* I   Length of input vector                                       */
)
{
    opus_int   i, shft;
    opus_uint32 nrg_tmp;
    opus_int32 nrg;

    /* Do a first run with the maximum shift we could have. */
    shft = 31-silk_CLZ32(len);
    /* Let's be conservative with rounding and start with nrg=len. */
    nrg  = len;
    for( i = 0; i < len - 1; i += 2 ) {
        nrg_tmp = silk_SMULBB( x[ i ], x[ i ] );
        nrg_tmp = silk_SMLABB_ovflw( nrg_tmp, x[ i + 1 ], x[ i + 1 ] );
        nrg = (opus_int32)silk_ADD_RSHIFT_uint( nrg, nrg_tmp, shft );
    }
    if( i < len ) {
        /* One sample left to process */
        nrg_tmp = silk_SMULBB( x[ i ], x[ i ] );
        nrg = (opus_int32)silk_ADD_RSHIFT_uint( nrg, nrg_tmp, shft );
    }
    silk_assert( nrg >= 0 );
    /* Make sure the result will fit in a 32-bit signed integer with two bits
       of headroom. */
    shft = silk_max_32(0, shft+3 - silk_CLZ32(nrg));
    nrg = 0;
    for( i = 0 ; i < len - 1; i += 2 ) {
        nrg_tmp = silk_SMULBB( x[ i ], x[ i ] );
        nrg_tmp = silk_SMLABB_ovflw( nrg_tmp, x[ i + 1 ], x[ i + 1 ] );
        nrg = (opus_int32)silk_ADD_RSHIFT_uint( nrg, nrg_tmp, shft );
    }
    if( i < len ) {
        /* One sample left to process */
        nrg_tmp = silk_SMULBB( x[ i ], x[ i ] );
        nrg = (opus_int32)silk_ADD_RSHIFT_uint( nrg, nrg_tmp, shft );
    }

    silk_assert( nrg >= 0 );

    /* Output arguments */
    *shift  = shft;
    *energy = nrg;
}

