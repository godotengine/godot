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

#include "main_FLP.h"

/* Autocorrelations for a warped frequency axis */
void silk_warped_autocorrelation_FLP(
    silk_float                      *corr,                              /* O    Result [order + 1]                          */
    const silk_float                *input,                             /* I    Input data to correlate                     */
    const silk_float                warping,                            /* I    Warping coefficient                         */
    const opus_int                  length,                             /* I    Length of input                             */
    const opus_int                  order                               /* I    Correlation order (even)                    */
)
{
    opus_int    n, i;
    double      tmp1, tmp2;
    double      state[ MAX_SHAPE_LPC_ORDER + 1 ] = { 0 };
    double      C[     MAX_SHAPE_LPC_ORDER + 1 ] = { 0 };

    /* Order must be even */
    celt_assert( ( order & 1 ) == 0 );

    /* Loop over samples */
    for( n = 0; n < length; n++ ) {
        tmp1 = input[ n ];
        /* Loop over allpass sections */
        for( i = 0; i < order; i += 2 ) {
            /* Output of allpass section */
            /* We voluntarily use two multiples instead of factoring the expression to
               reduce the length of the dependency chain (tmp1->tmp2->tmp1... ). */
            tmp2 = state[ i ] + warping * state[ i + 1 ] - warping * tmp1;
            state[ i ] = tmp1;
            C[ i ] += state[ 0 ] * tmp1;
            /* Output of allpass section */
            tmp1 = state[ i + 1 ] + warping * state[ i + 2 ] - warping * tmp2;
            state[ i + 1 ] = tmp2;
            C[ i + 1 ] += state[ 0 ] * tmp2;
        }
        state[ order ] = tmp1;
        C[ order ] += state[ 0 ] * tmp1;
    }

    /* Copy correlations in silk_float output format */
    for( i = 0; i < order + 1; i++ ) {
        corr[ i ] = ( silk_float )C[ i ];
    }
}
