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

#include "main.h"

/* Compute quantization errors for an LPC_order element input vector for a VQ codebook */
void silk_NLSF_VQ(
    opus_int32                  err_Q26[],                      /* O    Quantization errors [K]                     */
    const opus_int16            in_Q15[],                       /* I    Input vectors to be quantized [LPC_order]   */
    const opus_uint8            pCB_Q8[],                       /* I    Codebook vectors [K*LPC_order]              */
    const opus_int              K,                              /* I    Number of codebook vectors                  */
    const opus_int              LPC_order                       /* I    Number of LPCs                              */
)
{
    opus_int        i, m;
    opus_int32      diff_Q15, sum_error_Q30, sum_error_Q26;

    silk_assert( LPC_order <= 16 );
    silk_assert( ( LPC_order & 1 ) == 0 );

    /* Loop over codebook */
    for( i = 0; i < K; i++ ) {
        sum_error_Q26 = 0;
        for( m = 0; m < LPC_order; m += 2 ) {
            /* Compute weighted squared quantization error for index m */
            diff_Q15 = silk_SUB_LSHIFT32( in_Q15[ m ], (opus_int32)*pCB_Q8++, 7 ); /* range: [ -32767 : 32767 ]*/
            sum_error_Q30 = silk_SMULBB( diff_Q15, diff_Q15 );

            /* Compute weighted squared quantization error for index m + 1 */
            diff_Q15 = silk_SUB_LSHIFT32( in_Q15[m + 1], (opus_int32)*pCB_Q8++, 7 ); /* range: [ -32767 : 32767 ]*/
            sum_error_Q30 = silk_SMLABB( sum_error_Q30, diff_Q15, diff_Q15 );

            sum_error_Q26 = silk_ADD_RSHIFT32( sum_error_Q26, sum_error_Q30, 4 );

            silk_assert( sum_error_Q26 >= 0 );
            silk_assert( sum_error_Q30 >= 0 );
        }
        err_Q26[ i ] = sum_error_Q26;
    }
}
