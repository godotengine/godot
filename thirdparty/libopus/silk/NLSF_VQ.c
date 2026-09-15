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
    opus_int32                  err_Q24[],                      /* O    Quantization errors [K]                     */
    const opus_int16            in_Q15[],                       /* I    Input vectors to be quantized [LPC_order]   */
    const opus_uint8            pCB_Q8[],                       /* I    Codebook vectors [K*LPC_order]              */
    const opus_int16            pWght_Q9[],                     /* I    Codebook weights [K*LPC_order]              */
    const opus_int              K,                              /* I    Number of codebook vectors                  */
    const opus_int              LPC_order                       /* I    Number of LPCs                              */
)
{
    opus_int         i, m;
    opus_int32       diff_Q15, diffw_Q24, sum_error_Q24, pred_Q24;
    const opus_int16 *w_Q9_ptr;
    const opus_uint8 *cb_Q8_ptr;

    celt_assert( ( LPC_order & 1 ) == 0 );

    /* Loop over codebook */
    cb_Q8_ptr = pCB_Q8;
    w_Q9_ptr = pWght_Q9;
    for( i = 0; i < K; i++ ) {
        sum_error_Q24 = 0;
        pred_Q24 = 0;
        for( m = LPC_order-2; m >= 0; m -= 2 ) {
            /* Compute weighted absolute predictive quantization error for index m + 1 */
            diff_Q15 = silk_SUB_LSHIFT32( in_Q15[ m + 1 ], (opus_int32)cb_Q8_ptr[ m + 1 ], 7 ); /* range: [ -32767 : 32767 ]*/
            diffw_Q24 = silk_SMULBB( diff_Q15, w_Q9_ptr[ m + 1 ] );
            sum_error_Q24 = silk_ADD32( sum_error_Q24, silk_abs( silk_SUB_RSHIFT32( diffw_Q24, pred_Q24, 1 ) ) );
            pred_Q24 = diffw_Q24;

            /* Compute weighted absolute predictive quantization error for index m */
            diff_Q15 = silk_SUB_LSHIFT32( in_Q15[ m ], (opus_int32)cb_Q8_ptr[ m ], 7 ); /* range: [ -32767 : 32767 ]*/
            diffw_Q24 = silk_SMULBB( diff_Q15, w_Q9_ptr[ m ] );
            sum_error_Q24 = silk_ADD32( sum_error_Q24, silk_abs( silk_SUB_RSHIFT32( diffw_Q24, pred_Q24, 1 ) ) );
            pred_Q24 = diffw_Q24;

            silk_assert( sum_error_Q24 >= 0 );
        }
        err_Q24[ i ] = sum_error_Q24;
        cb_Q8_ptr += LPC_order;
        w_Q9_ptr += LPC_order;
    }
}
