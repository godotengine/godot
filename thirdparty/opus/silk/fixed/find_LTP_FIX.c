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

#include "main_FIX.h"
#include "tuning_parameters.h"

void silk_find_LTP_FIX(
    opus_int32                      XXLTP_Q17[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ], /* O    Correlation matrix                                               */
    opus_int32                      xXLTP_Q17[ MAX_NB_SUBFR * LTP_ORDER ],  /* O    Correlation vector                                                          */
    const opus_int16                r_ptr[],                                /* I    Residual signal after LPC                                                   */
    const opus_int                  lag[ MAX_NB_SUBFR ],                    /* I    LTP lags                                                                    */
    const opus_int                  subfr_length,                           /* I    Subframe length                                                             */
    const opus_int                  nb_subfr,                               /* I    Number of subframes                                                         */
    int                             arch                                    /* I    Run-time architecture                                                       */
)
{
    opus_int   i, k, extra_shifts;
    opus_int   xx_shifts, xX_shifts, XX_shifts;
    const opus_int16 *lag_ptr;
    opus_int32 *XXLTP_Q17_ptr, *xXLTP_Q17_ptr;
    opus_int32 xx, nrg, temp;

    xXLTP_Q17_ptr = xXLTP_Q17;
    XXLTP_Q17_ptr = XXLTP_Q17;
    for( k = 0; k < nb_subfr; k++ ) {
        lag_ptr = r_ptr - ( lag[ k ] + LTP_ORDER / 2 );

        silk_sum_sqr_shift( &xx, &xx_shifts, r_ptr, subfr_length + LTP_ORDER );                            /* xx in Q( -xx_shifts ) */
        silk_corrMatrix_FIX( lag_ptr, subfr_length, LTP_ORDER, XXLTP_Q17_ptr, &nrg, &XX_shifts, arch );    /* XXLTP_Q17_ptr and nrg in Q( -XX_shifts ) */
        extra_shifts = xx_shifts - XX_shifts;
        if( extra_shifts > 0 ) {
            /* Shift XX */
            xX_shifts = xx_shifts;
            for( i = 0; i < LTP_ORDER * LTP_ORDER; i++ ) {
                XXLTP_Q17_ptr[ i ] = silk_RSHIFT32( XXLTP_Q17_ptr[ i ], extra_shifts );              /* Q( -xX_shifts ) */
            }
            nrg = silk_RSHIFT32( nrg, extra_shifts );                                                /* Q( -xX_shifts ) */
        } else if( extra_shifts < 0 ) {
            /* Shift xx */
            xX_shifts = XX_shifts;
            xx = silk_RSHIFT32( xx, -extra_shifts );                                                 /* Q( -xX_shifts ) */
        } else {
            xX_shifts = xx_shifts;
        }
        silk_corrVector_FIX( lag_ptr, r_ptr, subfr_length, LTP_ORDER, xXLTP_Q17_ptr, xX_shifts, arch );    /* xXLTP_Q17_ptr in Q( -xX_shifts ) */

        /* At this point all correlations are in Q(-xX_shifts) */
        temp = silk_SMLAWB( 1, nrg, SILK_FIX_CONST( LTP_CORR_INV_MAX, 16 ) );
        temp = silk_max( temp, xx );
TIC(div)
#if 0
        for( i = 0; i < LTP_ORDER * LTP_ORDER; i++ ) {
            XXLTP_Q17_ptr[ i ] = silk_DIV32_varQ( XXLTP_Q17_ptr[ i ], temp, 17 );
        }
        for( i = 0; i < LTP_ORDER; i++ ) {
            xXLTP_Q17_ptr[ i ] = silk_DIV32_varQ( xXLTP_Q17_ptr[ i ], temp, 17 );
        }
#else
        for( i = 0; i < LTP_ORDER * LTP_ORDER; i++ ) {
            XXLTP_Q17_ptr[ i ] = (opus_int32)( silk_LSHIFT64( (opus_int64)XXLTP_Q17_ptr[ i ], 17 ) / temp );
        }
        for( i = 0; i < LTP_ORDER; i++ ) {
            xXLTP_Q17_ptr[ i ] = (opus_int32)( silk_LSHIFT64( (opus_int64)xXLTP_Q17_ptr[ i ], 17 ) / temp );
        }
#endif
TOC(div)
        r_ptr         += subfr_length;
        XXLTP_Q17_ptr += LTP_ORDER * LTP_ORDER;
        xXLTP_Q17_ptr += LTP_ORDER;
    }
}
