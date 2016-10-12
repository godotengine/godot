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
#include "tuning_parameters.h"

void silk_find_LTP_FLP(
    silk_float                      b[ MAX_NB_SUBFR * LTP_ORDER ],      /* O    LTP coefs                                   */
    silk_float                      WLTP[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ], /* O    Weight for LTP quantization       */
    silk_float                      *LTPredCodGain,                     /* O    LTP coding gain                             */
    const silk_float                r_lpc[],                            /* I    LPC residual                                */
    const opus_int                  lag[  MAX_NB_SUBFR ],               /* I    LTP lags                                    */
    const silk_float                Wght[ MAX_NB_SUBFR ],               /* I    Weights                                     */
    const opus_int                  subfr_length,                       /* I    Subframe length                             */
    const opus_int                  nb_subfr,                           /* I    number of subframes                         */
    const opus_int                  mem_offset                          /* I    Number of samples in LTP memory             */
)
{
    opus_int   i, k;
    silk_float *b_ptr, temp, *WLTP_ptr;
    silk_float LPC_res_nrg, LPC_LTP_res_nrg;
    silk_float d[ MAX_NB_SUBFR ], m, g, delta_b[ LTP_ORDER ];
    silk_float w[ MAX_NB_SUBFR ], nrg[ MAX_NB_SUBFR ], regu;
    silk_float Rr[ LTP_ORDER ], rr[ MAX_NB_SUBFR ];
    const silk_float *r_ptr, *lag_ptr;

    b_ptr    = b;
    WLTP_ptr = WLTP;
    r_ptr    = &r_lpc[ mem_offset ];
    for( k = 0; k < nb_subfr; k++ ) {
        lag_ptr = r_ptr - ( lag[ k ] + LTP_ORDER / 2 );

        silk_corrMatrix_FLP( lag_ptr, subfr_length, LTP_ORDER, WLTP_ptr );
        silk_corrVector_FLP( lag_ptr, r_ptr, subfr_length, LTP_ORDER, Rr );

        rr[ k ] = ( silk_float )silk_energy_FLP( r_ptr, subfr_length );
        regu = 1.0f + rr[ k ] +
            matrix_ptr( WLTP_ptr, 0, 0, LTP_ORDER ) +
            matrix_ptr( WLTP_ptr, LTP_ORDER-1, LTP_ORDER-1, LTP_ORDER );
        regu *= LTP_DAMPING / 3;
        silk_regularize_correlations_FLP( WLTP_ptr, &rr[ k ], regu, LTP_ORDER );
        silk_solve_LDL_FLP( WLTP_ptr, LTP_ORDER, Rr, b_ptr );

        /* Calculate residual energy */
        nrg[ k ] = silk_residual_energy_covar_FLP( b_ptr, WLTP_ptr, Rr, rr[ k ], LTP_ORDER );

        temp = Wght[ k ] / ( nrg[ k ] * Wght[ k ] + 0.01f * subfr_length );
        silk_scale_vector_FLP( WLTP_ptr, temp, LTP_ORDER * LTP_ORDER );
        w[ k ] = matrix_ptr( WLTP_ptr, LTP_ORDER / 2, LTP_ORDER / 2, LTP_ORDER );

        r_ptr    += subfr_length;
        b_ptr    += LTP_ORDER;
        WLTP_ptr += LTP_ORDER * LTP_ORDER;
    }

    /* Compute LTP coding gain */
    if( LTPredCodGain != NULL ) {
        LPC_LTP_res_nrg = 1e-6f;
        LPC_res_nrg     = 0.0f;
        for( k = 0; k < nb_subfr; k++ ) {
            LPC_res_nrg     += rr[  k ] * Wght[ k ];
            LPC_LTP_res_nrg += nrg[ k ] * Wght[ k ];
        }

        silk_assert( LPC_LTP_res_nrg > 0 );
        *LTPredCodGain = 3.0f * silk_log2( LPC_res_nrg / LPC_LTP_res_nrg );
    }

    /* Smoothing */
    /* d = sum( B, 1 ); */
    b_ptr = b;
    for( k = 0; k < nb_subfr; k++ ) {
        d[ k ] = 0;
        for( i = 0; i < LTP_ORDER; i++ ) {
            d[ k ] += b_ptr[ i ];
        }
        b_ptr += LTP_ORDER;
    }
    /* m = ( w * d' ) / ( sum( w ) + 1e-3 ); */
    temp = 1e-3f;
    for( k = 0; k < nb_subfr; k++ ) {
        temp += w[ k ];
    }
    m = 0;
    for( k = 0; k < nb_subfr; k++ ) {
        m += d[ k ] * w[ k ];
    }
    m = m / temp;

    b_ptr = b;
    for( k = 0; k < nb_subfr; k++ ) {
        g = LTP_SMOOTHING / ( LTP_SMOOTHING + w[ k ] ) * ( m - d[ k ] );
        temp = 0;
        for( i = 0; i < LTP_ORDER; i++ ) {
            delta_b[ i ] = silk_max_float( b_ptr[ i ], 0.1f );
            temp += delta_b[ i ];
        }
        temp = g / temp;
        for( i = 0; i < LTP_ORDER; i++ ) {
            b_ptr[ i ] = b_ptr[ i ] + delta_b[ i ] * temp;
        }
        b_ptr += LTP_ORDER;
    }
}
