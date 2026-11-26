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

#define MAX_ITERATIONS_RESIDUAL_NRG         10
#define REGULARIZATION_FACTOR               1e-8f

/* Residual energy: nrg = wxx - 2 * wXx * c + c' * wXX * c */
silk_float silk_residual_energy_covar_FLP(                              /* O    Weighted residual energy                    */
    const silk_float                *c,                                 /* I    Filter coefficients                         */
    silk_float                      *wXX,                               /* I/O  Weighted correlation matrix, reg. out       */
    const silk_float                *wXx,                               /* I    Weighted correlation vector                 */
    const silk_float                wxx,                                /* I    Weighted correlation value                  */
    const opus_int                  D                                   /* I    Dimension                                   */
)
{
    opus_int   i, j, k;
    silk_float tmp, nrg = 0.0f, regularization;

    /* Safety checks */
    celt_assert( D >= 0 );

    regularization = REGULARIZATION_FACTOR * ( wXX[ 0 ] + wXX[ D * D - 1 ] );
    for( k = 0; k < MAX_ITERATIONS_RESIDUAL_NRG; k++ ) {
        nrg = wxx;

        tmp = 0.0f;
        for( i = 0; i < D; i++ ) {
            tmp += wXx[ i ] * c[ i ];
        }
        nrg -= 2.0f * tmp;

        /* compute c' * wXX * c, assuming wXX is symmetric */
        for( i = 0; i < D; i++ ) {
            tmp = 0.0f;
            for( j = i + 1; j < D; j++ ) {
                tmp += matrix_c_ptr( wXX, i, j, D ) * c[ j ];
            }
            nrg += c[ i ] * ( 2.0f * tmp + matrix_c_ptr( wXX, i, i, D ) * c[ i ] );
        }
        if( nrg > 0 ) {
            break;
        } else {
            /* Add white noise */
            for( i = 0; i < D; i++ ) {
                matrix_c_ptr( wXX, i, i, D ) +=  regularization;
            }
            /* Increase noise for next run */
            regularization *= 2.0f;
        }
    }
    if( k == MAX_ITERATIONS_RESIDUAL_NRG ) {
        silk_assert( nrg == 0 );
        nrg = 1.0f;
    }

    return nrg;
}

/* Calculates residual energies of input subframes where all subframes have LPC_order   */
/* of preceding samples                                                                 */
void silk_residual_energy_FLP(
    silk_float                      nrgs[ MAX_NB_SUBFR ],               /* O    Residual energy per subframe                */
    const silk_float                x[],                                /* I    Input signal                                */
    silk_float                      a[ 2 ][ MAX_LPC_ORDER ],            /* I    AR coefs for each frame half                */
    const silk_float                gains[],                            /* I    Quantization gains                          */
    const opus_int                  subfr_length,                       /* I    Subframe length                             */
    const opus_int                  nb_subfr,                           /* I    number of subframes                         */
    const opus_int                  LPC_order                           /* I    LPC order                                   */
)
{
    opus_int     shift;
    silk_float   *LPC_res_ptr, LPC_res[ ( MAX_FRAME_LENGTH + MAX_NB_SUBFR * MAX_LPC_ORDER ) / 2 ];

    LPC_res_ptr = LPC_res + LPC_order;
    shift = LPC_order + subfr_length;

    /* Filter input to create the LPC residual for each frame half, and measure subframe energies */
    silk_LPC_analysis_filter_FLP( LPC_res, a[ 0 ], x + 0 * shift, 2 * shift, LPC_order );
    nrgs[ 0 ] = ( silk_float )( gains[ 0 ] * gains[ 0 ] * silk_energy_FLP( LPC_res_ptr + 0 * shift, subfr_length ) );
    nrgs[ 1 ] = ( silk_float )( gains[ 1 ] * gains[ 1 ] * silk_energy_FLP( LPC_res_ptr + 1 * shift, subfr_length ) );

    if( nb_subfr == MAX_NB_SUBFR ) {
        silk_LPC_analysis_filter_FLP( LPC_res, a[ 1 ], x + 2 * shift, 2 * shift, LPC_order );
        nrgs[ 2 ] = ( silk_float )( gains[ 2 ] * gains[ 2 ] * silk_energy_FLP( LPC_res_ptr + 0 * shift, subfr_length ) );
        nrgs[ 3 ] = ( silk_float )( gains[ 3 ] * gains[ 3 ] * silk_energy_FLP( LPC_res_ptr + 1 * shift, subfr_length ) );
    }
}
