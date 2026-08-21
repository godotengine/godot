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

#include "SigProc_FLP.h"

silk_float silk_schur_FLP(                  /* O    returns residual energy                                     */
    silk_float          refl_coef[],        /* O    reflection coefficients (length order)                      */
    const silk_float    auto_corr[],        /* I    autocorrelation sequence (length order+1)                   */
    opus_int            order               /* I    order                                                       */
)
{
    opus_int   k, n;
    double C[ SILK_MAX_ORDER_LPC + 1 ][ 2 ];
    double Ctmp1, Ctmp2, rc_tmp;

    celt_assert( order >= 0 && order <= SILK_MAX_ORDER_LPC );

    /* Copy correlations */
    k = 0;
    do {
        C[ k ][ 0 ] = C[ k ][ 1 ] = auto_corr[ k ];
    } while( ++k <= order );

    for( k = 0; k < order; k++ ) {
        /* Get reflection coefficient */
        rc_tmp = -C[ k + 1 ][ 0 ] / silk_max_float( C[ 0 ][ 1 ], 1e-9f );

        /* Save the output */
        refl_coef[ k ] = (silk_float)rc_tmp;

        /* Update correlations */
        for( n = 0; n < order - k; n++ ) {
            Ctmp1 = C[ n + k + 1 ][ 0 ];
            Ctmp2 = C[ n ][ 1 ];
            C[ n + k + 1 ][ 0 ] = Ctmp1 + Ctmp2 * rc_tmp;
            C[ n ][ 1 ]         = Ctmp2 + Ctmp1 * rc_tmp;
        }
    }

    /* Return residual energy */
    return (silk_float)C[ 0 ][ 1 ];
}
