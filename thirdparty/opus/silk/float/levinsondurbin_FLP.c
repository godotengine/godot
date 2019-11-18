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

/* Solve the normal equations using the Levinson-Durbin recursion */
silk_float silk_levinsondurbin_FLP(         /* O    prediction error energy                                     */
    silk_float          A[],                /* O    prediction coefficients    [order]                          */
    const silk_float    corr[],             /* I    input auto-correlations [order + 1]                         */
    const opus_int      order               /* I    prediction order                                            */
)
{
    opus_int   i, mHalf, m;
    silk_float min_nrg, nrg, t, km, Atmp1, Atmp2;

    min_nrg = 1e-12f * corr[ 0 ] + 1e-9f;
    nrg = corr[ 0 ];
    nrg = silk_max_float(min_nrg, nrg);
    A[ 0 ] = corr[ 1 ] / nrg;
    nrg -= A[ 0 ] * corr[ 1 ];
    nrg = silk_max_float(min_nrg, nrg);

    for( m = 1; m < order; m++ )
    {
        t = corr[ m + 1 ];
        for( i = 0; i < m; i++ ) {
            t -= A[ i ] * corr[ m - i ];
        }

        /* reflection coefficient */
        km = t / nrg;

        /* residual energy */
        nrg -= km * t;
        nrg = silk_max_float(min_nrg, nrg);

        mHalf = m >> 1;
        for( i = 0; i < mHalf; i++ ) {
            Atmp1 = A[ i ];
            Atmp2 = A[ m - i - 1 ];
            A[ m - i - 1 ] -= km * Atmp1;
            A[ i ]         -= km * Atmp2;
        }
        if( m & 1 ) {
            A[ mHalf ]     -= km * A[ mHalf ];
        }
        A[ m ] = km;
    }

    /* return the residual energy */
    return nrg;
}

