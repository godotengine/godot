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
    silk_float                      XX[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ], /* O    Weight for LTP quantization         */
    silk_float                      xX[ MAX_NB_SUBFR * LTP_ORDER ],     /* O    Weight for LTP quantization                 */
    const silk_float                r_ptr[],                            /* I    LPC residual                                */
    const opus_int                  lag[ MAX_NB_SUBFR ],                /* I    LTP lags                                    */
    const opus_int                  subfr_length,                       /* I    Subframe length                             */
    const opus_int                  nb_subfr,                           /* I    number of subframes                         */
    int                             arch
)
{
    opus_int   k;
    silk_float *xX_ptr, *XX_ptr;
    const silk_float *lag_ptr;
    silk_float xx, temp;

    xX_ptr = xX;
    XX_ptr = XX;
    for( k = 0; k < nb_subfr; k++ ) {
        lag_ptr = r_ptr - ( lag[ k ] + LTP_ORDER / 2 );
        silk_corrMatrix_FLP( lag_ptr, subfr_length, LTP_ORDER, XX_ptr, arch );
        silk_corrVector_FLP( lag_ptr, r_ptr, subfr_length, LTP_ORDER, xX_ptr, arch );
        xx = ( silk_float )silk_energy_FLP( r_ptr, subfr_length + LTP_ORDER );
        temp = 1.0f / silk_max( xx, LTP_CORR_INV_MAX * 0.5f * ( XX_ptr[ 0 ] + XX_ptr[ 24 ] ) + 1.0f );
        silk_scale_vector_FLP( XX_ptr, temp, LTP_ORDER * LTP_ORDER );
        silk_scale_vector_FLP( xX_ptr, temp, LTP_ORDER );

        r_ptr  += subfr_length;
        XX_ptr += LTP_ORDER * LTP_ORDER;
        xX_ptr += LTP_ORDER;
    }
}
