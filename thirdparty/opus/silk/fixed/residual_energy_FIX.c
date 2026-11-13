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
#include "stack_alloc.h"

/* Calculates residual energies of input subframes where all subframes have LPC_order   */
/* of preceding samples                                                                 */
void silk_residual_energy_FIX(
          opus_int32                nrgs[ MAX_NB_SUBFR ],                   /* O    Residual energy per subframe                                                */
          opus_int                  nrgsQ[ MAX_NB_SUBFR ],                  /* O    Q value per subframe                                                        */
    const opus_int16                x[],                                    /* I    Input signal                                                                */
          opus_int16                a_Q12[ 2 ][ MAX_LPC_ORDER ],            /* I    AR coefs for each frame half                                                */
    const opus_int32                gains[ MAX_NB_SUBFR ],                  /* I    Quantization gains                                                          */
    const opus_int                  subfr_length,                           /* I    Subframe length                                                             */
    const opus_int                  nb_subfr,                               /* I    Number of subframes                                                         */
    const opus_int                  LPC_order,                              /* I    LPC order                                                                   */
          int                       arch                                    /* I    Run-time architecture                                                       */
)
{
    opus_int         offset, i, j, rshift, lz1, lz2;
    opus_int16       *LPC_res_ptr;
    VARDECL( opus_int16, LPC_res );
    const opus_int16 *x_ptr;
    opus_int32       tmp32;
    SAVE_STACK;

    x_ptr  = x;
    offset = LPC_order + subfr_length;

    /* Filter input to create the LPC residual for each frame half, and measure subframe energies */
    ALLOC( LPC_res, ( MAX_NB_SUBFR >> 1 ) * offset, opus_int16 );
    celt_assert( ( nb_subfr >> 1 ) * ( MAX_NB_SUBFR >> 1 ) == nb_subfr );
    for( i = 0; i < nb_subfr >> 1; i++ ) {
        /* Calculate half frame LPC residual signal including preceding samples */
        silk_LPC_analysis_filter( LPC_res, x_ptr, a_Q12[ i ], ( MAX_NB_SUBFR >> 1 ) * offset, LPC_order, arch );

        /* Point to first subframe of the just calculated LPC residual signal */
        LPC_res_ptr = LPC_res + LPC_order;
        for( j = 0; j < ( MAX_NB_SUBFR >> 1 ); j++ ) {
            /* Measure subframe energy */
            silk_sum_sqr_shift( &nrgs[ i * ( MAX_NB_SUBFR >> 1 ) + j ], &rshift, LPC_res_ptr, subfr_length );

            /* Set Q values for the measured energy */
            nrgsQ[ i * ( MAX_NB_SUBFR >> 1 ) + j ] = -rshift;

            /* Move to next subframe */
            LPC_res_ptr += offset;
        }
        /* Move to next frame half */
        x_ptr += ( MAX_NB_SUBFR >> 1 ) * offset;
    }

    /* Apply the squared subframe gains */
    for( i = 0; i < nb_subfr; i++ ) {
        /* Fully upscale gains and energies */
        lz1 = silk_CLZ32( nrgs[  i ] ) - 1;
        lz2 = silk_CLZ32( gains[ i ] ) - 1;

        tmp32 = silk_LSHIFT32( gains[ i ], lz2 );

        /* Find squared gains */
        tmp32 = silk_SMMUL( tmp32, tmp32 ); /* Q( 2 * lz2 - 32 )*/

        /* Scale energies */
        nrgs[ i ] = silk_SMMUL( tmp32, silk_LSHIFT32( nrgs[ i ], lz1 ) ); /* Q( nrgsQ[ i ] + lz1 + 2 * lz2 - 32 - 32 )*/
        nrgsQ[ i ] += lz1 + 2 * lz2 - 32 - 32;
    }
    RESTORE_STACK;
}
