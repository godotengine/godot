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
#include "tuning_parameters.h"

/* Finds LPC vector from correlations, and converts to NLSF */
void silk_find_LPC_FIX(
    silk_encoder_state              *psEncC,                                /* I/O  Encoder state                                                               */
    opus_int16                      NLSF_Q15[],                             /* O    NLSFs                                                                       */
    const opus_int16                x[],                                    /* I    Input signal                                                                */
    const opus_int32                minInvGain_Q30                          /* I    Inverse of max prediction gain                                              */
)
{
    opus_int     k, subfr_length;
    opus_int32   a_Q16[ MAX_LPC_ORDER ];
    opus_int     isInterpLower, shift;
    opus_int32   res_nrg0, res_nrg1;
    opus_int     rshift0, rshift1;

    /* Used only for LSF interpolation */
    opus_int32   a_tmp_Q16[ MAX_LPC_ORDER ], res_nrg_interp, res_nrg, res_tmp_nrg;
    opus_int     res_nrg_interp_Q, res_nrg_Q, res_tmp_nrg_Q;
    opus_int16   a_tmp_Q12[ MAX_LPC_ORDER ];
    opus_int16   NLSF0_Q15[ MAX_LPC_ORDER ];
    SAVE_STACK;

    subfr_length = psEncC->subfr_length + psEncC->predictLPCOrder;

    /* Default: no interpolation */
    psEncC->indices.NLSFInterpCoef_Q2 = 4;

    /* Burg AR analysis for the full frame */
    silk_burg_modified( &res_nrg, &res_nrg_Q, a_Q16, x, minInvGain_Q30, subfr_length, psEncC->nb_subfr, psEncC->predictLPCOrder, psEncC->arch );

    if( psEncC->useInterpolatedNLSFs && !psEncC->first_frame_after_reset && psEncC->nb_subfr == MAX_NB_SUBFR ) {
        VARDECL( opus_int16, LPC_res );

        /* Optimal solution for last 10 ms */
        silk_burg_modified( &res_tmp_nrg, &res_tmp_nrg_Q, a_tmp_Q16, x + 2 * subfr_length, minInvGain_Q30, subfr_length, 2, psEncC->predictLPCOrder, psEncC->arch );

        /* subtract residual energy here, as that's easier than adding it to the    */
        /* residual energy of the first 10 ms in each iteration of the search below */
        shift = res_tmp_nrg_Q - res_nrg_Q;
        if( shift >= 0 ) {
            if( shift < 32 ) {
                res_nrg = res_nrg - silk_RSHIFT( res_tmp_nrg, shift );
            }
        } else {
            silk_assert( shift > -32 );
            res_nrg   = silk_RSHIFT( res_nrg, -shift ) - res_tmp_nrg;
            res_nrg_Q = res_tmp_nrg_Q;
        }

        /* Convert to NLSFs */
        silk_A2NLSF( NLSF_Q15, a_tmp_Q16, psEncC->predictLPCOrder );

        ALLOC( LPC_res, 2 * subfr_length, opus_int16 );

        /* Search over interpolation indices to find the one with lowest residual energy */
        for( k = 3; k >= 0; k-- ) {
            /* Interpolate NLSFs for first half */
            silk_interpolate( NLSF0_Q15, psEncC->prev_NLSFq_Q15, NLSF_Q15, k, psEncC->predictLPCOrder );

            /* Convert to LPC for residual energy evaluation */
            silk_NLSF2A( a_tmp_Q12, NLSF0_Q15, psEncC->predictLPCOrder, psEncC->arch );

            /* Calculate residual energy with NLSF interpolation */
            silk_LPC_analysis_filter( LPC_res, x, a_tmp_Q12, 2 * subfr_length, psEncC->predictLPCOrder, psEncC->arch );

            silk_sum_sqr_shift( &res_nrg0, &rshift0, LPC_res + psEncC->predictLPCOrder,                subfr_length - psEncC->predictLPCOrder );
            silk_sum_sqr_shift( &res_nrg1, &rshift1, LPC_res + psEncC->predictLPCOrder + subfr_length, subfr_length - psEncC->predictLPCOrder );

            /* Add subframe energies from first half frame */
            shift = rshift0 - rshift1;
            if( shift >= 0 ) {
                res_nrg1         = silk_RSHIFT( res_nrg1, shift );
                res_nrg_interp_Q = -rshift0;
            } else {
                res_nrg0         = silk_RSHIFT( res_nrg0, -shift );
                res_nrg_interp_Q = -rshift1;
            }
            res_nrg_interp = silk_ADD32( res_nrg0, res_nrg1 );

            /* Compare with first half energy without NLSF interpolation, or best interpolated value so far */
            shift = res_nrg_interp_Q - res_nrg_Q;
            if( shift >= 0 ) {
                if( silk_RSHIFT( res_nrg_interp, shift ) < res_nrg ) {
                    isInterpLower = silk_TRUE;
                } else {
                    isInterpLower = silk_FALSE;
                }
            } else {
                if( -shift < 32 ) {
                    if( res_nrg_interp < silk_RSHIFT( res_nrg, -shift ) ) {
                        isInterpLower = silk_TRUE;
                    } else {
                        isInterpLower = silk_FALSE;
                    }
                } else {
                    isInterpLower = silk_FALSE;
                }
            }

            /* Determine whether current interpolated NLSFs are best so far */
            if( isInterpLower == silk_TRUE ) {
                /* Interpolation has lower residual energy */
                res_nrg   = res_nrg_interp;
                res_nrg_Q = res_nrg_interp_Q;
                psEncC->indices.NLSFInterpCoef_Q2 = (opus_int8)k;
            }
        }
    }

    if( psEncC->indices.NLSFInterpCoef_Q2 == 4 ) {
        /* NLSF interpolation is currently inactive, calculate NLSFs from full frame AR coefficients */
        silk_A2NLSF( NLSF_Q15, a_Q16, psEncC->predictLPCOrder );
    }

    celt_assert( psEncC->indices.NLSFInterpCoef_Q2 == 4 || ( psEncC->useInterpolatedNLSFs && !psEncC->first_frame_after_reset && psEncC->nb_subfr == MAX_NB_SUBFR ) );
    RESTORE_STACK;
}
