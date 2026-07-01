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

/* Find least-squares prediction gain for one signal based on another and quantize it */
opus_int32 silk_stereo_find_predictor(                          /* O    Returns predictor in Q13                    */
    opus_int32                  *ratio_Q14,                     /* O    Ratio of residual and mid energies          */
    const opus_int16            x[],                            /* I    Basis signal                                */
    const opus_int16            y[],                            /* I    Target signal                               */
    opus_int32                  mid_res_amp_Q0[],               /* I/O  Smoothed mid, residual norms                */
    opus_int                    length,                         /* I    Number of samples                           */
    opus_int                    smooth_coef_Q16                 /* I    Smoothing coefficient                       */
)
{
    opus_int   scale, scale1, scale2;
    opus_int32 nrgx, nrgy, corr, pred_Q13, pred2_Q10;

    /* Find  predictor */
    silk_sum_sqr_shift( &nrgx, &scale1, x, length );
    silk_sum_sqr_shift( &nrgy, &scale2, y, length );
    scale = silk_max_int( scale1, scale2 );
    scale = scale + ( scale & 1 );          /* make even */
    nrgy = silk_RSHIFT32( nrgy, scale - scale2 );
    nrgx = silk_RSHIFT32( nrgx, scale - scale1 );
    nrgx = silk_max_int( nrgx, 1 );
    corr = silk_inner_prod_aligned_scale( x, y, scale, length );
    pred_Q13 = silk_DIV32_varQ( corr, nrgx, 13 );
    pred_Q13 = silk_LIMIT( pred_Q13, -(1 << 14), 1 << 14 );
    pred2_Q10 = silk_SMULWB( pred_Q13, pred_Q13 );

    /* Faster update for signals with large prediction parameters */
    smooth_coef_Q16 = (opus_int)silk_max_int( smooth_coef_Q16, silk_abs( pred2_Q10 ) );

    /* Smoothed mid and residual norms */
    silk_assert( smooth_coef_Q16 < 32768 );
    scale = silk_RSHIFT( scale, 1 );
    mid_res_amp_Q0[ 0 ] = silk_SMLAWB( mid_res_amp_Q0[ 0 ], silk_LSHIFT( silk_SQRT_APPROX( nrgx ), scale ) - mid_res_amp_Q0[ 0 ],
        smooth_coef_Q16 );
    /* Residual energy = nrgy - 2 * pred * corr + pred^2 * nrgx */
    nrgy = silk_SUB_LSHIFT32( nrgy, silk_SMULWB( corr, pred_Q13 ), 3 + 1 );
    nrgy = silk_ADD_LSHIFT32( nrgy, silk_SMULWB( nrgx, pred2_Q10 ), 6 );
    mid_res_amp_Q0[ 1 ] = silk_SMLAWB( mid_res_amp_Q0[ 1 ], silk_LSHIFT( silk_SQRT_APPROX( nrgy ), scale ) - mid_res_amp_Q0[ 1 ],
        smooth_coef_Q16 );

    /* Ratio of smoothed residual and mid norms */
    *ratio_Q14 = silk_DIV32_varQ( mid_res_amp_Q0[ 1 ], silk_max( mid_res_amp_Q0[ 0 ], 1 ), 14 );
    *ratio_Q14 = silk_LIMIT( *ratio_Q14, 0, 32767 );

    return pred_Q13;
}
