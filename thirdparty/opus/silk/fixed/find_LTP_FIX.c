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

/* Head room for correlations */
#define LTP_CORRS_HEAD_ROOM                             2

void silk_fit_LTP(
    opus_int32 LTP_coefs_Q16[ LTP_ORDER ],
    opus_int16 LTP_coefs_Q14[ LTP_ORDER ]
);

void silk_find_LTP_FIX(
    opus_int16                      b_Q14[ MAX_NB_SUBFR * LTP_ORDER ],      /* O    LTP coefs                                                                   */
    opus_int32                      WLTP[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ], /* O    Weight for LTP quantization                                           */
    opus_int                        *LTPredCodGain_Q7,                      /* O    LTP coding gain                                                             */
    const opus_int16                r_lpc[],                                /* I    residual signal after LPC signal + state for first 10 ms                    */
    const opus_int                  lag[ MAX_NB_SUBFR ],                    /* I    LTP lags                                                                    */
    const opus_int32                Wght_Q15[ MAX_NB_SUBFR ],               /* I    weights                                                                     */
    const opus_int                  subfr_length,                           /* I    subframe length                                                             */
    const opus_int                  nb_subfr,                               /* I    number of subframes                                                         */
    const opus_int                  mem_offset,                             /* I    number of samples in LTP memory                                             */
    opus_int                        corr_rshifts[ MAX_NB_SUBFR ],           /* O    right shifts applied to correlations                                        */
    int                             arch                                    /* I    Run-time architecture                                                       */
)
{
    opus_int   i, k, lshift;
    const opus_int16 *r_ptr, *lag_ptr;
    opus_int16 *b_Q14_ptr;

    opus_int32 regu;
    opus_int32 *WLTP_ptr;
    opus_int32 b_Q16[ LTP_ORDER ], delta_b_Q14[ LTP_ORDER ], d_Q14[ MAX_NB_SUBFR ], nrg[ MAX_NB_SUBFR ], g_Q26;
    opus_int32 w[ MAX_NB_SUBFR ], WLTP_max, max_abs_d_Q14, max_w_bits;

    opus_int32 temp32, denom32;
    opus_int   extra_shifts;
    opus_int   rr_shifts, maxRshifts, maxRshifts_wxtra, LZs;
    opus_int32 LPC_res_nrg, LPC_LTP_res_nrg, div_Q16;
    opus_int32 Rr[ LTP_ORDER ], rr[ MAX_NB_SUBFR ];
    opus_int32 wd, m_Q12;

    b_Q14_ptr = b_Q14;
    WLTP_ptr  = WLTP;
    r_ptr     = &r_lpc[ mem_offset ];
    for( k = 0; k < nb_subfr; k++ ) {
        lag_ptr = r_ptr - ( lag[ k ] + LTP_ORDER / 2 );

        silk_sum_sqr_shift( &rr[ k ], &rr_shifts, r_ptr, subfr_length ); /* rr[ k ] in Q( -rr_shifts ) */

        /* Assure headroom */
        LZs = silk_CLZ32( rr[k] );
        if( LZs < LTP_CORRS_HEAD_ROOM ) {
            rr[ k ] = silk_RSHIFT_ROUND( rr[ k ], LTP_CORRS_HEAD_ROOM - LZs );
            rr_shifts += ( LTP_CORRS_HEAD_ROOM - LZs );
        }
        corr_rshifts[ k ] = rr_shifts;
        silk_corrMatrix_FIX( lag_ptr, subfr_length, LTP_ORDER, LTP_CORRS_HEAD_ROOM, WLTP_ptr, &corr_rshifts[ k ], arch );  /* WLTP_fix_ptr in Q( -corr_rshifts[ k ] ) */

        /* The correlation vector always has lower max abs value than rr and/or RR so head room is assured */
        silk_corrVector_FIX( lag_ptr, r_ptr, subfr_length, LTP_ORDER, Rr, corr_rshifts[ k ], arch );  /* Rr_fix_ptr   in Q( -corr_rshifts[ k ] ) */
        if( corr_rshifts[ k ] > rr_shifts ) {
            rr[ k ] = silk_RSHIFT( rr[ k ], corr_rshifts[ k ] - rr_shifts ); /* rr[ k ] in Q( -corr_rshifts[ k ] ) */
        }
        silk_assert( rr[ k ] >= 0 );

        regu = 1;
        regu = silk_SMLAWB( regu, rr[ k ], SILK_FIX_CONST( LTP_DAMPING/3, 16 ) );
        regu = silk_SMLAWB( regu, matrix_ptr( WLTP_ptr, 0, 0, LTP_ORDER ), SILK_FIX_CONST( LTP_DAMPING/3, 16 ) );
        regu = silk_SMLAWB( regu, matrix_ptr( WLTP_ptr, LTP_ORDER-1, LTP_ORDER-1, LTP_ORDER ), SILK_FIX_CONST( LTP_DAMPING/3, 16 ) );
        silk_regularize_correlations_FIX( WLTP_ptr, &rr[k], regu, LTP_ORDER );

        silk_solve_LDL_FIX( WLTP_ptr, LTP_ORDER, Rr, b_Q16 ); /* WLTP_fix_ptr and Rr_fix_ptr both in Q(-corr_rshifts[k]) */

        /* Limit and store in Q14 */
        silk_fit_LTP( b_Q16, b_Q14_ptr );

        /* Calculate residual energy */
        nrg[ k ] = silk_residual_energy16_covar_FIX( b_Q14_ptr, WLTP_ptr, Rr, rr[ k ], LTP_ORDER, 14 ); /* nrg_fix in Q( -corr_rshifts[ k ] ) */

        /* temp = Wght[ k ] / ( nrg[ k ] * Wght[ k ] + 0.01f * subfr_length ); */
        extra_shifts = silk_min_int( corr_rshifts[ k ], LTP_CORRS_HEAD_ROOM );
        denom32 = silk_LSHIFT_SAT32( silk_SMULWB( nrg[ k ], Wght_Q15[ k ] ), 1 + extra_shifts ) + /* Q( -corr_rshifts[ k ] + extra_shifts ) */
            silk_RSHIFT( silk_SMULWB( (opus_int32)subfr_length, 655 ), corr_rshifts[ k ] - extra_shifts );    /* Q( -corr_rshifts[ k ] + extra_shifts ) */
        denom32 = silk_max( denom32, 1 );
        silk_assert( ((opus_int64)Wght_Q15[ k ] << 16 ) < silk_int32_MAX );                       /* Wght always < 0.5 in Q0 */
        temp32 = silk_DIV32( silk_LSHIFT( (opus_int32)Wght_Q15[ k ], 16 ), denom32 );             /* Q( 15 + 16 + corr_rshifts[k] - extra_shifts ) */
        temp32 = silk_RSHIFT( temp32, 31 + corr_rshifts[ k ] - extra_shifts - 26 );               /* Q26 */

        /* Limit temp such that the below scaling never wraps around */
        WLTP_max = 0;
        for( i = 0; i < LTP_ORDER * LTP_ORDER; i++ ) {
            WLTP_max = silk_max( WLTP_ptr[ i ], WLTP_max );
        }
        lshift = silk_CLZ32( WLTP_max ) - 1 - 3; /* keep 3 bits free for vq_nearest_neighbor_fix */
        silk_assert( 26 - 18 + lshift >= 0 );
        if( 26 - 18 + lshift < 31 ) {
            temp32 = silk_min_32( temp32, silk_LSHIFT( (opus_int32)1, 26 - 18 + lshift ) );
        }

        silk_scale_vector32_Q26_lshift_18( WLTP_ptr, temp32, LTP_ORDER * LTP_ORDER ); /* WLTP_ptr in Q( 18 - corr_rshifts[ k ] ) */

        w[ k ] = matrix_ptr( WLTP_ptr, LTP_ORDER/2, LTP_ORDER/2, LTP_ORDER ); /* w in Q( 18 - corr_rshifts[ k ] ) */
        silk_assert( w[k] >= 0 );

        r_ptr     += subfr_length;
        b_Q14_ptr += LTP_ORDER;
        WLTP_ptr  += LTP_ORDER * LTP_ORDER;
    }

    maxRshifts = 0;
    for( k = 0; k < nb_subfr; k++ ) {
        maxRshifts = silk_max_int( corr_rshifts[ k ], maxRshifts );
    }

    /* Compute LTP coding gain */
    if( LTPredCodGain_Q7 != NULL ) {
        LPC_LTP_res_nrg = 0;
        LPC_res_nrg     = 0;
        silk_assert( LTP_CORRS_HEAD_ROOM >= 2 ); /* Check that no overflow will happen when adding */
        for( k = 0; k < nb_subfr; k++ ) {
            LPC_res_nrg     = silk_ADD32( LPC_res_nrg,     silk_RSHIFT( silk_ADD32( silk_SMULWB(  rr[ k ], Wght_Q15[ k ] ), 1 ), 1 + ( maxRshifts - corr_rshifts[ k ] ) ) ); /* Q( -maxRshifts ) */
            LPC_LTP_res_nrg = silk_ADD32( LPC_LTP_res_nrg, silk_RSHIFT( silk_ADD32( silk_SMULWB( nrg[ k ], Wght_Q15[ k ] ), 1 ), 1 + ( maxRshifts - corr_rshifts[ k ] ) ) ); /* Q( -maxRshifts ) */
        }
        LPC_LTP_res_nrg = silk_max( LPC_LTP_res_nrg, 1 ); /* avoid division by zero */

        div_Q16 = silk_DIV32_varQ( LPC_res_nrg, LPC_LTP_res_nrg, 16 );
        *LTPredCodGain_Q7 = ( opus_int )silk_SMULBB( 3, silk_lin2log( div_Q16 ) - ( 16 << 7 ) );

        silk_assert( *LTPredCodGain_Q7 == ( opus_int )silk_SAT16( silk_MUL( 3, silk_lin2log( div_Q16 ) - ( 16 << 7 ) ) ) );
    }

    /* smoothing */
    /* d = sum( B, 1 ); */
    b_Q14_ptr = b_Q14;
    for( k = 0; k < nb_subfr; k++ ) {
        d_Q14[ k ] = 0;
        for( i = 0; i < LTP_ORDER; i++ ) {
            d_Q14[ k ] += b_Q14_ptr[ i ];
        }
        b_Q14_ptr += LTP_ORDER;
    }

    /* m = ( w * d' ) / ( sum( w ) + 1e-3 ); */

    /* Find maximum absolute value of d_Q14 and the bits used by w in Q0 */
    max_abs_d_Q14 = 0;
    max_w_bits    = 0;
    for( k = 0; k < nb_subfr; k++ ) {
        max_abs_d_Q14 = silk_max_32( max_abs_d_Q14, silk_abs( d_Q14[ k ] ) );
        /* w[ k ] is in Q( 18 - corr_rshifts[ k ] ) */
        /* Find bits needed in Q( 18 - maxRshifts ) */
        max_w_bits = silk_max_32( max_w_bits, 32 - silk_CLZ32( w[ k ] ) + corr_rshifts[ k ] - maxRshifts );
    }

    /* max_abs_d_Q14 = (5 << 15); worst case, i.e. LTP_ORDER * -silk_int16_MIN */
    silk_assert( max_abs_d_Q14 <= ( 5 << 15 ) );

    /* How many bits is needed for w*d' in Q( 18 - maxRshifts ) in the worst case, of all d_Q14's being equal to max_abs_d_Q14 */
    extra_shifts = max_w_bits + 32 - silk_CLZ32( max_abs_d_Q14 ) - 14;

    /* Subtract what we got available; bits in output var plus maxRshifts */
    extra_shifts -= ( 32 - 1 - 2 + maxRshifts ); /* Keep sign bit free as well as 2 bits for accumulation */
    extra_shifts = silk_max_int( extra_shifts, 0 );

    maxRshifts_wxtra = maxRshifts + extra_shifts;

    temp32 = silk_RSHIFT( 262, maxRshifts + extra_shifts ) + 1; /* 1e-3f in Q( 18 - (maxRshifts + extra_shifts) ) */
    wd = 0;
    for( k = 0; k < nb_subfr; k++ ) {
        /* w has at least 2 bits of headroom so no overflow should happen */
        temp32 = silk_ADD32( temp32,                     silk_RSHIFT( w[ k ], maxRshifts_wxtra - corr_rshifts[ k ] ) );                      /* Q( 18 - maxRshifts_wxtra ) */
        wd     = silk_ADD32( wd, silk_LSHIFT( silk_SMULWW( silk_RSHIFT( w[ k ], maxRshifts_wxtra - corr_rshifts[ k ] ), d_Q14[ k ] ), 2 ) ); /* Q( 18 - maxRshifts_wxtra ) */
    }
    m_Q12 = silk_DIV32_varQ( wd, temp32, 12 );

    b_Q14_ptr = b_Q14;
    for( k = 0; k < nb_subfr; k++ ) {
        /* w_fix[ k ] from Q( 18 - corr_rshifts[ k ] ) to Q( 16 ) */
        if( 2 - corr_rshifts[k] > 0 ) {
            temp32 = silk_RSHIFT( w[ k ], 2 - corr_rshifts[ k ] );
        } else {
            temp32 = silk_LSHIFT_SAT32( w[ k ], corr_rshifts[ k ] - 2 );
        }

        g_Q26 = silk_MUL(
            silk_DIV32(
                SILK_FIX_CONST( LTP_SMOOTHING, 26 ),
                silk_RSHIFT( SILK_FIX_CONST( LTP_SMOOTHING, 26 ), 10 ) + temp32 ),                          /* Q10 */
            silk_LSHIFT_SAT32( silk_SUB_SAT32( (opus_int32)m_Q12, silk_RSHIFT( d_Q14[ k ], 2 ) ), 4 ) );    /* Q16 */

        temp32 = 0;
        for( i = 0; i < LTP_ORDER; i++ ) {
            delta_b_Q14[ i ] = silk_max_16( b_Q14_ptr[ i ], 1638 );     /* 1638_Q14 = 0.1_Q0 */
            temp32 += delta_b_Q14[ i ];                                 /* Q14 */
        }
        temp32 = silk_DIV32( g_Q26, temp32 );                           /* Q14 -> Q12 */
        for( i = 0; i < LTP_ORDER; i++ ) {
            b_Q14_ptr[ i ] = silk_LIMIT_32( (opus_int32)b_Q14_ptr[ i ] + silk_SMULWB( silk_LSHIFT_SAT32( temp32, 4 ), delta_b_Q14[ i ] ), -16000, 28000 );
        }
        b_Q14_ptr += LTP_ORDER;
    }
}

void silk_fit_LTP(
    opus_int32 LTP_coefs_Q16[ LTP_ORDER ],
    opus_int16 LTP_coefs_Q14[ LTP_ORDER ]
)
{
    opus_int i;

    for( i = 0; i < LTP_ORDER; i++ ) {
        LTP_coefs_Q14[ i ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( LTP_coefs_Q16[ i ], 2 ) );
    }
}
