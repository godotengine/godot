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

/***********************************************************
* Pitch analyser function
********************************************************** */
#include "SigProc_FIX.h"
#include "pitch_est_defines.h"
#include "stack_alloc.h"
#include "debug.h"
#include "pitch.h"

#define SCRATCH_SIZE    22
#define SF_LENGTH_4KHZ  ( PE_SUBFR_LENGTH_MS * 4 )
#define SF_LENGTH_8KHZ  ( PE_SUBFR_LENGTH_MS * 8 )
#define MIN_LAG_4KHZ    ( PE_MIN_LAG_MS * 4 )
#define MIN_LAG_8KHZ    ( PE_MIN_LAG_MS * 8 )
#define MAX_LAG_4KHZ    ( PE_MAX_LAG_MS * 4 )
#define MAX_LAG_8KHZ    ( PE_MAX_LAG_MS * 8 - 1 )
#define CSTRIDE_4KHZ    ( MAX_LAG_4KHZ + 1 - MIN_LAG_4KHZ )
#define CSTRIDE_8KHZ    ( MAX_LAG_8KHZ + 3 - ( MIN_LAG_8KHZ - 2 ) )
#define D_COMP_MIN      ( MIN_LAG_8KHZ - 3 )
#define D_COMP_MAX      ( MAX_LAG_8KHZ + 4 )
#define D_COMP_STRIDE   ( D_COMP_MAX - D_COMP_MIN )

typedef opus_int32 silk_pe_stage3_vals[ PE_NB_STAGE3_LAGS ];

/************************************************************/
/* Internally used functions                                */
/************************************************************/
static void silk_P_Ana_calc_corr_st3(
    silk_pe_stage3_vals cross_corr_st3[],              /* O 3 DIM correlation array */
    const opus_int16  frame[],                         /* I vector to correlate         */
    opus_int          start_lag,                       /* I lag offset to search around */
    opus_int          sf_length,                       /* I length of a 5 ms subframe   */
    opus_int          nb_subfr,                        /* I number of subframes         */
    opus_int          complexity,                      /* I Complexity setting          */
    int               arch                             /* I Run-time architecture       */
);

static void silk_P_Ana_calc_energy_st3(
    silk_pe_stage3_vals energies_st3[],                /* O 3 DIM energy array */
    const opus_int16  frame[],                         /* I vector to calc energy in    */
    opus_int          start_lag,                       /* I lag offset to search around */
    opus_int          sf_length,                       /* I length of one 5 ms subframe */
    opus_int          nb_subfr,                        /* I number of subframes         */
    opus_int          complexity,                      /* I Complexity setting          */
    int               arch                             /* I Run-time architecture       */
);

/*************************************************************/
/*      FIXED POINT CORE PITCH ANALYSIS FUNCTION             */
/*************************************************************/
opus_int silk_pitch_analysis_core(                  /* O    Voicing estimate: 0 voiced, 1 unvoiced                      */
    const opus_int16            *frame,             /* I    Signal of length PE_FRAME_LENGTH_MS*Fs_kHz                  */
    opus_int                    *pitch_out,         /* O    4 pitch lag values                                          */
    opus_int16                  *lagIndex,          /* O    Lag Index                                                   */
    opus_int8                   *contourIndex,      /* O    Pitch contour Index                                         */
    opus_int                    *LTPCorr_Q15,       /* I/O  Normalized correlation; input: value from previous frame    */
    opus_int                    prevLag,            /* I    Last lag of previous frame; set to zero is unvoiced         */
    const opus_int32            search_thres1_Q16,  /* I    First stage threshold for lag candidates 0 - 1              */
    const opus_int              search_thres2_Q13,  /* I    Final threshold for lag candidates 0 - 1                    */
    const opus_int              Fs_kHz,             /* I    Sample frequency (kHz)                                      */
    const opus_int              complexity,         /* I    Complexity setting, 0-2, where 2 is highest                 */
    const opus_int              nb_subfr,           /* I    number of 5 ms subframes                                    */
    int                         arch                /* I    Run-time architecture                                       */
)
{
    VARDECL( opus_int16, frame_8kHz );
    VARDECL( opus_int16, frame_4kHz );
    opus_int32 filt_state[ 6 ];
    const opus_int16 *input_frame_ptr;
    opus_int   i, k, d, j;
    VARDECL( opus_int16, C );
    VARDECL( opus_int32, xcorr32 );
    const opus_int16 *target_ptr, *basis_ptr;
    opus_int32 cross_corr, normalizer, energy, shift, energy_basis, energy_target;
    opus_int   d_srch[ PE_D_SRCH_LENGTH ], Cmax, length_d_srch, length_d_comp;
    VARDECL( opus_int16, d_comp );
    opus_int32 sum, threshold, lag_counter;
    opus_int   CBimax, CBimax_new, CBimax_old, lag, start_lag, end_lag, lag_new;
    opus_int32 CC[ PE_NB_CBKS_STAGE2_EXT ], CCmax, CCmax_b, CCmax_new_b, CCmax_new;
    VARDECL( silk_pe_stage3_vals, energies_st3 );
    VARDECL( silk_pe_stage3_vals, cross_corr_st3 );
    opus_int   frame_length, frame_length_8kHz, frame_length_4kHz;
    opus_int   sf_length;
    opus_int   min_lag;
    opus_int   max_lag;
    opus_int32 contour_bias_Q15, diff;
    opus_int   nb_cbk_search, cbk_size;
    opus_int32 delta_lag_log2_sqr_Q7, lag_log2_Q7, prevLag_log2_Q7, prev_lag_bias_Q13;
    const opus_int8 *Lag_CB_ptr;
    SAVE_STACK;
    /* Check for valid sampling frequency */
    silk_assert( Fs_kHz == 8 || Fs_kHz == 12 || Fs_kHz == 16 );

    /* Check for valid complexity setting */
    silk_assert( complexity >= SILK_PE_MIN_COMPLEX );
    silk_assert( complexity <= SILK_PE_MAX_COMPLEX );

    silk_assert( search_thres1_Q16 >= 0 && search_thres1_Q16 <= (1<<16) );
    silk_assert( search_thres2_Q13 >= 0 && search_thres2_Q13 <= (1<<13) );

    /* Set up frame lengths max / min lag for the sampling frequency */
    frame_length      = ( PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS ) * Fs_kHz;
    frame_length_4kHz = ( PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS ) * 4;
    frame_length_8kHz = ( PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS ) * 8;
    sf_length         = PE_SUBFR_LENGTH_MS * Fs_kHz;
    min_lag           = PE_MIN_LAG_MS * Fs_kHz;
    max_lag           = PE_MAX_LAG_MS * Fs_kHz - 1;

    /* Resample from input sampled at Fs_kHz to 8 kHz */
    ALLOC( frame_8kHz, frame_length_8kHz, opus_int16 );
    if( Fs_kHz == 16 ) {
        silk_memset( filt_state, 0, 2 * sizeof( opus_int32 ) );
        silk_resampler_down2( filt_state, frame_8kHz, frame, frame_length );
    } else if( Fs_kHz == 12 ) {
        silk_memset( filt_state, 0, 6 * sizeof( opus_int32 ) );
        silk_resampler_down2_3( filt_state, frame_8kHz, frame, frame_length );
    } else {
        silk_assert( Fs_kHz == 8 );
        silk_memcpy( frame_8kHz, frame, frame_length_8kHz * sizeof(opus_int16) );
    }

    /* Decimate again to 4 kHz */
    silk_memset( filt_state, 0, 2 * sizeof( opus_int32 ) );/* Set state to zero */
    ALLOC( frame_4kHz, frame_length_4kHz, opus_int16 );
    silk_resampler_down2( filt_state, frame_4kHz, frame_8kHz, frame_length_8kHz );

    /* Low-pass filter */
    for( i = frame_length_4kHz - 1; i > 0; i-- ) {
        frame_4kHz[ i ] = silk_ADD_SAT16( frame_4kHz[ i ], frame_4kHz[ i - 1 ] );
    }

    /*******************************************************************************
    ** Scale 4 kHz signal down to prevent correlations measures from overflowing
    ** find scaling as max scaling for each 8kHz(?) subframe
    *******************************************************************************/

    /* Inner product is calculated with different lengths, so scale for the worst case */
    silk_sum_sqr_shift( &energy, &shift, frame_4kHz, frame_length_4kHz );
    if( shift > 0 ) {
        shift = silk_RSHIFT( shift, 1 );
        for( i = 0; i < frame_length_4kHz; i++ ) {
            frame_4kHz[ i ] = silk_RSHIFT( frame_4kHz[ i ], shift );
        }
    }

    /******************************************************************************
    * FIRST STAGE, operating in 4 khz
    ******************************************************************************/
    ALLOC( C, nb_subfr * CSTRIDE_8KHZ, opus_int16 );
    ALLOC( xcorr32, MAX_LAG_4KHZ-MIN_LAG_4KHZ+1, opus_int32 );
    silk_memset( C, 0, (nb_subfr >> 1) * CSTRIDE_4KHZ * sizeof( opus_int16 ) );
    target_ptr = &frame_4kHz[ silk_LSHIFT( SF_LENGTH_4KHZ, 2 ) ];
    for( k = 0; k < nb_subfr >> 1; k++ ) {
        /* Check that we are within range of the array */
        silk_assert( target_ptr >= frame_4kHz );
        silk_assert( target_ptr + SF_LENGTH_8KHZ <= frame_4kHz + frame_length_4kHz );

        basis_ptr = target_ptr - MIN_LAG_4KHZ;

        /* Check that we are within range of the array */
        silk_assert( basis_ptr >= frame_4kHz );
        silk_assert( basis_ptr + SF_LENGTH_8KHZ <= frame_4kHz + frame_length_4kHz );

        celt_pitch_xcorr( target_ptr, target_ptr - MAX_LAG_4KHZ, xcorr32, SF_LENGTH_8KHZ, MAX_LAG_4KHZ - MIN_LAG_4KHZ + 1, arch );

        /* Calculate first vector products before loop */
        cross_corr = xcorr32[ MAX_LAG_4KHZ - MIN_LAG_4KHZ ];
        normalizer = silk_inner_prod_aligned( target_ptr, target_ptr, SF_LENGTH_8KHZ, arch );
        normalizer = silk_ADD32( normalizer, silk_inner_prod_aligned( basis_ptr,  basis_ptr, SF_LENGTH_8KHZ, arch ) );
        normalizer = silk_ADD32( normalizer, silk_SMULBB( SF_LENGTH_8KHZ, 4000 ) );

        matrix_ptr( C, k, 0, CSTRIDE_4KHZ ) =
            (opus_int16)silk_DIV32_varQ( cross_corr, normalizer, 13 + 1 );                      /* Q13 */

        /* From now on normalizer is computed recursively */
        for( d = MIN_LAG_4KHZ + 1; d <= MAX_LAG_4KHZ; d++ ) {
            basis_ptr--;

            /* Check that we are within range of the array */
            silk_assert( basis_ptr >= frame_4kHz );
            silk_assert( basis_ptr + SF_LENGTH_8KHZ <= frame_4kHz + frame_length_4kHz );

            cross_corr = xcorr32[ MAX_LAG_4KHZ - d ];

            /* Add contribution of new sample and remove contribution from oldest sample */
            normalizer = silk_ADD32( normalizer,
                silk_SMULBB( basis_ptr[ 0 ], basis_ptr[ 0 ] ) -
                silk_SMULBB( basis_ptr[ SF_LENGTH_8KHZ ], basis_ptr[ SF_LENGTH_8KHZ ] ) );

            matrix_ptr( C, k, d - MIN_LAG_4KHZ, CSTRIDE_4KHZ) =
                (opus_int16)silk_DIV32_varQ( cross_corr, normalizer, 13 + 1 );                  /* Q13 */
        }
        /* Update target pointer */
        target_ptr += SF_LENGTH_8KHZ;
    }

    /* Combine two subframes into single correlation measure and apply short-lag bias */
    if( nb_subfr == PE_MAX_NB_SUBFR ) {
        for( i = MAX_LAG_4KHZ; i >= MIN_LAG_4KHZ; i-- ) {
            sum = (opus_int32)matrix_ptr( C, 0, i - MIN_LAG_4KHZ, CSTRIDE_4KHZ )
                + (opus_int32)matrix_ptr( C, 1, i - MIN_LAG_4KHZ, CSTRIDE_4KHZ );               /* Q14 */
            sum = silk_SMLAWB( sum, sum, silk_LSHIFT( -i, 4 ) );                                /* Q14 */
            C[ i - MIN_LAG_4KHZ ] = (opus_int16)sum;                                            /* Q14 */
        }
    } else {
        /* Only short-lag bias */
        for( i = MAX_LAG_4KHZ; i >= MIN_LAG_4KHZ; i-- ) {
            sum = silk_LSHIFT( (opus_int32)C[ i - MIN_LAG_4KHZ ], 1 );                          /* Q14 */
            sum = silk_SMLAWB( sum, sum, silk_LSHIFT( -i, 4 ) );                                /* Q14 */
            C[ i - MIN_LAG_4KHZ ] = (opus_int16)sum;                                            /* Q14 */
        }
    }

    /* Sort */
    length_d_srch = silk_ADD_LSHIFT32( 4, complexity, 1 );
    silk_assert( 3 * length_d_srch <= PE_D_SRCH_LENGTH );
    silk_insertion_sort_decreasing_int16( C, d_srch, CSTRIDE_4KHZ,
                                          length_d_srch );

    /* Escape if correlation is very low already here */
    Cmax = (opus_int)C[ 0 ];                                                    /* Q14 */
    if( Cmax < SILK_FIX_CONST( 0.2, 14 ) ) {
        silk_memset( pitch_out, 0, nb_subfr * sizeof( opus_int ) );
        *LTPCorr_Q15  = 0;
        *lagIndex     = 0;
        *contourIndex = 0;
        RESTORE_STACK;
        return 1;
    }

    threshold = silk_SMULWB( search_thres1_Q16, Cmax );
    for( i = 0; i < length_d_srch; i++ ) {
        /* Convert to 8 kHz indices for the sorted correlation that exceeds the threshold */
        if( C[ i ] > threshold ) {
            d_srch[ i ] = silk_LSHIFT( d_srch[ i ] + MIN_LAG_4KHZ, 1 );
        } else {
            length_d_srch = i;
            break;
        }
    }
    silk_assert( length_d_srch > 0 );

    ALLOC( d_comp, D_COMP_STRIDE, opus_int16 );
    for( i = D_COMP_MIN; i < D_COMP_MAX; i++ ) {
        d_comp[ i - D_COMP_MIN ] = 0;
    }
    for( i = 0; i < length_d_srch; i++ ) {
        d_comp[ d_srch[ i ] - D_COMP_MIN ] = 1;
    }

    /* Convolution */
    for( i = D_COMP_MAX - 1; i >= MIN_LAG_8KHZ; i-- ) {
        d_comp[ i - D_COMP_MIN ] +=
            d_comp[ i - 1 - D_COMP_MIN ] + d_comp[ i - 2 - D_COMP_MIN ];
    }

    length_d_srch = 0;
    for( i = MIN_LAG_8KHZ; i < MAX_LAG_8KHZ + 1; i++ ) {
        if( d_comp[ i + 1 - D_COMP_MIN ] > 0 ) {
            d_srch[ length_d_srch ] = i;
            length_d_srch++;
        }
    }

    /* Convolution */
    for( i = D_COMP_MAX - 1; i >= MIN_LAG_8KHZ; i-- ) {
        d_comp[ i - D_COMP_MIN ] += d_comp[ i - 1 - D_COMP_MIN ]
            + d_comp[ i - 2 - D_COMP_MIN ] + d_comp[ i - 3 - D_COMP_MIN ];
    }

    length_d_comp = 0;
    for( i = MIN_LAG_8KHZ; i < D_COMP_MAX; i++ ) {
        if( d_comp[ i - D_COMP_MIN ] > 0 ) {
            d_comp[ length_d_comp ] = i - 2;
            length_d_comp++;
        }
    }

    /**********************************************************************************
    ** SECOND STAGE, operating at 8 kHz, on lag sections with high correlation
    *************************************************************************************/

    /******************************************************************************
    ** Scale signal down to avoid correlations measures from overflowing
    *******************************************************************************/
    /* find scaling as max scaling for each subframe */
    silk_sum_sqr_shift( &energy, &shift, frame_8kHz, frame_length_8kHz );
    if( shift > 0 ) {
        shift = silk_RSHIFT( shift, 1 );
        for( i = 0; i < frame_length_8kHz; i++ ) {
            frame_8kHz[ i ] = silk_RSHIFT( frame_8kHz[ i ], shift );
        }
    }

    /*********************************************************************************
    * Find energy of each subframe projected onto its history, for a range of delays
    *********************************************************************************/
    silk_memset( C, 0, nb_subfr * CSTRIDE_8KHZ * sizeof( opus_int16 ) );

    target_ptr = &frame_8kHz[ PE_LTP_MEM_LENGTH_MS * 8 ];
    for( k = 0; k < nb_subfr; k++ ) {

        /* Check that we are within range of the array */
        silk_assert( target_ptr >= frame_8kHz );
        silk_assert( target_ptr + SF_LENGTH_8KHZ <= frame_8kHz + frame_length_8kHz );

        energy_target = silk_ADD32( silk_inner_prod_aligned( target_ptr, target_ptr, SF_LENGTH_8KHZ, arch ), 1 );
        for( j = 0; j < length_d_comp; j++ ) {
            d = d_comp[ j ];
            basis_ptr = target_ptr - d;

            /* Check that we are within range of the array */
            silk_assert( basis_ptr >= frame_8kHz );
            silk_assert( basis_ptr + SF_LENGTH_8KHZ <= frame_8kHz + frame_length_8kHz );

            cross_corr = silk_inner_prod_aligned( target_ptr, basis_ptr, SF_LENGTH_8KHZ, arch );
            if( cross_corr > 0 ) {
                energy_basis = silk_inner_prod_aligned( basis_ptr, basis_ptr, SF_LENGTH_8KHZ, arch );
                matrix_ptr( C, k, d - ( MIN_LAG_8KHZ - 2 ), CSTRIDE_8KHZ ) =
                    (opus_int16)silk_DIV32_varQ( cross_corr,
                                                 silk_ADD32( energy_target,
                                                             energy_basis ),
                                                 13 + 1 );                                      /* Q13 */
            } else {
                matrix_ptr( C, k, d - ( MIN_LAG_8KHZ - 2 ), CSTRIDE_8KHZ ) = 0;
            }
        }
        target_ptr += SF_LENGTH_8KHZ;
    }

    /* search over lag range and lags codebook */
    /* scale factor for lag codebook, as a function of center lag */

    CCmax   = silk_int32_MIN;
    CCmax_b = silk_int32_MIN;

    CBimax = 0; /* To avoid returning undefined lag values */
    lag = -1;   /* To check if lag with strong enough correlation has been found */

    if( prevLag > 0 ) {
        if( Fs_kHz == 12 ) {
            prevLag = silk_DIV32_16( silk_LSHIFT( prevLag, 1 ), 3 );
        } else if( Fs_kHz == 16 ) {
            prevLag = silk_RSHIFT( prevLag, 1 );
        }
        prevLag_log2_Q7 = silk_lin2log( (opus_int32)prevLag );
    } else {
        prevLag_log2_Q7 = 0;
    }
    silk_assert( search_thres2_Q13 == silk_SAT16( search_thres2_Q13 ) );
    /* Set up stage 2 codebook based on number of subframes */
    if( nb_subfr == PE_MAX_NB_SUBFR ) {
        cbk_size   = PE_NB_CBKS_STAGE2_EXT;
        Lag_CB_ptr = &silk_CB_lags_stage2[ 0 ][ 0 ];
        if( Fs_kHz == 8 && complexity > SILK_PE_MIN_COMPLEX ) {
            /* If input is 8 khz use a larger codebook here because it is last stage */
            nb_cbk_search = PE_NB_CBKS_STAGE2_EXT;
        } else {
            nb_cbk_search = PE_NB_CBKS_STAGE2;
        }
    } else {
        cbk_size       = PE_NB_CBKS_STAGE2_10MS;
        Lag_CB_ptr     = &silk_CB_lags_stage2_10_ms[ 0 ][ 0 ];
        nb_cbk_search  = PE_NB_CBKS_STAGE2_10MS;
    }

    for( k = 0; k < length_d_srch; k++ ) {
        d = d_srch[ k ];
        for( j = 0; j < nb_cbk_search; j++ ) {
            CC[ j ] = 0;
            for( i = 0; i < nb_subfr; i++ ) {
                opus_int d_subfr;
                /* Try all codebooks */
                d_subfr = d + matrix_ptr( Lag_CB_ptr, i, j, cbk_size );
                CC[ j ] = CC[ j ]
                    + (opus_int32)matrix_ptr( C, i,
                                              d_subfr - ( MIN_LAG_8KHZ - 2 ),
                                              CSTRIDE_8KHZ );
            }
        }
        /* Find best codebook */
        CCmax_new = silk_int32_MIN;
        CBimax_new = 0;
        for( i = 0; i < nb_cbk_search; i++ ) {
            if( CC[ i ] > CCmax_new ) {
                CCmax_new = CC[ i ];
                CBimax_new = i;
            }
        }

        /* Bias towards shorter lags */
        lag_log2_Q7 = silk_lin2log( d ); /* Q7 */
        silk_assert( lag_log2_Q7 == silk_SAT16( lag_log2_Q7 ) );
        silk_assert( nb_subfr * SILK_FIX_CONST( PE_SHORTLAG_BIAS, 13 ) == silk_SAT16( nb_subfr * SILK_FIX_CONST( PE_SHORTLAG_BIAS, 13 ) ) );
        CCmax_new_b = CCmax_new - silk_RSHIFT( silk_SMULBB( nb_subfr * SILK_FIX_CONST( PE_SHORTLAG_BIAS, 13 ), lag_log2_Q7 ), 7 ); /* Q13 */

        /* Bias towards previous lag */
        silk_assert( nb_subfr * SILK_FIX_CONST( PE_PREVLAG_BIAS, 13 ) == silk_SAT16( nb_subfr * SILK_FIX_CONST( PE_PREVLAG_BIAS, 13 ) ) );
        if( prevLag > 0 ) {
            delta_lag_log2_sqr_Q7 = lag_log2_Q7 - prevLag_log2_Q7;
            silk_assert( delta_lag_log2_sqr_Q7 == silk_SAT16( delta_lag_log2_sqr_Q7 ) );
            delta_lag_log2_sqr_Q7 = silk_RSHIFT( silk_SMULBB( delta_lag_log2_sqr_Q7, delta_lag_log2_sqr_Q7 ), 7 );
            prev_lag_bias_Q13 = silk_RSHIFT( silk_SMULBB( nb_subfr * SILK_FIX_CONST( PE_PREVLAG_BIAS, 13 ), *LTPCorr_Q15 ), 15 ); /* Q13 */
            prev_lag_bias_Q13 = silk_DIV32( silk_MUL( prev_lag_bias_Q13, delta_lag_log2_sqr_Q7 ), delta_lag_log2_sqr_Q7 + SILK_FIX_CONST( 0.5, 7 ) );
            CCmax_new_b -= prev_lag_bias_Q13; /* Q13 */
        }

        if( CCmax_new_b > CCmax_b                                   &&  /* Find maximum biased correlation                  */
            CCmax_new > silk_SMULBB( nb_subfr, search_thres2_Q13 )  &&  /* Correlation needs to be high enough to be voiced */
            silk_CB_lags_stage2[ 0 ][ CBimax_new ] <= MIN_LAG_8KHZ      /* Lag must be in range                             */
         ) {
            CCmax_b = CCmax_new_b;
            CCmax   = CCmax_new;
            lag     = d;
            CBimax  = CBimax_new;
        }
    }

    if( lag == -1 ) {
        /* No suitable candidate found */
        silk_memset( pitch_out, 0, nb_subfr * sizeof( opus_int ) );
        *LTPCorr_Q15  = 0;
        *lagIndex     = 0;
        *contourIndex = 0;
        RESTORE_STACK;
        return 1;
    }

    /* Output normalized correlation */
    *LTPCorr_Q15 = (opus_int)silk_LSHIFT( silk_DIV32_16( CCmax, nb_subfr ), 2 );
    silk_assert( *LTPCorr_Q15 >= 0 );

    if( Fs_kHz > 8 ) {
        VARDECL( opus_int16, scratch_mem );
        /***************************************************************************/
        /* Scale input signal down to avoid correlations measures from overflowing */
        /***************************************************************************/
        /* find scaling as max scaling for each subframe */
        silk_sum_sqr_shift( &energy, &shift, frame, frame_length );
        ALLOC( scratch_mem, shift > 0 ? frame_length : ALLOC_NONE, opus_int16 );
        if( shift > 0 ) {
            /* Move signal to scratch mem because the input signal should be unchanged */
            shift = silk_RSHIFT( shift, 1 );
            for( i = 0; i < frame_length; i++ ) {
                scratch_mem[ i ] = silk_RSHIFT( frame[ i ], shift );
            }
            input_frame_ptr = scratch_mem;
        } else {
            input_frame_ptr = frame;
        }

        /* Search in original signal */

        CBimax_old = CBimax;
        /* Compensate for decimation */
        silk_assert( lag == silk_SAT16( lag ) );
        if( Fs_kHz == 12 ) {
            lag = silk_RSHIFT( silk_SMULBB( lag, 3 ), 1 );
        } else if( Fs_kHz == 16 ) {
            lag = silk_LSHIFT( lag, 1 );
        } else {
            lag = silk_SMULBB( lag, 3 );
        }

        lag = silk_LIMIT_int( lag, min_lag, max_lag );
        start_lag = silk_max_int( lag - 2, min_lag );
        end_lag   = silk_min_int( lag + 2, max_lag );
        lag_new   = lag;                                    /* to avoid undefined lag */
        CBimax    = 0;                                      /* to avoid undefined lag */

        CCmax = silk_int32_MIN;
        /* pitch lags according to second stage */
        for( k = 0; k < nb_subfr; k++ ) {
            pitch_out[ k ] = lag + 2 * silk_CB_lags_stage2[ k ][ CBimax_old ];
        }

        /* Set up codebook parameters according to complexity setting and frame length */
        if( nb_subfr == PE_MAX_NB_SUBFR ) {
            nb_cbk_search   = (opus_int)silk_nb_cbk_searchs_stage3[ complexity ];
            cbk_size        = PE_NB_CBKS_STAGE3_MAX;
            Lag_CB_ptr      = &silk_CB_lags_stage3[ 0 ][ 0 ];
        } else {
            nb_cbk_search   = PE_NB_CBKS_STAGE3_10MS;
            cbk_size        = PE_NB_CBKS_STAGE3_10MS;
            Lag_CB_ptr      = &silk_CB_lags_stage3_10_ms[ 0 ][ 0 ];
        }

        /* Calculate the correlations and energies needed in stage 3 */
        ALLOC( energies_st3, nb_subfr * nb_cbk_search, silk_pe_stage3_vals );
        ALLOC( cross_corr_st3, nb_subfr * nb_cbk_search, silk_pe_stage3_vals );
        silk_P_Ana_calc_corr_st3(  cross_corr_st3, input_frame_ptr, start_lag, sf_length, nb_subfr, complexity, arch );
        silk_P_Ana_calc_energy_st3( energies_st3, input_frame_ptr, start_lag, sf_length, nb_subfr, complexity, arch );

        lag_counter = 0;
        silk_assert( lag == silk_SAT16( lag ) );
        contour_bias_Q15 = silk_DIV32_16( SILK_FIX_CONST( PE_FLATCONTOUR_BIAS, 15 ), lag );

        target_ptr = &input_frame_ptr[ PE_LTP_MEM_LENGTH_MS * Fs_kHz ];
        energy_target = silk_ADD32( silk_inner_prod_aligned( target_ptr, target_ptr, nb_subfr * sf_length, arch ), 1 );
        for( d = start_lag; d <= end_lag; d++ ) {
            for( j = 0; j < nb_cbk_search; j++ ) {
                cross_corr = 0;
                energy     = energy_target;
                for( k = 0; k < nb_subfr; k++ ) {
                    cross_corr = silk_ADD32( cross_corr,
                        matrix_ptr( cross_corr_st3, k, j,
                                    nb_cbk_search )[ lag_counter ] );
                    energy     = silk_ADD32( energy,
                        matrix_ptr( energies_st3, k, j,
                                    nb_cbk_search )[ lag_counter ] );
                    silk_assert( energy >= 0 );
                }
                if( cross_corr > 0 ) {
                    CCmax_new = silk_DIV32_varQ( cross_corr, energy, 13 + 1 );          /* Q13 */
                    /* Reduce depending on flatness of contour */
                    diff = silk_int16_MAX - silk_MUL( contour_bias_Q15, j );            /* Q15 */
                    silk_assert( diff == silk_SAT16( diff ) );
                    CCmax_new = silk_SMULWB( CCmax_new, diff );                         /* Q14 */
                } else {
                    CCmax_new = 0;
                }

                if( CCmax_new > CCmax && ( d + silk_CB_lags_stage3[ 0 ][ j ] ) <= max_lag ) {
                    CCmax   = CCmax_new;
                    lag_new = d;
                    CBimax  = j;
                }
            }
            lag_counter++;
        }

        for( k = 0; k < nb_subfr; k++ ) {
            pitch_out[ k ] = lag_new + matrix_ptr( Lag_CB_ptr, k, CBimax, cbk_size );
            pitch_out[ k ] = silk_LIMIT( pitch_out[ k ], min_lag, PE_MAX_LAG_MS * Fs_kHz );
        }
        *lagIndex = (opus_int16)( lag_new - min_lag);
        *contourIndex = (opus_int8)CBimax;
    } else {        /* Fs_kHz == 8 */
        /* Save Lags */
        for( k = 0; k < nb_subfr; k++ ) {
            pitch_out[ k ] = lag + matrix_ptr( Lag_CB_ptr, k, CBimax, cbk_size );
            pitch_out[ k ] = silk_LIMIT( pitch_out[ k ], MIN_LAG_8KHZ, PE_MAX_LAG_MS * 8 );
        }
        *lagIndex = (opus_int16)( lag - MIN_LAG_8KHZ );
        *contourIndex = (opus_int8)CBimax;
    }
    silk_assert( *lagIndex >= 0 );
    /* return as voiced */
    RESTORE_STACK;
    return 0;
}

/***********************************************************************
 * Calculates the correlations used in stage 3 search. In order to cover
 * the whole lag codebook for all the searched offset lags (lag +- 2),
 * the following correlations are needed in each sub frame:
 *
 * sf1: lag range [-8,...,7] total 16 correlations
 * sf2: lag range [-4,...,4] total 9 correlations
 * sf3: lag range [-3,....4] total 8 correltions
 * sf4: lag range [-6,....8] total 15 correlations
 *
 * In total 48 correlations. The direct implementation computed in worst
 * case 4*12*5 = 240 correlations, but more likely around 120.
 ***********************************************************************/
static void silk_P_Ana_calc_corr_st3(
    silk_pe_stage3_vals cross_corr_st3[],              /* O 3 DIM correlation array */
    const opus_int16  frame[],                         /* I vector to correlate         */
    opus_int          start_lag,                       /* I lag offset to search around */
    opus_int          sf_length,                       /* I length of a 5 ms subframe   */
    opus_int          nb_subfr,                        /* I number of subframes         */
    opus_int          complexity,                      /* I Complexity setting          */
    int               arch                             /* I Run-time architecture       */
)
{
    const opus_int16 *target_ptr;
    opus_int   i, j, k, lag_counter, lag_low, lag_high;
    opus_int   nb_cbk_search, delta, idx, cbk_size;
    VARDECL( opus_int32, scratch_mem );
    VARDECL( opus_int32, xcorr32 );
    const opus_int8 *Lag_range_ptr, *Lag_CB_ptr;
    SAVE_STACK;

    silk_assert( complexity >= SILK_PE_MIN_COMPLEX );
    silk_assert( complexity <= SILK_PE_MAX_COMPLEX );

    if( nb_subfr == PE_MAX_NB_SUBFR ) {
        Lag_range_ptr = &silk_Lag_range_stage3[ complexity ][ 0 ][ 0 ];
        Lag_CB_ptr    = &silk_CB_lags_stage3[ 0 ][ 0 ];
        nb_cbk_search = silk_nb_cbk_searchs_stage3[ complexity ];
        cbk_size      = PE_NB_CBKS_STAGE3_MAX;
    } else {
        silk_assert( nb_subfr == PE_MAX_NB_SUBFR >> 1);
        Lag_range_ptr = &silk_Lag_range_stage3_10_ms[ 0 ][ 0 ];
        Lag_CB_ptr    = &silk_CB_lags_stage3_10_ms[ 0 ][ 0 ];
        nb_cbk_search = PE_NB_CBKS_STAGE3_10MS;
        cbk_size      = PE_NB_CBKS_STAGE3_10MS;
    }
    ALLOC( scratch_mem, SCRATCH_SIZE, opus_int32 );
    ALLOC( xcorr32, SCRATCH_SIZE, opus_int32 );

    target_ptr = &frame[ silk_LSHIFT( sf_length, 2 ) ]; /* Pointer to middle of frame */
    for( k = 0; k < nb_subfr; k++ ) {
        lag_counter = 0;

        /* Calculate the correlations for each subframe */
        lag_low  = matrix_ptr( Lag_range_ptr, k, 0, 2 );
        lag_high = matrix_ptr( Lag_range_ptr, k, 1, 2 );
        silk_assert(lag_high-lag_low+1 <= SCRATCH_SIZE);
        celt_pitch_xcorr( target_ptr, target_ptr - start_lag - lag_high, xcorr32, sf_length, lag_high - lag_low + 1, arch );
        for( j = lag_low; j <= lag_high; j++ ) {
            silk_assert( lag_counter < SCRATCH_SIZE );
            scratch_mem[ lag_counter ] = xcorr32[ lag_high - j ];
            lag_counter++;
        }

        delta = matrix_ptr( Lag_range_ptr, k, 0, 2 );
        for( i = 0; i < nb_cbk_search; i++ ) {
            /* Fill out the 3 dim array that stores the correlations for */
            /* each code_book vector for each start lag */
            idx = matrix_ptr( Lag_CB_ptr, k, i, cbk_size ) - delta;
            for( j = 0; j < PE_NB_STAGE3_LAGS; j++ ) {
                silk_assert( idx + j < SCRATCH_SIZE );
                silk_assert( idx + j < lag_counter );
                matrix_ptr( cross_corr_st3, k, i, nb_cbk_search )[ j ] =
                    scratch_mem[ idx + j ];
            }
        }
        target_ptr += sf_length;
    }
    RESTORE_STACK;
}

/********************************************************************/
/* Calculate the energies for first two subframes. The energies are */
/* calculated recursively.                                          */
/********************************************************************/
static void silk_P_Ana_calc_energy_st3(
    silk_pe_stage3_vals energies_st3[],                 /* O 3 DIM energy array */
    const opus_int16  frame[],                          /* I vector to calc energy in    */
    opus_int          start_lag,                        /* I lag offset to search around */
    opus_int          sf_length,                        /* I length of one 5 ms subframe */
    opus_int          nb_subfr,                         /* I number of subframes         */
    opus_int          complexity,                       /* I Complexity setting          */
    int               arch                              /* I Run-time architecture       */
)
{
    const opus_int16 *target_ptr, *basis_ptr;
    opus_int32 energy;
    opus_int   k, i, j, lag_counter;
    opus_int   nb_cbk_search, delta, idx, cbk_size, lag_diff;
    VARDECL( opus_int32, scratch_mem );
    const opus_int8 *Lag_range_ptr, *Lag_CB_ptr;
    SAVE_STACK;

    silk_assert( complexity >= SILK_PE_MIN_COMPLEX );
    silk_assert( complexity <= SILK_PE_MAX_COMPLEX );

    if( nb_subfr == PE_MAX_NB_SUBFR ) {
        Lag_range_ptr = &silk_Lag_range_stage3[ complexity ][ 0 ][ 0 ];
        Lag_CB_ptr    = &silk_CB_lags_stage3[ 0 ][ 0 ];
        nb_cbk_search = silk_nb_cbk_searchs_stage3[ complexity ];
        cbk_size      = PE_NB_CBKS_STAGE3_MAX;
    } else {
        silk_assert( nb_subfr == PE_MAX_NB_SUBFR >> 1);
        Lag_range_ptr = &silk_Lag_range_stage3_10_ms[ 0 ][ 0 ];
        Lag_CB_ptr    = &silk_CB_lags_stage3_10_ms[ 0 ][ 0 ];
        nb_cbk_search = PE_NB_CBKS_STAGE3_10MS;
        cbk_size      = PE_NB_CBKS_STAGE3_10MS;
    }
    ALLOC( scratch_mem, SCRATCH_SIZE, opus_int32 );

    target_ptr = &frame[ silk_LSHIFT( sf_length, 2 ) ];
    for( k = 0; k < nb_subfr; k++ ) {
        lag_counter = 0;

        /* Calculate the energy for first lag */
        basis_ptr = target_ptr - ( start_lag + matrix_ptr( Lag_range_ptr, k, 0, 2 ) );
        energy = silk_inner_prod_aligned( basis_ptr, basis_ptr, sf_length, arch );
        silk_assert( energy >= 0 );
        scratch_mem[ lag_counter ] = energy;
        lag_counter++;

        lag_diff = ( matrix_ptr( Lag_range_ptr, k, 1, 2 ) -  matrix_ptr( Lag_range_ptr, k, 0, 2 ) + 1 );
        for( i = 1; i < lag_diff; i++ ) {
            /* remove part outside new window */
            energy -= silk_SMULBB( basis_ptr[ sf_length - i ], basis_ptr[ sf_length - i ] );
            silk_assert( energy >= 0 );

            /* add part that comes into window */
            energy = silk_ADD_SAT32( energy, silk_SMULBB( basis_ptr[ -i ], basis_ptr[ -i ] ) );
            silk_assert( energy >= 0 );
            silk_assert( lag_counter < SCRATCH_SIZE );
            scratch_mem[ lag_counter ] = energy;
            lag_counter++;
        }

        delta = matrix_ptr( Lag_range_ptr, k, 0, 2 );
        for( i = 0; i < nb_cbk_search; i++ ) {
            /* Fill out the 3 dim array that stores the correlations for    */
            /* each code_book vector for each start lag                     */
            idx = matrix_ptr( Lag_CB_ptr, k, i, cbk_size ) - delta;
            for( j = 0; j < PE_NB_STAGE3_LAGS; j++ ) {
                silk_assert( idx + j < SCRATCH_SIZE );
                silk_assert( idx + j < lag_counter );
                matrix_ptr( energies_st3, k, i, nb_cbk_search )[ j ] =
                    scratch_mem[ idx + j ];
                silk_assert(
                    matrix_ptr( energies_st3, k, i, nb_cbk_search )[ j ] >= 0 );
            }
        }
        target_ptr += sf_length;
    }
    RESTORE_STACK;
}
