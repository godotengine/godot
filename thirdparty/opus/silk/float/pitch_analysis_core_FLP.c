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

/*****************************************************************************
* Pitch analyser function
******************************************************************************/
#include "SigProc_FLP.h"
#include "SigProc_FIX.h"
#include "pitch_est_defines.h"
#include "pitch.h"

#define SCRATCH_SIZE        22

/************************************************************/
/* Internally used functions                                */
/************************************************************/
static void silk_P_Ana_calc_corr_st3(
    silk_float cross_corr_st3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ][ PE_NB_STAGE3_LAGS ], /* O 3 DIM correlation array */
    const silk_float    frame[],            /* I vector to correlate                                            */
    opus_int            start_lag,          /* I start lag                                                      */
    opus_int            sf_length,          /* I sub frame length                                               */
    opus_int            nb_subfr,           /* I number of subframes                                            */
    opus_int            complexity,         /* I Complexity setting                                             */
    int                 arch                /* I Run-time architecture                                          */
);

static void silk_P_Ana_calc_energy_st3(
    silk_float energies_st3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ][ PE_NB_STAGE3_LAGS ], /* O 3 DIM correlation array */
    const silk_float    frame[],            /* I vector to correlate                                            */
    opus_int            start_lag,          /* I start lag                                                      */
    opus_int            sf_length,          /* I sub frame length                                               */
    opus_int            nb_subfr,           /* I number of subframes                                            */
    opus_int            complexity          /* I Complexity setting                                             */
);

/************************************************************/
/* CORE PITCH ANALYSIS FUNCTION                             */
/************************************************************/
opus_int silk_pitch_analysis_core_FLP(      /* O    Voicing estimate: 0 voiced, 1 unvoiced                      */
    const silk_float    *frame,             /* I    Signal of length PE_FRAME_LENGTH_MS*Fs_kHz                  */
    opus_int            *pitch_out,         /* O    Pitch lag values [nb_subfr]                                 */
    opus_int16          *lagIndex,          /* O    Lag Index                                                   */
    opus_int8           *contourIndex,      /* O    Pitch contour Index                                         */
    silk_float          *LTPCorr,           /* I/O  Normalized correlation; input: value from previous frame    */
    opus_int            prevLag,            /* I    Last lag of previous frame; set to zero is unvoiced         */
    const silk_float    search_thres1,      /* I    First stage threshold for lag candidates 0 - 1              */
    const silk_float    search_thres2,      /* I    Final threshold for lag candidates 0 - 1                    */
    const opus_int      Fs_kHz,             /* I    sample frequency (kHz)                                      */
    const opus_int      complexity,         /* I    Complexity setting, 0-2, where 2 is highest                 */
    const opus_int      nb_subfr,           /* I    Number of 5 ms subframes                                    */
    int                 arch                /* I    Run-time architecture                                       */
)
{
    opus_int   i, k, d, j;
    silk_float frame_8kHz[  PE_MAX_FRAME_LENGTH_MS * 8 ];
    silk_float frame_4kHz[  PE_MAX_FRAME_LENGTH_MS * 4 ];
    opus_int16 frame_8_FIX[ PE_MAX_FRAME_LENGTH_MS * 8 ];
    opus_int16 frame_4_FIX[ PE_MAX_FRAME_LENGTH_MS * 4 ];
    opus_int32 filt_state[ 6 ];
    silk_float threshold, contour_bias;
    silk_float C[ PE_MAX_NB_SUBFR][ (PE_MAX_LAG >> 1) + 5 ];
    opus_val32 xcorr[ PE_MAX_LAG_MS * 4 - PE_MIN_LAG_MS * 4 + 1 ];
    silk_float CC[ PE_NB_CBKS_STAGE2_EXT ];
    const silk_float *target_ptr, *basis_ptr;
    double    cross_corr, normalizer, energy, energy_tmp;
    opus_int   d_srch[ PE_D_SRCH_LENGTH ];
    opus_int16 d_comp[ (PE_MAX_LAG >> 1) + 5 ];
    opus_int   length_d_srch, length_d_comp;
    silk_float Cmax, CCmax, CCmax_b, CCmax_new_b, CCmax_new;
    opus_int   CBimax, CBimax_new, lag, start_lag, end_lag, lag_new;
    opus_int   cbk_size;
    silk_float lag_log2, prevLag_log2, delta_lag_log2_sqr;
    silk_float energies_st3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ][ PE_NB_STAGE3_LAGS ];
    silk_float cross_corr_st3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ][ PE_NB_STAGE3_LAGS ];
    opus_int   lag_counter;
    opus_int   frame_length, frame_length_8kHz, frame_length_4kHz;
    opus_int   sf_length, sf_length_8kHz, sf_length_4kHz;
    opus_int   min_lag, min_lag_8kHz, min_lag_4kHz;
    opus_int   max_lag, max_lag_8kHz, max_lag_4kHz;
    opus_int   nb_cbk_search;
    const opus_int8 *Lag_CB_ptr;

    /* Check for valid sampling frequency */
    silk_assert( Fs_kHz == 8 || Fs_kHz == 12 || Fs_kHz == 16 );

    /* Check for valid complexity setting */
    silk_assert( complexity >= SILK_PE_MIN_COMPLEX );
    silk_assert( complexity <= SILK_PE_MAX_COMPLEX );

    silk_assert( search_thres1 >= 0.0f && search_thres1 <= 1.0f );
    silk_assert( search_thres2 >= 0.0f && search_thres2 <= 1.0f );

    /* Set up frame lengths max / min lag for the sampling frequency */
    frame_length      = ( PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS ) * Fs_kHz;
    frame_length_4kHz = ( PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS ) * 4;
    frame_length_8kHz = ( PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS ) * 8;
    sf_length         = PE_SUBFR_LENGTH_MS * Fs_kHz;
    sf_length_4kHz    = PE_SUBFR_LENGTH_MS * 4;
    sf_length_8kHz    = PE_SUBFR_LENGTH_MS * 8;
    min_lag           = PE_MIN_LAG_MS * Fs_kHz;
    min_lag_4kHz      = PE_MIN_LAG_MS * 4;
    min_lag_8kHz      = PE_MIN_LAG_MS * 8;
    max_lag           = PE_MAX_LAG_MS * Fs_kHz - 1;
    max_lag_4kHz      = PE_MAX_LAG_MS * 4;
    max_lag_8kHz      = PE_MAX_LAG_MS * 8 - 1;

    /* Resample from input sampled at Fs_kHz to 8 kHz */
    if( Fs_kHz == 16 ) {
        /* Resample to 16 -> 8 khz */
        opus_int16 frame_16_FIX[ 16 * PE_MAX_FRAME_LENGTH_MS ];
        silk_float2short_array( frame_16_FIX, frame, frame_length );
        silk_memset( filt_state, 0, 2 * sizeof( opus_int32 ) );
        silk_resampler_down2( filt_state, frame_8_FIX, frame_16_FIX, frame_length );
        silk_short2float_array( frame_8kHz, frame_8_FIX, frame_length_8kHz );
    } else if( Fs_kHz == 12 ) {
        /* Resample to 12 -> 8 khz */
        opus_int16 frame_12_FIX[ 12 * PE_MAX_FRAME_LENGTH_MS ];
        silk_float2short_array( frame_12_FIX, frame, frame_length );
        silk_memset( filt_state, 0, 6 * sizeof( opus_int32 ) );
        silk_resampler_down2_3( filt_state, frame_8_FIX, frame_12_FIX, frame_length );
        silk_short2float_array( frame_8kHz, frame_8_FIX, frame_length_8kHz );
    } else {
        silk_assert( Fs_kHz == 8 );
        silk_float2short_array( frame_8_FIX, frame, frame_length_8kHz );
    }

    /* Decimate again to 4 kHz */
    silk_memset( filt_state, 0, 2 * sizeof( opus_int32 ) );
    silk_resampler_down2( filt_state, frame_4_FIX, frame_8_FIX, frame_length_8kHz );
    silk_short2float_array( frame_4kHz, frame_4_FIX, frame_length_4kHz );

    /* Low-pass filter */
    for( i = frame_length_4kHz - 1; i > 0; i-- ) {
        frame_4kHz[ i ] += frame_4kHz[ i - 1 ];
    }

    /******************************************************************************
    * FIRST STAGE, operating in 4 khz
    ******************************************************************************/
    silk_memset(C, 0, sizeof(silk_float) * nb_subfr * ((PE_MAX_LAG >> 1) + 5));
    target_ptr = &frame_4kHz[ silk_LSHIFT( sf_length_4kHz, 2 ) ];
    for( k = 0; k < nb_subfr >> 1; k++ ) {
        /* Check that we are within range of the array */
        silk_assert( target_ptr >= frame_4kHz );
        silk_assert( target_ptr + sf_length_8kHz <= frame_4kHz + frame_length_4kHz );

        basis_ptr = target_ptr - min_lag_4kHz;

        /* Check that we are within range of the array */
        silk_assert( basis_ptr >= frame_4kHz );
        silk_assert( basis_ptr + sf_length_8kHz <= frame_4kHz + frame_length_4kHz );

        celt_pitch_xcorr( target_ptr, target_ptr-max_lag_4kHz, xcorr, sf_length_8kHz, max_lag_4kHz - min_lag_4kHz + 1, arch );

        /* Calculate first vector products before loop */
        cross_corr = xcorr[ max_lag_4kHz - min_lag_4kHz ];
        normalizer = silk_energy_FLP( target_ptr, sf_length_8kHz ) +
                     silk_energy_FLP( basis_ptr,  sf_length_8kHz ) +
                     sf_length_8kHz * 4000.0f;

        C[ 0 ][ min_lag_4kHz ] += (silk_float)( 2 * cross_corr / normalizer );

        /* From now on normalizer is computed recursively */
        for( d = min_lag_4kHz + 1; d <= max_lag_4kHz; d++ ) {
            basis_ptr--;

            /* Check that we are within range of the array */
            silk_assert( basis_ptr >= frame_4kHz );
            silk_assert( basis_ptr + sf_length_8kHz <= frame_4kHz + frame_length_4kHz );

            cross_corr = xcorr[ max_lag_4kHz - d ];

            /* Add contribution of new sample and remove contribution from oldest sample */
            normalizer +=
                basis_ptr[ 0 ] * (double)basis_ptr[ 0 ] -
                basis_ptr[ sf_length_8kHz ] * (double)basis_ptr[ sf_length_8kHz ];
            C[ 0 ][ d ] += (silk_float)( 2 * cross_corr / normalizer );
        }
        /* Update target pointer */
        target_ptr += sf_length_8kHz;
    }

    /* Apply short-lag bias */
    for( i = max_lag_4kHz; i >= min_lag_4kHz; i-- ) {
        C[ 0 ][ i ] -= C[ 0 ][ i ] * i / 4096.0f;
    }

    /* Sort */
    length_d_srch = 4 + 2 * complexity;
    silk_assert( 3 * length_d_srch <= PE_D_SRCH_LENGTH );
    silk_insertion_sort_decreasing_FLP( &C[ 0 ][ min_lag_4kHz ], d_srch, max_lag_4kHz - min_lag_4kHz + 1, length_d_srch );

    /* Escape if correlation is very low already here */
    Cmax = C[ 0 ][ min_lag_4kHz ];
    if( Cmax < 0.2f ) {
        silk_memset( pitch_out, 0, nb_subfr * sizeof( opus_int ) );
        *LTPCorr      = 0.0f;
        *lagIndex     = 0;
        *contourIndex = 0;
        return 1;
    }

    threshold = search_thres1 * Cmax;
    for( i = 0; i < length_d_srch; i++ ) {
        /* Convert to 8 kHz indices for the sorted correlation that exceeds the threshold */
        if( C[ 0 ][ min_lag_4kHz + i ] > threshold ) {
            d_srch[ i ] = silk_LSHIFT( d_srch[ i ] + min_lag_4kHz, 1 );
        } else {
            length_d_srch = i;
            break;
        }
    }
    silk_assert( length_d_srch > 0 );

    for( i = min_lag_8kHz - 5; i < max_lag_8kHz + 5; i++ ) {
        d_comp[ i ] = 0;
    }
    for( i = 0; i < length_d_srch; i++ ) {
        d_comp[ d_srch[ i ] ] = 1;
    }

    /* Convolution */
    for( i = max_lag_8kHz + 3; i >= min_lag_8kHz; i-- ) {
        d_comp[ i ] += d_comp[ i - 1 ] + d_comp[ i - 2 ];
    }

    length_d_srch = 0;
    for( i = min_lag_8kHz; i < max_lag_8kHz + 1; i++ ) {
        if( d_comp[ i + 1 ] > 0 ) {
            d_srch[ length_d_srch ] = i;
            length_d_srch++;
        }
    }

    /* Convolution */
    for( i = max_lag_8kHz + 3; i >= min_lag_8kHz; i-- ) {
        d_comp[ i ] += d_comp[ i - 1 ] + d_comp[ i - 2 ] + d_comp[ i - 3 ];
    }

    length_d_comp = 0;
    for( i = min_lag_8kHz; i < max_lag_8kHz + 4; i++ ) {
        if( d_comp[ i ] > 0 ) {
            d_comp[ length_d_comp ] = (opus_int16)( i - 2 );
            length_d_comp++;
        }
    }

    /**********************************************************************************
    ** SECOND STAGE, operating at 8 kHz, on lag sections with high correlation
    *************************************************************************************/
    /*********************************************************************************
    * Find energy of each subframe projected onto its history, for a range of delays
    *********************************************************************************/
    silk_memset( C, 0, PE_MAX_NB_SUBFR*((PE_MAX_LAG >> 1) + 5) * sizeof(silk_float));

    if( Fs_kHz == 8 ) {
        target_ptr = &frame[ PE_LTP_MEM_LENGTH_MS * 8 ];
    } else {
        target_ptr = &frame_8kHz[ PE_LTP_MEM_LENGTH_MS * 8 ];
    }
    for( k = 0; k < nb_subfr; k++ ) {
        energy_tmp = silk_energy_FLP( target_ptr, sf_length_8kHz ) + 1.0;
        for( j = 0; j < length_d_comp; j++ ) {
            d = d_comp[ j ];
            basis_ptr = target_ptr - d;
            cross_corr = silk_inner_product_FLP( basis_ptr, target_ptr, sf_length_8kHz );
            if( cross_corr > 0.0f ) {
                energy = silk_energy_FLP( basis_ptr, sf_length_8kHz );
                C[ k ][ d ] = (silk_float)( 2 * cross_corr / ( energy + energy_tmp ) );
            } else {
                C[ k ][ d ] = 0.0f;
            }
        }
        target_ptr += sf_length_8kHz;
    }

    /* search over lag range and lags codebook */
    /* scale factor for lag codebook, as a function of center lag */

    CCmax   = 0.0f; /* This value doesn't matter */
    CCmax_b = -1000.0f;

    CBimax = 0; /* To avoid returning undefined lag values */
    lag = -1;   /* To check if lag with strong enough correlation has been found */

    if( prevLag > 0 ) {
        if( Fs_kHz == 12 ) {
            prevLag = silk_LSHIFT( prevLag, 1 ) / 3;
        } else if( Fs_kHz == 16 ) {
            prevLag = silk_RSHIFT( prevLag, 1 );
        }
        prevLag_log2 = silk_log2( (silk_float)prevLag );
    } else {
        prevLag_log2 = 0;
    }

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
            CC[j] = 0.0f;
            for( i = 0; i < nb_subfr; i++ ) {
                /* Try all codebooks */
                CC[ j ] += C[ i ][ d + matrix_ptr( Lag_CB_ptr, i, j, cbk_size )];
            }
        }
        /* Find best codebook */
        CCmax_new  = -1000.0f;
        CBimax_new = 0;
        for( i = 0; i < nb_cbk_search; i++ ) {
            if( CC[ i ] > CCmax_new ) {
                CCmax_new = CC[ i ];
                CBimax_new = i;
            }
        }

        /* Bias towards shorter lags */
        lag_log2 = silk_log2( (silk_float)d );
        CCmax_new_b = CCmax_new - PE_SHORTLAG_BIAS * nb_subfr * lag_log2;

        /* Bias towards previous lag */
        if( prevLag > 0 ) {
            delta_lag_log2_sqr = lag_log2 - prevLag_log2;
            delta_lag_log2_sqr *= delta_lag_log2_sqr;
            CCmax_new_b -= PE_PREVLAG_BIAS * nb_subfr * (*LTPCorr) * delta_lag_log2_sqr / ( delta_lag_log2_sqr + 0.5f );
        }

        if( CCmax_new_b > CCmax_b &&                /* Find maximum biased correlation                  */
            CCmax_new > nb_subfr * search_thres2    /* Correlation needs to be high enough to be voiced */
        ) {
            CCmax_b = CCmax_new_b;
            CCmax   = CCmax_new;
            lag     = d;
            CBimax  = CBimax_new;
        }
    }

    if( lag == -1 ) {
        /* No suitable candidate found */
        silk_memset( pitch_out, 0, PE_MAX_NB_SUBFR * sizeof(opus_int) );
        *LTPCorr      = 0.0f;
        *lagIndex     = 0;
        *contourIndex = 0;
        return 1;
    }

    /* Output normalized correlation */
    *LTPCorr = (silk_float)( CCmax / nb_subfr );
    silk_assert( *LTPCorr >= 0.0f );

    if( Fs_kHz > 8 ) {
        /* Search in original signal */

        /* Compensate for decimation */
        silk_assert( lag == silk_SAT16( lag ) );
        if( Fs_kHz == 12 ) {
            lag = silk_RSHIFT_ROUND( silk_SMULBB( lag, 3 ), 1 );
        } else { /* Fs_kHz == 16 */
            lag = silk_LSHIFT( lag, 1 );
        }

        lag = silk_LIMIT_int( lag, min_lag, max_lag );
        start_lag = silk_max_int( lag - 2, min_lag );
        end_lag   = silk_min_int( lag + 2, max_lag );
        lag_new   = lag;                                    /* to avoid undefined lag */
        CBimax    = 0;                                      /* to avoid undefined lag */

        CCmax = -1000.0f;

        /* Calculate the correlations and energies needed in stage 3 */
        silk_P_Ana_calc_corr_st3( cross_corr_st3, frame, start_lag, sf_length, nb_subfr, complexity, arch );
        silk_P_Ana_calc_energy_st3( energies_st3, frame, start_lag, sf_length, nb_subfr, complexity );

        lag_counter = 0;
        silk_assert( lag == silk_SAT16( lag ) );
        contour_bias = PE_FLATCONTOUR_BIAS / lag;

        /* Set up cbk parameters according to complexity setting and frame length */
        if( nb_subfr == PE_MAX_NB_SUBFR ) {
            nb_cbk_search = (opus_int)silk_nb_cbk_searchs_stage3[ complexity ];
            cbk_size      = PE_NB_CBKS_STAGE3_MAX;
            Lag_CB_ptr    = &silk_CB_lags_stage3[ 0 ][ 0 ];
        } else {
            nb_cbk_search = PE_NB_CBKS_STAGE3_10MS;
            cbk_size      = PE_NB_CBKS_STAGE3_10MS;
            Lag_CB_ptr    = &silk_CB_lags_stage3_10_ms[ 0 ][ 0 ];
        }

        target_ptr = &frame[ PE_LTP_MEM_LENGTH_MS * Fs_kHz ];
        energy_tmp = silk_energy_FLP( target_ptr, nb_subfr * sf_length ) + 1.0;
        for( d = start_lag; d <= end_lag; d++ ) {
            for( j = 0; j < nb_cbk_search; j++ ) {
                cross_corr = 0.0;
                energy = energy_tmp;
                for( k = 0; k < nb_subfr; k++ ) {
                    cross_corr += cross_corr_st3[ k ][ j ][ lag_counter ];
                    energy     +=   energies_st3[ k ][ j ][ lag_counter ];
                }
                if( cross_corr > 0.0 ) {
                    CCmax_new = (silk_float)( 2 * cross_corr / energy );
                    /* Reduce depending on flatness of contour */
                    CCmax_new *= 1.0f - contour_bias * j;
                } else {
                    CCmax_new = 0.0f;
                }

                if( CCmax_new > CCmax && ( d + (opus_int)silk_CB_lags_stage3[ 0 ][ j ] ) <= max_lag ) {
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
        *lagIndex = (opus_int16)( lag_new - min_lag );
        *contourIndex = (opus_int8)CBimax;
    } else {        /* Fs_kHz == 8 */
        /* Save Lags */
        for( k = 0; k < nb_subfr; k++ ) {
            pitch_out[ k ] = lag + matrix_ptr( Lag_CB_ptr, k, CBimax, cbk_size );
            pitch_out[ k ] = silk_LIMIT( pitch_out[ k ], min_lag_8kHz, PE_MAX_LAG_MS * 8 );
        }
        *lagIndex = (opus_int16)( lag - min_lag_8kHz );
        *contourIndex = (opus_int8)CBimax;
    }
    silk_assert( *lagIndex >= 0 );
    /* return as voiced */
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
    silk_float cross_corr_st3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ][ PE_NB_STAGE3_LAGS ], /* O 3 DIM correlation array */
    const silk_float    frame[],            /* I vector to correlate                                            */
    opus_int            start_lag,          /* I start lag                                                      */
    opus_int            sf_length,          /* I sub frame length                                               */
    opus_int            nb_subfr,           /* I number of subframes                                            */
    opus_int            complexity,         /* I Complexity setting                                             */
    int                 arch                /* I Run-time architecture                                          */
)
{
    const silk_float *target_ptr;
    opus_int   i, j, k, lag_counter, lag_low, lag_high;
    opus_int   nb_cbk_search, delta, idx, cbk_size;
    silk_float scratch_mem[ SCRATCH_SIZE ];
    opus_val32 xcorr[ SCRATCH_SIZE ];
    const opus_int8 *Lag_range_ptr, *Lag_CB_ptr;

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

    target_ptr = &frame[ silk_LSHIFT( sf_length, 2 ) ]; /* Pointer to middle of frame */
    for( k = 0; k < nb_subfr; k++ ) {
        lag_counter = 0;

        /* Calculate the correlations for each subframe */
        lag_low  = matrix_ptr( Lag_range_ptr, k, 0, 2 );
        lag_high = matrix_ptr( Lag_range_ptr, k, 1, 2 );
        silk_assert(lag_high-lag_low+1 <= SCRATCH_SIZE);
        celt_pitch_xcorr( target_ptr, target_ptr - start_lag - lag_high, xcorr, sf_length, lag_high - lag_low + 1, arch );
        for( j = lag_low; j <= lag_high; j++ ) {
            silk_assert( lag_counter < SCRATCH_SIZE );
            scratch_mem[ lag_counter ] = xcorr[ lag_high - j ];
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
                cross_corr_st3[ k ][ i ][ j ] = scratch_mem[ idx + j ];
            }
        }
        target_ptr += sf_length;
    }
}

/********************************************************************/
/* Calculate the energies for first two subframes. The energies are */
/* calculated recursively.                                          */
/********************************************************************/
static void silk_P_Ana_calc_energy_st3(
    silk_float energies_st3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ][ PE_NB_STAGE3_LAGS ], /* O 3 DIM correlation array */
    const silk_float    frame[],            /* I vector to correlate                                            */
    opus_int            start_lag,          /* I start lag                                                      */
    opus_int            sf_length,          /* I sub frame length                                               */
    opus_int            nb_subfr,           /* I number of subframes                                            */
    opus_int            complexity          /* I Complexity setting                                             */
)
{
    const silk_float *target_ptr, *basis_ptr;
    double    energy;
    opus_int   k, i, j, lag_counter;
    opus_int   nb_cbk_search, delta, idx, cbk_size, lag_diff;
    silk_float scratch_mem[ SCRATCH_SIZE ];
    const opus_int8 *Lag_range_ptr, *Lag_CB_ptr;

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

    target_ptr = &frame[ silk_LSHIFT( sf_length, 2 ) ];
    for( k = 0; k < nb_subfr; k++ ) {
        lag_counter = 0;

        /* Calculate the energy for first lag */
        basis_ptr = target_ptr - ( start_lag + matrix_ptr( Lag_range_ptr, k, 0, 2 ) );
        energy = silk_energy_FLP( basis_ptr, sf_length ) + 1e-3;
        silk_assert( energy >= 0.0 );
        scratch_mem[lag_counter] = (silk_float)energy;
        lag_counter++;

        lag_diff = ( matrix_ptr( Lag_range_ptr, k, 1, 2 ) -  matrix_ptr( Lag_range_ptr, k, 0, 2 ) + 1 );
        for( i = 1; i < lag_diff; i++ ) {
            /* remove part outside new window */
            energy -= basis_ptr[sf_length - i] * (double)basis_ptr[sf_length - i];
            silk_assert( energy >= 0.0 );

            /* add part that comes into window */
            energy += basis_ptr[ -i ] * (double)basis_ptr[ -i ];
            silk_assert( energy >= 0.0 );
            silk_assert( lag_counter < SCRATCH_SIZE );
            scratch_mem[lag_counter] = (silk_float)energy;
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
                energies_st3[ k ][ i ][ j ] = scratch_mem[ idx + j ];
                silk_assert( energies_st3[ k ][ i ][ j ] >= 0.0f );
            }
        }
        target_ptr += sf_length;
    }
}
