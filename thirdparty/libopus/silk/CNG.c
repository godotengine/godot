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
#include "stack_alloc.h"

/* Generates excitation for CNG LPC synthesis */
static OPUS_INLINE void silk_CNG_exc(
    opus_int32                       exc_Q14[],          /* O    CNG excitation signal Q10                   */
    opus_int32                       exc_buf_Q14[],      /* I    Random samples buffer Q10                   */
    opus_int                         length,             /* I    Length                                      */
    opus_int32                       *rand_seed          /* I/O  Seed to random index generator              */
)
{
    opus_int32 seed;
    opus_int   i, idx, exc_mask;

    exc_mask = CNG_BUF_MASK_MAX;
    while( exc_mask > length ) {
        exc_mask = silk_RSHIFT( exc_mask, 1 );
    }

    seed = *rand_seed;
    for( i = 0; i < length; i++ ) {
        seed = silk_RAND( seed );
        idx = (opus_int)( silk_RSHIFT( seed, 24 ) & exc_mask );
        silk_assert( idx >= 0 );
        silk_assert( idx <= CNG_BUF_MASK_MAX );
        exc_Q14[ i ] = exc_buf_Q14[ idx ];
    }
    *rand_seed = seed;
}

void silk_CNG_Reset(
    silk_decoder_state          *psDec                          /* I/O  Decoder state                               */
)
{
    opus_int i, NLSF_step_Q15, NLSF_acc_Q15;

    NLSF_step_Q15 = silk_DIV32_16( silk_int16_MAX, psDec->LPC_order + 1 );
    NLSF_acc_Q15 = 0;
    for( i = 0; i < psDec->LPC_order; i++ ) {
        NLSF_acc_Q15 += NLSF_step_Q15;
        psDec->sCNG.CNG_smth_NLSF_Q15[ i ] = NLSF_acc_Q15;
    }
    psDec->sCNG.CNG_smth_Gain_Q16 = 0;
    psDec->sCNG.rand_seed = 3176576;
}

/* Updates CNG estimate, and applies the CNG when packet was lost   */
void silk_CNG(
    silk_decoder_state          *psDec,                         /* I/O  Decoder state                               */
    silk_decoder_control        *psDecCtrl,                     /* I/O  Decoder control                             */
    opus_int16                  frame[],                        /* I/O  Signal                                      */
    opus_int                    length                          /* I    Length of residual                          */
)
{
    opus_int   i, subfr;
    opus_int32 LPC_pred_Q10, max_Gain_Q16, gain_Q16, gain_Q10;
    opus_int16 A_Q12[ MAX_LPC_ORDER ];
    silk_CNG_struct *psCNG = &psDec->sCNG;
    SAVE_STACK;

    if( psDec->fs_kHz != psCNG->fs_kHz ) {
        /* Reset state */
        silk_CNG_Reset( psDec );

        psCNG->fs_kHz = psDec->fs_kHz;
    }
    if( psDec->lossCnt == 0 && psDec->prevSignalType == TYPE_NO_VOICE_ACTIVITY ) {
        /* Update CNG parameters */

        /* Smoothing of LSF's  */
        for( i = 0; i < psDec->LPC_order; i++ ) {
            psCNG->CNG_smth_NLSF_Q15[ i ] += silk_SMULWB( (opus_int32)psDec->prevNLSF_Q15[ i ] - (opus_int32)psCNG->CNG_smth_NLSF_Q15[ i ], CNG_NLSF_SMTH_Q16 );
        }
        /* Find the subframe with the highest gain */
        max_Gain_Q16 = 0;
        subfr        = 0;
        for( i = 0; i < psDec->nb_subfr; i++ ) {
            if( psDecCtrl->Gains_Q16[ i ] > max_Gain_Q16 ) {
                max_Gain_Q16 = psDecCtrl->Gains_Q16[ i ];
                subfr        = i;
            }
        }
        /* Update CNG excitation buffer with excitation from this subframe */
        silk_memmove( &psCNG->CNG_exc_buf_Q14[ psDec->subfr_length ], psCNG->CNG_exc_buf_Q14, ( psDec->nb_subfr - 1 ) * psDec->subfr_length * sizeof( opus_int32 ) );
        silk_memcpy(   psCNG->CNG_exc_buf_Q14, &psDec->exc_Q14[ subfr * psDec->subfr_length ], psDec->subfr_length * sizeof( opus_int32 ) );

        /* Smooth gains */
        for( i = 0; i < psDec->nb_subfr; i++ ) {
            psCNG->CNG_smth_Gain_Q16 += silk_SMULWB( psDecCtrl->Gains_Q16[ i ] - psCNG->CNG_smth_Gain_Q16, CNG_GAIN_SMTH_Q16 );
            /* If the smoothed gain is 3 dB greater than this subframe's gain, use this subframe's gain to adapt faster. */
            if( silk_SMULWW( psCNG->CNG_smth_Gain_Q16, CNG_GAIN_SMTH_THRESHOLD_Q16 ) > psDecCtrl->Gains_Q16[ i ] ) {
                psCNG->CNG_smth_Gain_Q16 = psDecCtrl->Gains_Q16[ i ];
            }
        }
    }

    /* Add CNG when packet is lost or during DTX */
    if( psDec->lossCnt ) {
        VARDECL( opus_int32, CNG_sig_Q14 );
        ALLOC( CNG_sig_Q14, length + MAX_LPC_ORDER, opus_int32 );

        /* Generate CNG excitation */
        gain_Q16 = silk_SMULWW( psDec->sPLC.randScale_Q14, psDec->sPLC.prevGain_Q16[1] );
        if( gain_Q16 >= (1 << 21) || psCNG->CNG_smth_Gain_Q16 > (1 << 23) ) {
            gain_Q16 = silk_SMULTT( gain_Q16, gain_Q16 );
            gain_Q16 = silk_SUB_LSHIFT32(silk_SMULTT( psCNG->CNG_smth_Gain_Q16, psCNG->CNG_smth_Gain_Q16 ), gain_Q16, 5 );
            gain_Q16 = silk_LSHIFT32( silk_SQRT_APPROX( gain_Q16 ), 16 );
        } else {
            gain_Q16 = silk_SMULWW( gain_Q16, gain_Q16 );
            gain_Q16 = silk_SUB_LSHIFT32(silk_SMULWW( psCNG->CNG_smth_Gain_Q16, psCNG->CNG_smth_Gain_Q16 ), gain_Q16, 5 );
            gain_Q16 = silk_LSHIFT32( silk_SQRT_APPROX( gain_Q16 ), 8 );
        }
        gain_Q10 = silk_RSHIFT( gain_Q16, 6 );

        silk_CNG_exc( CNG_sig_Q14 + MAX_LPC_ORDER, psCNG->CNG_exc_buf_Q14, length, &psCNG->rand_seed );

        /* Convert CNG NLSF to filter representation */
        silk_NLSF2A( A_Q12, psCNG->CNG_smth_NLSF_Q15, psDec->LPC_order, psDec->arch );

        /* Generate CNG signal, by synthesis filtering */
        silk_memcpy( CNG_sig_Q14, psCNG->CNG_synth_state, MAX_LPC_ORDER * sizeof( opus_int32 ) );
        celt_assert( psDec->LPC_order == 10 || psDec->LPC_order == 16 );
        for( i = 0; i < length; i++ ) {
            /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
            LPC_pred_Q10 = silk_RSHIFT( psDec->LPC_order, 1 );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i -  1 ], A_Q12[ 0 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i -  2 ], A_Q12[ 1 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i -  3 ], A_Q12[ 2 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i -  4 ], A_Q12[ 3 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i -  5 ], A_Q12[ 4 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i -  6 ], A_Q12[ 5 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i -  7 ], A_Q12[ 6 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i -  8 ], A_Q12[ 7 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i -  9 ], A_Q12[ 8 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i - 10 ], A_Q12[ 9 ] );
            if( psDec->LPC_order == 16 ) {
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i - 11 ], A_Q12[ 10 ] );
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i - 12 ], A_Q12[ 11 ] );
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i - 13 ], A_Q12[ 12 ] );
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i - 14 ], A_Q12[ 13 ] );
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i - 15 ], A_Q12[ 14 ] );
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, CNG_sig_Q14[ MAX_LPC_ORDER + i - 16 ], A_Q12[ 15 ] );
            }

            /* Update states */
            CNG_sig_Q14[ MAX_LPC_ORDER + i ] = silk_ADD_SAT32( CNG_sig_Q14[ MAX_LPC_ORDER + i ], silk_LSHIFT_SAT32( LPC_pred_Q10, 4 ) );

            /* Scale with Gain and add to input signal */
            frame[ i ] = (opus_int16)silk_ADD_SAT16( frame[ i ], silk_SAT16( silk_RSHIFT_ROUND( silk_SMULWW( CNG_sig_Q14[ MAX_LPC_ORDER + i ], gain_Q10 ), 8 ) ) );

        }
        silk_memcpy( psCNG->CNG_synth_state, &CNG_sig_Q14[ length ], MAX_LPC_ORDER * sizeof( opus_int32 ) );
    } else {
        silk_memset( psCNG->CNG_synth_state, 0, psDec->LPC_order *  sizeof( opus_int32 ) );
    }
    RESTORE_STACK;
}
