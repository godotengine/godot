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

/* Decode parameters from payload */
void silk_decode_parameters(
    silk_decoder_state          *psDec,                         /* I/O  State                                       */
    silk_decoder_control        *psDecCtrl,                     /* I/O  Decoder control                             */
    opus_int                    condCoding                      /* I    The type of conditional coding to use       */
)
{
    opus_int   i, k, Ix;
    opus_int16 pNLSF_Q15[ MAX_LPC_ORDER ], pNLSF0_Q15[ MAX_LPC_ORDER ];
    const opus_int8 *cbk_ptr_Q7;

    /* Dequant Gains */
    silk_gains_dequant( psDecCtrl->Gains_Q16, psDec->indices.GainsIndices,
        &psDec->LastGainIndex, condCoding == CODE_CONDITIONALLY, psDec->nb_subfr );

    /****************/
    /* Decode NLSFs */
    /****************/
    silk_NLSF_decode( pNLSF_Q15, psDec->indices.NLSFIndices, psDec->psNLSF_CB );

    /* Convert NLSF parameters to AR prediction filter coefficients */
    silk_NLSF2A( psDecCtrl->PredCoef_Q12[ 1 ], pNLSF_Q15, psDec->LPC_order, psDec->arch );

    /* If just reset, e.g., because internal Fs changed, do not allow interpolation */
    /* improves the case of packet loss in the first frame after a switch           */
    if( psDec->first_frame_after_reset == 1 ) {
        psDec->indices.NLSFInterpCoef_Q2 = 4;
    }

    if( psDec->indices.NLSFInterpCoef_Q2 < 4 ) {
        /* Calculation of the interpolated NLSF0 vector from the interpolation factor, */
        /* the previous NLSF1, and the current NLSF1                                   */
        for( i = 0; i < psDec->LPC_order; i++ ) {
            pNLSF0_Q15[ i ] = psDec->prevNLSF_Q15[ i ] + silk_RSHIFT( silk_MUL( psDec->indices.NLSFInterpCoef_Q2,
                pNLSF_Q15[ i ] - psDec->prevNLSF_Q15[ i ] ), 2 );
        }

        /* Convert NLSF parameters to AR prediction filter coefficients */
        silk_NLSF2A( psDecCtrl->PredCoef_Q12[ 0 ], pNLSF0_Q15, psDec->LPC_order, psDec->arch );
    } else {
        /* Copy LPC coefficients for first half from second half */
        silk_memcpy( psDecCtrl->PredCoef_Q12[ 0 ], psDecCtrl->PredCoef_Q12[ 1 ], psDec->LPC_order * sizeof( opus_int16 ) );
    }

    silk_memcpy( psDec->prevNLSF_Q15, pNLSF_Q15, psDec->LPC_order * sizeof( opus_int16 ) );

    /* After a packet loss do BWE of LPC coefs */
    if( psDec->lossCnt ) {
        silk_bwexpander( psDecCtrl->PredCoef_Q12[ 0 ], psDec->LPC_order, BWE_AFTER_LOSS_Q16 );
        silk_bwexpander( psDecCtrl->PredCoef_Q12[ 1 ], psDec->LPC_order, BWE_AFTER_LOSS_Q16 );
    }

    if( psDec->indices.signalType == TYPE_VOICED ) {
        /*********************/
        /* Decode pitch lags */
        /*********************/

        /* Decode pitch values */
        silk_decode_pitch( psDec->indices.lagIndex, psDec->indices.contourIndex, psDecCtrl->pitchL, psDec->fs_kHz, psDec->nb_subfr );

        /* Decode Codebook Index */
        cbk_ptr_Q7 = silk_LTP_vq_ptrs_Q7[ psDec->indices.PERIndex ]; /* set pointer to start of codebook */

        for( k = 0; k < psDec->nb_subfr; k++ ) {
            Ix = psDec->indices.LTPIndex[ k ];
            for( i = 0; i < LTP_ORDER; i++ ) {
                psDecCtrl->LTPCoef_Q14[ k * LTP_ORDER + i ] = silk_LSHIFT( cbk_ptr_Q7[ Ix * LTP_ORDER + i ], 7 );
            }
        }

        /**********************/
        /* Decode LTP scaling */
        /**********************/
        Ix = psDec->indices.LTP_scaleIndex;
        psDecCtrl->LTP_scale_Q14 = silk_LTPScales_table_Q14[ Ix ];
    } else {
        silk_memset( psDecCtrl->pitchL,      0,             psDec->nb_subfr * sizeof( opus_int   ) );
        silk_memset( psDecCtrl->LTPCoef_Q14, 0, LTP_ORDER * psDec->nb_subfr * sizeof( opus_int16 ) );
        psDec->indices.PERIndex  = 0;
        psDecCtrl->LTP_scale_Q14 = 0;
    }
}
