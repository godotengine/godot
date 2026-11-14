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

/* Set decoder sampling rate */
opus_int silk_decoder_set_fs(
    silk_decoder_state          *psDec,                         /* I/O  Decoder state pointer                       */
    opus_int                    fs_kHz,                         /* I    Sampling frequency (kHz)                    */
    opus_int32                  fs_API_Hz                       /* I    API Sampling frequency (Hz)                 */
)
{
    opus_int frame_length, ret = 0;

    celt_assert( fs_kHz == 8 || fs_kHz == 12 || fs_kHz == 16 );
    celt_assert( psDec->nb_subfr == MAX_NB_SUBFR || psDec->nb_subfr == MAX_NB_SUBFR/2 );

    /* New (sub)frame length */
    psDec->subfr_length = silk_SMULBB( SUB_FRAME_LENGTH_MS, fs_kHz );
    frame_length = silk_SMULBB( psDec->nb_subfr, psDec->subfr_length );

    /* Initialize resampler when switching internal or external sampling frequency */
    if( psDec->fs_kHz != fs_kHz || psDec->fs_API_hz != fs_API_Hz ) {
        /* Initialize the resampler for dec_API.c preparing resampling from fs_kHz to API_fs_Hz */
        ret += silk_resampler_init( &psDec->resampler_state, silk_SMULBB( fs_kHz, 1000 ), fs_API_Hz, 0 );

        psDec->fs_API_hz = fs_API_Hz;
    }

    if( psDec->fs_kHz != fs_kHz || frame_length != psDec->frame_length ) {
        if( fs_kHz == 8 ) {
            if( psDec->nb_subfr == MAX_NB_SUBFR ) {
                psDec->pitch_contour_iCDF = silk_pitch_contour_NB_iCDF;
            } else {
                psDec->pitch_contour_iCDF = silk_pitch_contour_10_ms_NB_iCDF;
            }
        } else {
            if( psDec->nb_subfr == MAX_NB_SUBFR ) {
                psDec->pitch_contour_iCDF = silk_pitch_contour_iCDF;
            } else {
                psDec->pitch_contour_iCDF = silk_pitch_contour_10_ms_iCDF;
            }
        }
        if( psDec->fs_kHz != fs_kHz ) {
            psDec->ltp_mem_length = silk_SMULBB( LTP_MEM_LENGTH_MS, fs_kHz );
            if( fs_kHz == 8 || fs_kHz == 12 ) {
                psDec->LPC_order = MIN_LPC_ORDER;
                psDec->psNLSF_CB = &silk_NLSF_CB_NB_MB;
            } else {
                psDec->LPC_order = MAX_LPC_ORDER;
                psDec->psNLSF_CB = &silk_NLSF_CB_WB;
            }
            if( fs_kHz == 16 ) {
                psDec->pitch_lag_low_bits_iCDF = silk_uniform8_iCDF;
            } else if( fs_kHz == 12 ) {
                psDec->pitch_lag_low_bits_iCDF = silk_uniform6_iCDF;
            } else if( fs_kHz == 8 ) {
                psDec->pitch_lag_low_bits_iCDF = silk_uniform4_iCDF;
            } else {
                /* unsupported sampling rate */
                celt_assert( 0 );
            }
            psDec->first_frame_after_reset = 1;
            psDec->lagPrev                 = 100;
            psDec->LastGainIndex           = 10;
            psDec->prevSignalType          = TYPE_NO_VOICE_ACTIVITY;
            silk_memset( psDec->outBuf, 0, sizeof(psDec->outBuf));
            silk_memset( psDec->sLPC_Q14_buf, 0, sizeof(psDec->sLPC_Q14_buf) );
        }

        psDec->fs_kHz       = fs_kHz;
        psDec->frame_length = frame_length;
    }

    /* Check that settings are valid */
    celt_assert( psDec->frame_length > 0 && psDec->frame_length <= MAX_FRAME_LENGTH );

    return ret;
}
