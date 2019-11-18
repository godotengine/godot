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

/*
 * Matrix of resampling methods used:
 *                                 Fs_out (kHz)
 *                        8      12     16     24     48
 *
 *               8        C      UF     U      UF     UF
 *              12        AF     C      UF     U      UF
 * Fs_in (kHz)  16        D      AF     C      UF     UF
 *              24        AF     D      AF     C      U
 *              48        AF     AF     AF     D      C
 *
 * C   -> Copy (no resampling)
 * D   -> Allpass-based 2x downsampling
 * U   -> Allpass-based 2x upsampling
 * UF  -> Allpass-based 2x upsampling followed by FIR interpolation
 * AF  -> AR2 filter followed by FIR interpolation
 */

#include "resampler_private.h"

/* Tables with delay compensation values to equalize total delay for different modes */
static const opus_int8 delay_matrix_enc[ 5 ][ 3 ] = {
/* in  \ out  8  12  16 */
/*  8 */   {  6,  0,  3 },
/* 12 */   {  0,  7,  3 },
/* 16 */   {  0,  1, 10 },
/* 24 */   {  0,  2,  6 },
/* 48 */   { 18, 10, 12 }
};

static const opus_int8 delay_matrix_dec[ 3 ][ 5 ] = {
/* in  \ out  8  12  16  24  48 */
/*  8 */   {  4,  0,  2,  0,  0 },
/* 12 */   {  0,  9,  4,  7,  4 },
/* 16 */   {  0,  3, 12,  7,  7 }
};

/* Simple way to make [8000, 12000, 16000, 24000, 48000] to [0, 1, 2, 3, 4] */
#define rateID(R) ( ( ( ((R)>>12) - ((R)>16000) ) >> ((R)>24000) ) - 1 )

#define USE_silk_resampler_copy                     (0)
#define USE_silk_resampler_private_up2_HQ_wrapper   (1)
#define USE_silk_resampler_private_IIR_FIR          (2)
#define USE_silk_resampler_private_down_FIR         (3)

/* Initialize/reset the resampler state for a given pair of input/output sampling rates */
opus_int silk_resampler_init(
    silk_resampler_state_struct *S,                 /* I/O  Resampler state                                             */
    opus_int32                  Fs_Hz_in,           /* I    Input sampling rate (Hz)                                    */
    opus_int32                  Fs_Hz_out,          /* I    Output sampling rate (Hz)                                   */
    opus_int                    forEnc              /* I    If 1: encoder; if 0: decoder                                */
)
{
    opus_int up2x;

    /* Clear state */
    silk_memset( S, 0, sizeof( silk_resampler_state_struct ) );

    /* Input checking */
    if( forEnc ) {
        if( ( Fs_Hz_in  != 8000 && Fs_Hz_in  != 12000 && Fs_Hz_in  != 16000 && Fs_Hz_in  != 24000 && Fs_Hz_in  != 48000 ) ||
            ( Fs_Hz_out != 8000 && Fs_Hz_out != 12000 && Fs_Hz_out != 16000 ) ) {
            silk_assert( 0 );
            return -1;
        }
        S->inputDelay = delay_matrix_enc[ rateID( Fs_Hz_in ) ][ rateID( Fs_Hz_out ) ];
    } else {
        if( ( Fs_Hz_in  != 8000 && Fs_Hz_in  != 12000 && Fs_Hz_in  != 16000 ) ||
            ( Fs_Hz_out != 8000 && Fs_Hz_out != 12000 && Fs_Hz_out != 16000 && Fs_Hz_out != 24000 && Fs_Hz_out != 48000 ) ) {
            silk_assert( 0 );
            return -1;
        }
        S->inputDelay = delay_matrix_dec[ rateID( Fs_Hz_in ) ][ rateID( Fs_Hz_out ) ];
    }

    S->Fs_in_kHz  = silk_DIV32_16( Fs_Hz_in,  1000 );
    S->Fs_out_kHz = silk_DIV32_16( Fs_Hz_out, 1000 );

    /* Number of samples processed per batch */
    S->batchSize = S->Fs_in_kHz * RESAMPLER_MAX_BATCH_SIZE_MS;

    /* Find resampler with the right sampling ratio */
    up2x = 0;
    if( Fs_Hz_out > Fs_Hz_in ) {
        /* Upsample */
        if( Fs_Hz_out == silk_MUL( Fs_Hz_in, 2 ) ) {                            /* Fs_out : Fs_in = 2 : 1 */
            /* Special case: directly use 2x upsampler */
            S->resampler_function = USE_silk_resampler_private_up2_HQ_wrapper;
        } else {
            /* Default resampler */
            S->resampler_function = USE_silk_resampler_private_IIR_FIR;
            up2x = 1;
        }
    } else if ( Fs_Hz_out < Fs_Hz_in ) {
        /* Downsample */
         S->resampler_function = USE_silk_resampler_private_down_FIR;
        if( silk_MUL( Fs_Hz_out, 4 ) == silk_MUL( Fs_Hz_in, 3 ) ) {             /* Fs_out : Fs_in = 3 : 4 */
            S->FIR_Fracs = 3;
            S->FIR_Order = RESAMPLER_DOWN_ORDER_FIR0;
            S->Coefs = silk_Resampler_3_4_COEFS;
        } else if( silk_MUL( Fs_Hz_out, 3 ) == silk_MUL( Fs_Hz_in, 2 ) ) {      /* Fs_out : Fs_in = 2 : 3 */
            S->FIR_Fracs = 2;
            S->FIR_Order = RESAMPLER_DOWN_ORDER_FIR0;
            S->Coefs = silk_Resampler_2_3_COEFS;
        } else if( silk_MUL( Fs_Hz_out, 2 ) == Fs_Hz_in ) {                     /* Fs_out : Fs_in = 1 : 2 */
            S->FIR_Fracs = 1;
            S->FIR_Order = RESAMPLER_DOWN_ORDER_FIR1;
            S->Coefs = silk_Resampler_1_2_COEFS;
        } else if( silk_MUL( Fs_Hz_out, 3 ) == Fs_Hz_in ) {                     /* Fs_out : Fs_in = 1 : 3 */
            S->FIR_Fracs = 1;
            S->FIR_Order = RESAMPLER_DOWN_ORDER_FIR2;
            S->Coefs = silk_Resampler_1_3_COEFS;
        } else if( silk_MUL( Fs_Hz_out, 4 ) == Fs_Hz_in ) {                     /* Fs_out : Fs_in = 1 : 4 */
            S->FIR_Fracs = 1;
            S->FIR_Order = RESAMPLER_DOWN_ORDER_FIR2;
            S->Coefs = silk_Resampler_1_4_COEFS;
        } else if( silk_MUL( Fs_Hz_out, 6 ) == Fs_Hz_in ) {                     /* Fs_out : Fs_in = 1 : 6 */
            S->FIR_Fracs = 1;
            S->FIR_Order = RESAMPLER_DOWN_ORDER_FIR2;
            S->Coefs = silk_Resampler_1_6_COEFS;
        } else {
            /* None available */
            silk_assert( 0 );
            return -1;
        }
    } else {
        /* Input and output sampling rates are equal: copy */
        S->resampler_function = USE_silk_resampler_copy;
    }

    /* Ratio of input/output samples */
    S->invRatio_Q16 = silk_LSHIFT32( silk_DIV32( silk_LSHIFT32( Fs_Hz_in, 14 + up2x ), Fs_Hz_out ), 2 );
    /* Make sure the ratio is rounded up */
    while( silk_SMULWW( S->invRatio_Q16, Fs_Hz_out ) < silk_LSHIFT32( Fs_Hz_in, up2x ) ) {
        S->invRatio_Q16++;
    }

    return 0;
}

/* Resampler: convert from one sampling rate to another */
/* Input and output sampling rate are at most 48000 Hz  */
opus_int silk_resampler(
    silk_resampler_state_struct *S,                 /* I/O  Resampler state                                             */
    opus_int16                  out[],              /* O    Output signal                                               */
    const opus_int16            in[],               /* I    Input signal                                                */
    opus_int32                  inLen               /* I    Number of input samples                                     */
)
{
    opus_int nSamples;

    /* Need at least 1 ms of input data */
    silk_assert( inLen >= S->Fs_in_kHz );
    /* Delay can't exceed the 1 ms of buffering */
    silk_assert( S->inputDelay <= S->Fs_in_kHz );

    nSamples = S->Fs_in_kHz - S->inputDelay;

    /* Copy to delay buffer */
    silk_memcpy( &S->delayBuf[ S->inputDelay ], in, nSamples * sizeof( opus_int16 ) );

    switch( S->resampler_function ) {
        case USE_silk_resampler_private_up2_HQ_wrapper:
            silk_resampler_private_up2_HQ_wrapper( S, out, S->delayBuf, S->Fs_in_kHz );
            silk_resampler_private_up2_HQ_wrapper( S, &out[ S->Fs_out_kHz ], &in[ nSamples ], inLen - S->Fs_in_kHz );
            break;
        case USE_silk_resampler_private_IIR_FIR:
            silk_resampler_private_IIR_FIR( S, out, S->delayBuf, S->Fs_in_kHz );
            silk_resampler_private_IIR_FIR( S, &out[ S->Fs_out_kHz ], &in[ nSamples ], inLen - S->Fs_in_kHz );
            break;
        case USE_silk_resampler_private_down_FIR:
            silk_resampler_private_down_FIR( S, out, S->delayBuf, S->Fs_in_kHz );
            silk_resampler_private_down_FIR( S, &out[ S->Fs_out_kHz ], &in[ nSamples ], inLen - S->Fs_in_kHz );
            break;
        default:
            silk_memcpy( out, S->delayBuf, S->Fs_in_kHz * sizeof( opus_int16 ) );
            silk_memcpy( &out[ S->Fs_out_kHz ], &in[ nSamples ], ( inLen - S->Fs_in_kHz ) * sizeof( opus_int16 ) );
    }

    /* Copy to delay buffer */
    silk_memcpy( S->delayBuf, &in[ inLen - S->inputDelay ], S->inputDelay * sizeof( opus_int16 ) );

    return 0;
}
