/* Copyright (c) 2014, Cisco Systems, INC
   Written by XiangMingZhu WeiZhou MinPeng YanWang

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include "main.h"
#include "celt/x86/x86cpu.h"
#include "stack_alloc.h"

static OPUS_INLINE void silk_nsq_scale_states_sse4_1(
    const silk_encoder_state *psEncC,           /* I    Encoder State                   */
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                       */
    const opus_int32    x_Q3[],                 /* I    input in Q3                     */
    opus_int32          x_sc_Q10[],             /* O    input scaled with 1/Gain        */
    const opus_int16    sLTP[],                 /* I    re-whitened LTP state in Q0     */
    opus_int32          sLTP_Q15[],             /* O    LTP state matching scaled input */
    opus_int            subfr,                  /* I    subframe number                 */
    const opus_int      LTP_scale_Q14,          /* I                                    */
    const opus_int32    Gains_Q16[ MAX_NB_SUBFR ], /* I                                 */
    const opus_int      pitchL[ MAX_NB_SUBFR ], /* I    Pitch lag                       */
    const opus_int      signal_type             /* I    Signal type                     */
);

static OPUS_INLINE void silk_noise_shape_quantizer_10_16_sse4_1(
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                       */
    opus_int            signalType,             /* I    Signal type                     */
    const opus_int32    x_sc_Q10[],             /* I                                    */
    opus_int8           pulses[],               /* O                                    */
    opus_int16          xq[],                   /* O                                    */
    opus_int32          sLTP_Q15[],             /* I/O  LTP state                       */
    const opus_int16    a_Q12[],                /* I    Short term prediction coefs     */
    const opus_int16    b_Q14[],                /* I    Long term prediction coefs      */
    const opus_int16    AR_shp_Q13[],           /* I    Noise shaping AR coefs          */
    opus_int            lag,                    /* I    Pitch lag                       */
    opus_int32          HarmShapeFIRPacked_Q14, /* I                                    */
    opus_int            Tilt_Q14,               /* I    Spectral tilt                   */
    opus_int32          LF_shp_Q14,             /* I                                    */
    opus_int32          Gain_Q16,               /* I                                    */
    opus_int            offset_Q10,             /* I                                    */
    opus_int            length,                 /* I    Input length                    */
    opus_int32          table[][4]              /* I                                    */
);

void silk_NSQ_sse4_1(
    const silk_encoder_state    *psEncC,                                    /* I/O  Encoder State                   */
    silk_nsq_state              *NSQ,                                       /* I/O  NSQ state                       */
    SideInfoIndices             *psIndices,                                 /* I/O  Quantization Indices            */
    const opus_int32            x_Q3[],                                     /* I    Prefiltered input signal        */
    opus_int8                   pulses[],                                   /* O    Quantized pulse signal          */
    const opus_int16            PredCoef_Q12[ 2 * MAX_LPC_ORDER ],          /* I    Short term prediction coefs     */
    const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],    /* I    Long term prediction coefs      */
    const opus_int16            AR2_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I Noise shaping coefs             */
    const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],          /* I    Long term shaping coefs         */
    const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                   /* I    Spectral tilt                   */
    const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                 /* I    Low frequency shaping coefs     */
    const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                  /* I    Quantization step sizes         */
    const opus_int              pitchL[ MAX_NB_SUBFR ],                     /* I    Pitch lags                      */
    const opus_int              Lambda_Q10,                                 /* I    Rate/distortion tradeoff        */
    const opus_int              LTP_scale_Q14                               /* I    LTP state scaling               */
)
{
    opus_int            k, lag, start_idx, LSF_interpolation_flag;
    const opus_int16    *A_Q12, *B_Q14, *AR_shp_Q13;
    opus_int16          *pxq;
    VARDECL( opus_int32, sLTP_Q15 );
    VARDECL( opus_int16, sLTP );
    opus_int32          HarmShapeFIRPacked_Q14;
    opus_int            offset_Q10;
    VARDECL( opus_int32, x_sc_Q10 );

    opus_int32   table[ 64 ][ 4 ];
    opus_int32   tmp1;
    opus_int32   q1_Q10, q2_Q10, rd1_Q20, rd2_Q20;

    SAVE_STACK;

    NSQ->rand_seed = psIndices->Seed;

    /* Set unvoiced lag to the previous one, overwrite later for voiced */
    lag = NSQ->lagPrev;

    silk_assert( NSQ->prev_gain_Q16 != 0 );

    offset_Q10 = silk_Quantization_Offsets_Q10[ psIndices->signalType >> 1 ][ psIndices->quantOffsetType ];

    /* 0 */
    q1_Q10  = offset_Q10;
    q2_Q10  = offset_Q10 + ( 1024 - QUANT_LEVEL_ADJUST_Q10 );
    rd1_Q20 = q1_Q10 * Lambda_Q10;
    rd2_Q20 = q2_Q10 * Lambda_Q10;

    table[ 32 ][ 0 ] = q1_Q10;
    table[ 32 ][ 1 ] = q2_Q10;
    table[ 32 ][ 2 ] = 2 * (q1_Q10 - q2_Q10);
    table[ 32 ][ 3 ] = (rd1_Q20 - rd2_Q20) + (q1_Q10 * q1_Q10 - q2_Q10 * q2_Q10);

    /* -1 */
    q1_Q10  = offset_Q10 - ( 1024 - QUANT_LEVEL_ADJUST_Q10 );
    q2_Q10  = offset_Q10;
    rd1_Q20 = - q1_Q10 * Lambda_Q10;
    rd2_Q20 = q2_Q10 * Lambda_Q10;

    table[ 31 ][ 0 ] = q1_Q10;
    table[ 31 ][ 1 ] = q2_Q10;
    table[ 31 ][ 2 ] = 2 * (q1_Q10 - q2_Q10);
    table[ 31 ][ 3 ] = (rd1_Q20 - rd2_Q20) + (q1_Q10 * q1_Q10 - q2_Q10 * q2_Q10);

    /* > 0 */
    for (k = 1; k <= 31; k++)
    {
        tmp1 = offset_Q10 + silk_LSHIFT( k, 10 );

        q1_Q10  = tmp1 - QUANT_LEVEL_ADJUST_Q10;
        q2_Q10  = tmp1 - QUANT_LEVEL_ADJUST_Q10 + 1024;
        rd1_Q20 = q1_Q10 * Lambda_Q10;
        rd2_Q20 = q2_Q10 * Lambda_Q10;

        table[ 32 + k ][ 0 ] = q1_Q10;
        table[ 32 + k ][ 1 ] = q2_Q10;
        table[ 32 + k ][ 2 ] = 2 * (q1_Q10 - q2_Q10);
        table[ 32 + k ][ 3 ] = (rd1_Q20 - rd2_Q20) + (q1_Q10 * q1_Q10 - q2_Q10 * q2_Q10);
    }

    /* < -1 */
    for (k = -32; k <= -2; k++)
    {
        tmp1 = offset_Q10 + silk_LSHIFT( k, 10 );

        q1_Q10  = tmp1 + QUANT_LEVEL_ADJUST_Q10;
        q2_Q10  = tmp1 + QUANT_LEVEL_ADJUST_Q10 + 1024;
        rd1_Q20 = - q1_Q10 * Lambda_Q10;
        rd2_Q20 = - q2_Q10 * Lambda_Q10;

        table[ 32 + k ][ 0 ] = q1_Q10;
        table[ 32 + k ][ 1 ] = q2_Q10;
        table[ 32 + k ][ 2 ] = 2 * (q1_Q10 - q2_Q10);
        table[ 32 + k ][ 3 ] = (rd1_Q20 - rd2_Q20) + (q1_Q10 * q1_Q10 - q2_Q10 * q2_Q10);
    }

    if( psIndices->NLSFInterpCoef_Q2 == 4 ) {
        LSF_interpolation_flag = 0;
    } else {
        LSF_interpolation_flag = 1;
    }

    ALLOC( sLTP_Q15,
           psEncC->ltp_mem_length + psEncC->frame_length, opus_int32 );
    ALLOC( sLTP, psEncC->ltp_mem_length + psEncC->frame_length, opus_int16 );
    ALLOC( x_sc_Q10, psEncC->subfr_length, opus_int32 );
    /* Set up pointers to start of sub frame */
    NSQ->sLTP_shp_buf_idx = psEncC->ltp_mem_length;
    NSQ->sLTP_buf_idx     = psEncC->ltp_mem_length;
    pxq                   = &NSQ->xq[ psEncC->ltp_mem_length ];
    for( k = 0; k < psEncC->nb_subfr; k++ ) {
        A_Q12      = &PredCoef_Q12[ (( k >> 1 ) | ( 1 - LSF_interpolation_flag )) * MAX_LPC_ORDER ];
        B_Q14      = &LTPCoef_Q14[ k * LTP_ORDER ];
        AR_shp_Q13 = &AR2_Q13[     k * MAX_SHAPE_LPC_ORDER ];

        /* Noise shape parameters */
        silk_assert( HarmShapeGain_Q14[ k ] >= 0 );
        HarmShapeFIRPacked_Q14  =                          silk_RSHIFT( HarmShapeGain_Q14[ k ], 2 );
        HarmShapeFIRPacked_Q14 |= silk_LSHIFT( (opus_int32)silk_RSHIFT( HarmShapeGain_Q14[ k ], 1 ), 16 );

        NSQ->rewhite_flag = 0;
        if( psIndices->signalType == TYPE_VOICED ) {
            /* Voiced */
            lag = pitchL[ k ];

            /* Re-whitening */
            if( ( k & ( 3 - silk_LSHIFT( LSF_interpolation_flag, 1 ) ) ) == 0 ) {
                /* Rewhiten with new A coefs */
                start_idx = psEncC->ltp_mem_length - lag - psEncC->predictLPCOrder - LTP_ORDER / 2;
                silk_assert( start_idx > 0 );

                silk_LPC_analysis_filter( &sLTP[ start_idx ], &NSQ->xq[ start_idx + k * psEncC->subfr_length ],
                    A_Q12, psEncC->ltp_mem_length - start_idx, psEncC->predictLPCOrder, psEncC->arch );

                NSQ->rewhite_flag = 1;
                NSQ->sLTP_buf_idx = psEncC->ltp_mem_length;
            }
        }

        silk_nsq_scale_states_sse4_1( psEncC, NSQ, x_Q3, x_sc_Q10, sLTP, sLTP_Q15, k, LTP_scale_Q14, Gains_Q16, pitchL, psIndices->signalType );

        if ( opus_likely( ( 10 == psEncC->shapingLPCOrder ) && ( 16 == psEncC->predictLPCOrder) ) )
        {
            silk_noise_shape_quantizer_10_16_sse4_1( NSQ, psIndices->signalType, x_sc_Q10, pulses, pxq, sLTP_Q15, A_Q12, B_Q14,
                AR_shp_Q13, lag, HarmShapeFIRPacked_Q14, Tilt_Q14[ k ], LF_shp_Q14[ k ], Gains_Q16[ k ],
                offset_Q10, psEncC->subfr_length, &(table[32]) );
        }
        else
        {
            silk_noise_shape_quantizer( NSQ, psIndices->signalType, x_sc_Q10, pulses, pxq, sLTP_Q15, A_Q12, B_Q14,
                AR_shp_Q13, lag, HarmShapeFIRPacked_Q14, Tilt_Q14[ k ], LF_shp_Q14[ k ], Gains_Q16[ k ], Lambda_Q10,
                offset_Q10, psEncC->subfr_length, psEncC->shapingLPCOrder, psEncC->predictLPCOrder, psEncC->arch );
        }

        x_Q3   += psEncC->subfr_length;
        pulses += psEncC->subfr_length;
        pxq    += psEncC->subfr_length;
    }

    /* Update lagPrev for next frame */
    NSQ->lagPrev = pitchL[ psEncC->nb_subfr - 1 ];

    /* Save quantized speech and noise shaping signals */
    /* DEBUG_STORE_DATA( enc.pcm, &NSQ->xq[ psEncC->ltp_mem_length ], psEncC->frame_length * sizeof( opus_int16 ) ) */
    silk_memmove( NSQ->xq,           &NSQ->xq[           psEncC->frame_length ], psEncC->ltp_mem_length * sizeof( opus_int16 ) );
    silk_memmove( NSQ->sLTP_shp_Q14, &NSQ->sLTP_shp_Q14[ psEncC->frame_length ], psEncC->ltp_mem_length * sizeof( opus_int32 ) );
    RESTORE_STACK;
}

/***********************************/
/* silk_noise_shape_quantizer_10_16  */
/***********************************/
static OPUS_INLINE void silk_noise_shape_quantizer_10_16_sse4_1(
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                       */
    opus_int            signalType,             /* I    Signal type                     */
    const opus_int32    x_sc_Q10[],             /* I                                    */
    opus_int8           pulses[],               /* O                                    */
    opus_int16          xq[],                   /* O                                    */
    opus_int32          sLTP_Q15[],             /* I/O  LTP state                       */
    const opus_int16    a_Q12[],                /* I    Short term prediction coefs     */
    const opus_int16    b_Q14[],                /* I    Long term prediction coefs      */
    const opus_int16    AR_shp_Q13[],           /* I    Noise shaping AR coefs          */
    opus_int            lag,                    /* I    Pitch lag                       */
    opus_int32          HarmShapeFIRPacked_Q14, /* I                                    */
    opus_int            Tilt_Q14,               /* I    Spectral tilt                   */
    opus_int32          LF_shp_Q14,             /* I                                    */
    opus_int32          Gain_Q16,               /* I                                    */
    opus_int            offset_Q10,             /* I                                    */
    opus_int            length,                 /* I    Input length                    */
    opus_int32          table[][4]              /* I                                    */
)
{
    opus_int     i;
    opus_int32   LTP_pred_Q13, LPC_pred_Q10, n_AR_Q12, n_LTP_Q13;
    opus_int32   n_LF_Q12, r_Q10, q1_Q0, q1_Q10, q2_Q10;
    opus_int32   exc_Q14, LPC_exc_Q14, xq_Q14, Gain_Q10;
    opus_int32   tmp1, tmp2, sLF_AR_shp_Q14;
    opus_int32   *psLPC_Q14, *shp_lag_ptr, *pred_lag_ptr;

    __m128i xmm_tempa, xmm_tempb;

    __m128i xmm_one;

    __m128i psLPC_Q14_hi_01234567, psLPC_Q14_hi_89ABCDEF;
    __m128i psLPC_Q14_lo_01234567, psLPC_Q14_lo_89ABCDEF;
    __m128i a_Q12_01234567,        a_Q12_89ABCDEF;

    __m128i sAR2_Q14_hi_76543210, sAR2_Q14_lo_76543210;
    __m128i AR_shp_Q13_76543210;

    shp_lag_ptr  = &NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2 ];
    pred_lag_ptr = &sLTP_Q15[ NSQ->sLTP_buf_idx - lag + LTP_ORDER / 2 ];
    Gain_Q10     = silk_RSHIFT( Gain_Q16, 6 );

    /* Set up short term AR state */
    psLPC_Q14 = &NSQ->sLPC_Q14[ NSQ_LPC_BUF_LENGTH - 1 ];

    sLF_AR_shp_Q14 = NSQ->sLF_AR_shp_Q14;
    xq_Q14         = psLPC_Q14[ 0 ];
    LTP_pred_Q13   = 0;

    /* load a_Q12 */
    xmm_one = _mm_set_epi8( 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14 );

    /* load a_Q12[0] - a_Q12[7] */
    a_Q12_01234567 = _mm_loadu_si128( (__m128i *)(&a_Q12[ 0 ] ) );
    /* load a_Q12[ 8 ] - a_Q12[ 15 ] */
    a_Q12_89ABCDEF = _mm_loadu_si128( (__m128i *)(&a_Q12[ 8 ] ) );

    a_Q12_01234567 = _mm_shuffle_epi8( a_Q12_01234567, xmm_one );
    a_Q12_89ABCDEF = _mm_shuffle_epi8( a_Q12_89ABCDEF, xmm_one );

    /* load AR_shp_Q13 */
    AR_shp_Q13_76543210 = _mm_loadu_si128( (__m128i *)(&AR_shp_Q13[0] ) );

    /* load psLPC_Q14 */
    xmm_one = _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0 );

    xmm_tempa = _mm_loadu_si128( (__m128i *)(&psLPC_Q14[-16]) );
    xmm_tempb = _mm_loadu_si128( (__m128i *)(&psLPC_Q14[-12]) );

    xmm_tempa = _mm_shuffle_epi8( xmm_tempa, xmm_one );
    xmm_tempb = _mm_shuffle_epi8( xmm_tempb, xmm_one );

    psLPC_Q14_hi_89ABCDEF = _mm_unpackhi_epi64( xmm_tempa, xmm_tempb );
    psLPC_Q14_lo_89ABCDEF = _mm_unpacklo_epi64( xmm_tempa, xmm_tempb );

    xmm_tempa = _mm_loadu_si128( (__m128i *)(&psLPC_Q14[ -8 ]) );
    xmm_tempb = _mm_loadu_si128( (__m128i *)(&psLPC_Q14[ -4 ]) );

    xmm_tempa = _mm_shuffle_epi8( xmm_tempa, xmm_one );
    xmm_tempb = _mm_shuffle_epi8( xmm_tempb, xmm_one );

    psLPC_Q14_hi_01234567 = _mm_unpackhi_epi64( xmm_tempa, xmm_tempb );
    psLPC_Q14_lo_01234567 = _mm_unpacklo_epi64( xmm_tempa, xmm_tempb );

    /* load sAR2_Q14 */
    xmm_tempa = _mm_loadu_si128( (__m128i *)(&(NSQ->sAR2_Q14[ 0 ]) ) );
    xmm_tempb = _mm_loadu_si128( (__m128i *)(&(NSQ->sAR2_Q14[ 4 ]) ) );

    xmm_tempa = _mm_shuffle_epi8( xmm_tempa, xmm_one );
    xmm_tempb = _mm_shuffle_epi8( xmm_tempb, xmm_one );

    sAR2_Q14_hi_76543210 = _mm_unpackhi_epi64( xmm_tempa, xmm_tempb );
    sAR2_Q14_lo_76543210 = _mm_unpacklo_epi64( xmm_tempa, xmm_tempb );

    /* prepare 1 in 8 * 16bit */
    xmm_one = _mm_set1_epi16(1);

    for( i = 0; i < length; i++ )
    {
        /* Short-term prediction */
        __m128i xmm_hi_07, xmm_hi_8F, xmm_lo_07, xmm_lo_8F;

        /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
        LPC_pred_Q10 = 8; /* silk_RSHIFT( predictLPCOrder, 1 ); */

        /* shift psLPC_Q14 */
        psLPC_Q14_hi_89ABCDEF = _mm_alignr_epi8( psLPC_Q14_hi_01234567, psLPC_Q14_hi_89ABCDEF, 2 );
        psLPC_Q14_lo_89ABCDEF = _mm_alignr_epi8( psLPC_Q14_lo_01234567, psLPC_Q14_lo_89ABCDEF, 2 );

        psLPC_Q14_hi_01234567 = _mm_srli_si128( psLPC_Q14_hi_01234567, 2 );
        psLPC_Q14_lo_01234567 = _mm_srli_si128( psLPC_Q14_lo_01234567, 2 );

        psLPC_Q14_hi_01234567 = _mm_insert_epi16( psLPC_Q14_hi_01234567, (xq_Q14 >> 16), 7 );
        psLPC_Q14_lo_01234567 = _mm_insert_epi16( psLPC_Q14_lo_01234567, (xq_Q14),       7 );

        /* high part, use pmaddwd, results in 4 32-bit */
        xmm_hi_07 = _mm_madd_epi16( psLPC_Q14_hi_01234567, a_Q12_01234567 );
        xmm_hi_8F = _mm_madd_epi16( psLPC_Q14_hi_89ABCDEF, a_Q12_89ABCDEF );

        /* low part, use pmulhw, results in 8 16-bit, note we need simulate unsigned * signed, _mm_srai_epi16(psLPC_Q14_lo_01234567, 15) */
        xmm_tempa = _mm_cmpgt_epi16( _mm_setzero_si128(), psLPC_Q14_lo_01234567 );
        xmm_tempb = _mm_cmpgt_epi16( _mm_setzero_si128(), psLPC_Q14_lo_89ABCDEF );

        xmm_tempa = _mm_and_si128( xmm_tempa, a_Q12_01234567 );
        xmm_tempb = _mm_and_si128( xmm_tempb, a_Q12_89ABCDEF );

        xmm_lo_07 = _mm_mulhi_epi16( psLPC_Q14_lo_01234567, a_Q12_01234567 );
        xmm_lo_8F = _mm_mulhi_epi16( psLPC_Q14_lo_89ABCDEF, a_Q12_89ABCDEF );

        xmm_lo_07 = _mm_add_epi16( xmm_lo_07, xmm_tempa );
        xmm_lo_8F = _mm_add_epi16( xmm_lo_8F, xmm_tempb );

        xmm_lo_07 = _mm_madd_epi16( xmm_lo_07, xmm_one );
        xmm_lo_8F = _mm_madd_epi16( xmm_lo_8F, xmm_one );

        /* accumulate */
        xmm_hi_07 = _mm_add_epi32( xmm_hi_07, xmm_hi_8F );
        xmm_lo_07 = _mm_add_epi32( xmm_lo_07, xmm_lo_8F );

        xmm_hi_07 = _mm_add_epi32( xmm_hi_07, xmm_lo_07 );

        xmm_hi_07 = _mm_add_epi32( xmm_hi_07, _mm_unpackhi_epi64(xmm_hi_07, xmm_hi_07 ) );
        xmm_hi_07 = _mm_add_epi32( xmm_hi_07, _mm_shufflelo_epi16(xmm_hi_07, 0x0E ) );

        LPC_pred_Q10 += _mm_cvtsi128_si32( xmm_hi_07 );

        /* Long-term prediction */
        if ( opus_likely( signalType == TYPE_VOICED ) ) {
            /* Unrolled loop */
            /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
            LTP_pred_Q13 = 2;
            {
                __m128i b_Q14_3210, b_Q14_0123, pred_lag_ptr_0123;

                b_Q14_3210 = OP_CVTEPI16_EPI32_M64( b_Q14 );
                b_Q14_0123 = _mm_shuffle_epi32( b_Q14_3210, 0x1B );

                /* loaded: [0] [-1] [-2] [-3] */
                pred_lag_ptr_0123 = _mm_loadu_si128( (__m128i *)(&pred_lag_ptr[ -3 ] ) );
                /* shuffle to [-3] [-2] [-1] [0] and to new xmm */
                xmm_tempa = _mm_shuffle_epi32( pred_lag_ptr_0123, 0x1B );
                /*64-bit multiply, a[2] * b[-2], a[0] * b[0] */
                xmm_tempa = _mm_mul_epi32( xmm_tempa, b_Q14_3210 );
                /* right shift 2 bytes (16 bits), zero extended */
                xmm_tempa = _mm_srli_si128( xmm_tempa, 2 );

                /* a[1] * b[-1], a[3] * b[-3] */
                pred_lag_ptr_0123 = _mm_mul_epi32( pred_lag_ptr_0123, b_Q14_0123 );
                pred_lag_ptr_0123 = _mm_srli_si128( pred_lag_ptr_0123, 2 );

                pred_lag_ptr_0123 = _mm_add_epi32( pred_lag_ptr_0123, xmm_tempa );
                /* equal shift right 8 bytes*/
                xmm_tempa = _mm_shuffle_epi32( pred_lag_ptr_0123, _MM_SHUFFLE( 0, 0, 3, 2 ) );
                xmm_tempa = _mm_add_epi32( xmm_tempa, pred_lag_ptr_0123 );

                LTP_pred_Q13 += _mm_cvtsi128_si32( xmm_tempa );

                LTP_pred_Q13 = silk_SMLAWB( LTP_pred_Q13, pred_lag_ptr[ -4 ], b_Q14[ 4 ] );
                pred_lag_ptr++;
            }
        }

        /* Noise shape feedback */
        NSQ->sAR2_Q14[ 9 ] = NSQ->sAR2_Q14[ 8 ];
        NSQ->sAR2_Q14[ 8 ] = _mm_cvtsi128_si32( _mm_srli_si128(_mm_unpackhi_epi16( sAR2_Q14_lo_76543210, sAR2_Q14_hi_76543210 ), 12 ) );

        sAR2_Q14_hi_76543210 = _mm_slli_si128( sAR2_Q14_hi_76543210, 2 );
        sAR2_Q14_lo_76543210 = _mm_slli_si128( sAR2_Q14_lo_76543210, 2 );

        sAR2_Q14_hi_76543210 = _mm_insert_epi16( sAR2_Q14_hi_76543210, (xq_Q14 >> 16), 0 );
        sAR2_Q14_lo_76543210 = _mm_insert_epi16( sAR2_Q14_lo_76543210, (xq_Q14),       0 );

        /* high part, use pmaddwd, results in 4 32-bit */
        xmm_hi_07 = _mm_madd_epi16( sAR2_Q14_hi_76543210, AR_shp_Q13_76543210 );

        /* low part, use pmulhw, results in 8 16-bit, note we need simulate unsigned * signed,_mm_srai_epi16(sAR2_Q14_lo_76543210, 15) */
        xmm_tempa = _mm_cmpgt_epi16( _mm_setzero_si128(), sAR2_Q14_lo_76543210 );
        xmm_tempa = _mm_and_si128( xmm_tempa, AR_shp_Q13_76543210 );

        xmm_lo_07 = _mm_mulhi_epi16( sAR2_Q14_lo_76543210, AR_shp_Q13_76543210 );
        xmm_lo_07 = _mm_add_epi16( xmm_lo_07, xmm_tempa );

        xmm_lo_07 = _mm_madd_epi16( xmm_lo_07, xmm_one );

        /* accumulate */
        xmm_hi_07 = _mm_add_epi32( xmm_hi_07, xmm_lo_07 );

        xmm_hi_07 = _mm_add_epi32( xmm_hi_07, _mm_unpackhi_epi64(xmm_hi_07, xmm_hi_07 ) );
        xmm_hi_07 = _mm_add_epi32( xmm_hi_07, _mm_shufflelo_epi16(xmm_hi_07, 0x0E ) );

        n_AR_Q12 = 5 + _mm_cvtsi128_si32( xmm_hi_07 );

        n_AR_Q12 = silk_SMLAWB( n_AR_Q12, NSQ->sAR2_Q14[ 8 ], AR_shp_Q13[ 8 ] );
        n_AR_Q12 = silk_SMLAWB( n_AR_Q12, NSQ->sAR2_Q14[ 9 ], AR_shp_Q13[ 9 ] );

        n_AR_Q12 = silk_LSHIFT32( n_AR_Q12, 1 );                                /* Q11 -> Q12 */
        n_AR_Q12 = silk_SMLAWB( n_AR_Q12, sLF_AR_shp_Q14, Tilt_Q14 );

        n_LF_Q12 = silk_SMULWB( NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - 1 ], LF_shp_Q14 );
        n_LF_Q12 = silk_SMLAWT( n_LF_Q12, sLF_AR_shp_Q14, LF_shp_Q14 );

        silk_assert( lag > 0 || signalType != TYPE_VOICED );

        /* Combine prediction and noise shaping signals */
        tmp1 = silk_SUB32( silk_LSHIFT32( LPC_pred_Q10, 2 ), n_AR_Q12 );        /* Q12 */
        tmp1 = silk_SUB32( tmp1, n_LF_Q12 );                                    /* Q12 */
        if( lag > 0 ) {
            /* Symmetric, packed FIR coefficients */
            n_LTP_Q13 = silk_SMULWB( silk_ADD32( shp_lag_ptr[ 0 ], shp_lag_ptr[ -2 ] ), HarmShapeFIRPacked_Q14 );
            n_LTP_Q13 = silk_SMLAWT( n_LTP_Q13, shp_lag_ptr[ -1 ],                      HarmShapeFIRPacked_Q14 );
            n_LTP_Q13 = silk_LSHIFT( n_LTP_Q13, 1 );
            shp_lag_ptr++;

            tmp2 = silk_SUB32( LTP_pred_Q13, n_LTP_Q13 );                       /* Q13 */
            tmp1 = silk_ADD_LSHIFT32( tmp2, tmp1, 1 );                          /* Q13 */
            tmp1 = silk_RSHIFT_ROUND( tmp1, 3 );                                /* Q10 */
        } else {
            tmp1 = silk_RSHIFT_ROUND( tmp1, 2 );                                /* Q10 */
        }

        r_Q10 = silk_SUB32( x_sc_Q10[ i ], tmp1 );                              /* residual error Q10 */

        /* Generate dither */
        NSQ->rand_seed = silk_RAND( NSQ->rand_seed );

        /* Flip sign depending on dither */
        tmp2 = -r_Q10;
        if ( NSQ->rand_seed < 0 ) r_Q10 = tmp2;

        r_Q10 = silk_LIMIT_32( r_Q10, -(31 << 10), 30 << 10 );

        /* Find two quantization level candidates and measure their rate-distortion */
        q1_Q10 = silk_SUB32( r_Q10, offset_Q10 );
        q1_Q0 = silk_RSHIFT( q1_Q10, 10 );

        q1_Q10 = table[q1_Q0][0];
        q2_Q10 = table[q1_Q0][1];

        if (r_Q10 * table[q1_Q0][2] - table[q1_Q0][3] < 0)
        {
            q1_Q10 = q2_Q10;
        }

        pulses[ i ] = (opus_int8)silk_RSHIFT_ROUND( q1_Q10, 10 );

        /* Excitation */
        exc_Q14 = silk_LSHIFT( q1_Q10, 4 );

        tmp2 = -exc_Q14;
        if ( NSQ->rand_seed < 0 ) exc_Q14 = tmp2;

        /* Add predictions */
        LPC_exc_Q14 = silk_ADD_LSHIFT32( exc_Q14, LTP_pred_Q13, 1 );
        xq_Q14      = silk_ADD_LSHIFT32( LPC_exc_Q14, LPC_pred_Q10, 4 );

        /* Update states */
        psLPC_Q14++;
        *psLPC_Q14 = xq_Q14;
        sLF_AR_shp_Q14 = silk_SUB_LSHIFT32( xq_Q14, n_AR_Q12, 2 );

        NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx ] = silk_SUB_LSHIFT32( sLF_AR_shp_Q14, n_LF_Q12, 2 );
        sLTP_Q15[ NSQ->sLTP_buf_idx ] = silk_LSHIFT( LPC_exc_Q14, 1 );
        NSQ->sLTP_shp_buf_idx++;
        NSQ->sLTP_buf_idx++;

        /* Make dither dependent on quantized signal */
        NSQ->rand_seed = silk_ADD32_ovflw( NSQ->rand_seed, pulses[ i ] );
    }

    NSQ->sLF_AR_shp_Q14 = sLF_AR_shp_Q14;

    /* Scale XQ back to normal level before saving */
    psLPC_Q14 = &NSQ->sLPC_Q14[ NSQ_LPC_BUF_LENGTH ];

    /* write back sAR2_Q14 */
    xmm_tempa = _mm_unpackhi_epi16( sAR2_Q14_lo_76543210, sAR2_Q14_hi_76543210 );
    xmm_tempb = _mm_unpacklo_epi16( sAR2_Q14_lo_76543210, sAR2_Q14_hi_76543210 );
    _mm_storeu_si128( (__m128i *)(&NSQ->sAR2_Q14[ 4 ]), xmm_tempa );
    _mm_storeu_si128( (__m128i *)(&NSQ->sAR2_Q14[ 0 ]), xmm_tempb );

    /* xq[ i ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( silk_SMULWW( psLPC_Q14[ i ], Gain_Q10 ), 8 ) ); */
    {
        __m128i xmm_Gain_Q10;
        __m128i xmm_xq_Q14_3210, xmm_xq_Q14_x3x1, xmm_xq_Q14_7654, xmm_xq_Q14_x7x5;

        /* prepare (1 << 7) in packed 4 32-bits */
        xmm_tempa = _mm_set1_epi32( (1 << 7) );

        /* prepare Gain_Q10 in packed 4 32-bits */
        xmm_Gain_Q10 = _mm_set1_epi32( Gain_Q10 );

        /* process xq */
        for (i = 0; i < length - 7; i += 8)
        {
            xmm_xq_Q14_3210 = _mm_loadu_si128( (__m128i *)(&(psLPC_Q14[ i + 0 ] ) ) );
            xmm_xq_Q14_7654 = _mm_loadu_si128( (__m128i *)(&(psLPC_Q14[ i + 4 ] ) ) );

            /* equal shift right 4 bytes*/
            xmm_xq_Q14_x3x1 = _mm_shuffle_epi32( xmm_xq_Q14_3210, _MM_SHUFFLE( 0, 3, 2, 1 ) );
            /* equal shift right 4 bytes*/
            xmm_xq_Q14_x7x5 = _mm_shuffle_epi32( xmm_xq_Q14_7654, _MM_SHUFFLE( 0, 3, 2, 1 ) );

            xmm_xq_Q14_3210 = _mm_mul_epi32( xmm_xq_Q14_3210, xmm_Gain_Q10 );
            xmm_xq_Q14_x3x1 = _mm_mul_epi32( xmm_xq_Q14_x3x1, xmm_Gain_Q10 );
            xmm_xq_Q14_7654 = _mm_mul_epi32( xmm_xq_Q14_7654, xmm_Gain_Q10 );
            xmm_xq_Q14_x7x5 = _mm_mul_epi32( xmm_xq_Q14_x7x5, xmm_Gain_Q10 );

            xmm_xq_Q14_3210 = _mm_srli_epi64( xmm_xq_Q14_3210, 16 );
            xmm_xq_Q14_x3x1 = _mm_slli_epi64( xmm_xq_Q14_x3x1, 16 );
            xmm_xq_Q14_7654 = _mm_srli_epi64( xmm_xq_Q14_7654, 16 );
            xmm_xq_Q14_x7x5 = _mm_slli_epi64( xmm_xq_Q14_x7x5, 16 );

            xmm_xq_Q14_3210 = _mm_blend_epi16( xmm_xq_Q14_3210, xmm_xq_Q14_x3x1, 0xCC );
            xmm_xq_Q14_7654 = _mm_blend_epi16( xmm_xq_Q14_7654, xmm_xq_Q14_x7x5, 0xCC );

            /* silk_RSHIFT_ROUND(xq, 8) */
            xmm_xq_Q14_3210 = _mm_add_epi32( xmm_xq_Q14_3210, xmm_tempa );
            xmm_xq_Q14_7654 = _mm_add_epi32( xmm_xq_Q14_7654, xmm_tempa );

            xmm_xq_Q14_3210 = _mm_srai_epi32( xmm_xq_Q14_3210, 8 );
            xmm_xq_Q14_7654 = _mm_srai_epi32( xmm_xq_Q14_7654, 8 );

            /* silk_SAT16 */
            xmm_xq_Q14_3210 = _mm_packs_epi32( xmm_xq_Q14_3210, xmm_xq_Q14_7654 );

            /* save to xq */
            _mm_storeu_si128( (__m128i *)(&xq[ i ] ), xmm_xq_Q14_3210 );
        }
    }
    for ( ; i < length; i++)
    {
        xq[i] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( silk_SMULWW( psLPC_Q14[ i ], Gain_Q10 ), 8 ) );
    }

    /* Update LPC synth buffer */
    silk_memcpy( NSQ->sLPC_Q14, &NSQ->sLPC_Q14[ length ], NSQ_LPC_BUF_LENGTH * sizeof( opus_int32 ) );
}

static OPUS_INLINE void silk_nsq_scale_states_sse4_1(
    const silk_encoder_state *psEncC,           /* I    Encoder State                   */
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                       */
    const opus_int32    x_Q3[],                 /* I    input in Q3                     */
    opus_int32          x_sc_Q10[],             /* O    input scaled with 1/Gain        */
    const opus_int16    sLTP[],                 /* I    re-whitened LTP state in Q0     */
    opus_int32          sLTP_Q15[],             /* O    LTP state matching scaled input */
    opus_int            subfr,                  /* I    subframe number                 */
    const opus_int      LTP_scale_Q14,          /* I                                    */
    const opus_int32    Gains_Q16[ MAX_NB_SUBFR ], /* I                                 */
    const opus_int      pitchL[ MAX_NB_SUBFR ], /* I    Pitch lag                       */
    const opus_int      signal_type             /* I    Signal type                     */
)
{
    opus_int   i, lag;
    opus_int32 gain_adj_Q16, inv_gain_Q31, inv_gain_Q23;
    __m128i xmm_inv_gain_Q23, xmm_x_Q3_x2x0, xmm_x_Q3_x3x1;

    lag          = pitchL[ subfr ];
    inv_gain_Q31 = silk_INVERSE32_varQ( silk_max( Gains_Q16[ subfr ], 1 ), 47 );
    silk_assert( inv_gain_Q31 != 0 );

    /* Calculate gain adjustment factor */
    if( Gains_Q16[ subfr ] != NSQ->prev_gain_Q16 ) {
        gain_adj_Q16 =  silk_DIV32_varQ( NSQ->prev_gain_Q16, Gains_Q16[ subfr ], 16 );
    } else {
        gain_adj_Q16 = (opus_int32)1 << 16;
    }

    /* Scale input */
    inv_gain_Q23 = silk_RSHIFT_ROUND( inv_gain_Q31, 8 );

    /* prepare inv_gain_Q23 in packed 4 32-bits */
    xmm_inv_gain_Q23 = _mm_set1_epi32(inv_gain_Q23);

    for( i = 0; i < psEncC->subfr_length - 3; i += 4 ) {
        xmm_x_Q3_x2x0 = _mm_loadu_si128( (__m128i *)(&(x_Q3[ i ] ) ) );

        /* equal shift right 4 bytes*/
        xmm_x_Q3_x3x1 = _mm_shuffle_epi32( xmm_x_Q3_x2x0, _MM_SHUFFLE( 0, 3, 2, 1 ) );

        xmm_x_Q3_x2x0 = _mm_mul_epi32( xmm_x_Q3_x2x0, xmm_inv_gain_Q23 );
        xmm_x_Q3_x3x1 = _mm_mul_epi32( xmm_x_Q3_x3x1, xmm_inv_gain_Q23 );

        xmm_x_Q3_x2x0 = _mm_srli_epi64( xmm_x_Q3_x2x0, 16 );
        xmm_x_Q3_x3x1 = _mm_slli_epi64( xmm_x_Q3_x3x1, 16 );

        xmm_x_Q3_x2x0 = _mm_blend_epi16( xmm_x_Q3_x2x0, xmm_x_Q3_x3x1, 0xCC );

        _mm_storeu_si128( (__m128i *)(&(x_sc_Q10[ i ] ) ), xmm_x_Q3_x2x0 );
    }

    for( ; i < psEncC->subfr_length; i++ ) {
        x_sc_Q10[ i ] = silk_SMULWW( x_Q3[ i ], inv_gain_Q23 );
    }

    /* Save inverse gain */
    NSQ->prev_gain_Q16 = Gains_Q16[ subfr ];

    /* After rewhitening the LTP state is un-scaled, so scale with inv_gain_Q16 */
    if( NSQ->rewhite_flag ) {
        if( subfr == 0 ) {
            /* Do LTP downscaling */
            inv_gain_Q31 = silk_LSHIFT( silk_SMULWB( inv_gain_Q31, LTP_scale_Q14 ), 2 );
        }
        for( i = NSQ->sLTP_buf_idx - lag - LTP_ORDER / 2; i < NSQ->sLTP_buf_idx; i++ ) {
            silk_assert( i < MAX_FRAME_LENGTH );
            sLTP_Q15[ i ] = silk_SMULWB( inv_gain_Q31, sLTP[ i ] );
        }
    }

    /* Adjust for changing gain */
    if( gain_adj_Q16 != (opus_int32)1 << 16 ) {
        /* Scale long-term shaping state */
        __m128i xmm_gain_adj_Q16, xmm_sLTP_shp_Q14_x2x0, xmm_sLTP_shp_Q14_x3x1;

        /* prepare gain_adj_Q16 in packed 4 32-bits */
        xmm_gain_adj_Q16 = _mm_set1_epi32(gain_adj_Q16);

        for( i = NSQ->sLTP_shp_buf_idx - psEncC->ltp_mem_length; i < NSQ->sLTP_shp_buf_idx - 3; i += 4 )
        {
            xmm_sLTP_shp_Q14_x2x0 = _mm_loadu_si128( (__m128i *)(&(NSQ->sLTP_shp_Q14[ i ] ) ) );
            /* equal shift right 4 bytes*/
            xmm_sLTP_shp_Q14_x3x1 = _mm_shuffle_epi32( xmm_sLTP_shp_Q14_x2x0, _MM_SHUFFLE( 0, 3, 2, 1 ) );

            xmm_sLTP_shp_Q14_x2x0 = _mm_mul_epi32( xmm_sLTP_shp_Q14_x2x0, xmm_gain_adj_Q16 );
            xmm_sLTP_shp_Q14_x3x1 = _mm_mul_epi32( xmm_sLTP_shp_Q14_x3x1, xmm_gain_adj_Q16 );

            xmm_sLTP_shp_Q14_x2x0 = _mm_srli_epi64( xmm_sLTP_shp_Q14_x2x0, 16 );
            xmm_sLTP_shp_Q14_x3x1 = _mm_slli_epi64( xmm_sLTP_shp_Q14_x3x1, 16 );

            xmm_sLTP_shp_Q14_x2x0 = _mm_blend_epi16( xmm_sLTP_shp_Q14_x2x0, xmm_sLTP_shp_Q14_x3x1, 0xCC );

            _mm_storeu_si128( (__m128i *)(&(NSQ->sLTP_shp_Q14[ i ] ) ), xmm_sLTP_shp_Q14_x2x0 );
        }

        for( ; i < NSQ->sLTP_shp_buf_idx; i++ ) {
            NSQ->sLTP_shp_Q14[ i ] = silk_SMULWW( gain_adj_Q16, NSQ->sLTP_shp_Q14[ i ] );
        }

        /* Scale long-term prediction state */
        if( signal_type == TYPE_VOICED && NSQ->rewhite_flag == 0 ) {
            for( i = NSQ->sLTP_buf_idx - lag - LTP_ORDER / 2; i < NSQ->sLTP_buf_idx; i++ ) {
                sLTP_Q15[ i ] = silk_SMULWW( gain_adj_Q16, sLTP_Q15[ i ] );
            }
        }

        NSQ->sLF_AR_shp_Q14 = silk_SMULWW( gain_adj_Q16, NSQ->sLF_AR_shp_Q14 );

        /* Scale short-term prediction and shaping states */
        for( i = 0; i < NSQ_LPC_BUF_LENGTH; i++ ) {
            NSQ->sLPC_Q14[ i ] = silk_SMULWW( gain_adj_Q16, NSQ->sLPC_Q14[ i ] );
        }
        for( i = 0; i < MAX_SHAPE_LPC_ORDER; i++ ) {
            NSQ->sAR2_Q14[ i ] = silk_SMULWW( gain_adj_Q16, NSQ->sAR2_Q14[ i ] );
        }
    }
}
