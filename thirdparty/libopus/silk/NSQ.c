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
#include "NSQ.h"


static OPUS_INLINE void silk_nsq_scale_states(
    const silk_encoder_state *psEncC,           /* I    Encoder State                   */
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                       */
    const opus_int16    x16[],                  /* I    input                           */
    opus_int32          x_sc_Q10[],             /* O    input scaled with 1/Gain        */
    const opus_int16    sLTP[],                 /* I    re-whitened LTP state in Q0     */
    opus_int32          sLTP_Q15[],             /* O    LTP state matching scaled input */
    opus_int            subfr,                  /* I    subframe number                 */
    const opus_int      LTP_scale_Q14,          /* I                                    */
    const opus_int32    Gains_Q16[ MAX_NB_SUBFR ], /* I                                 */
    const opus_int      pitchL[ MAX_NB_SUBFR ], /* I    Pitch lag                       */
    const opus_int      signal_type             /* I    Signal type                     */
);

#if !defined(OPUS_X86_MAY_HAVE_SSE4_1)
static OPUS_INLINE void silk_noise_shape_quantizer(
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
    opus_int            Lambda_Q10,             /* I                                    */
    opus_int            offset_Q10,             /* I                                    */
    opus_int            length,                 /* I    Input length                    */
    opus_int            shapingLPCOrder,        /* I    Noise shaping AR filter order   */
    opus_int            predictLPCOrder,        /* I    Prediction filter order         */
    int                 arch                    /* I    Architecture                    */
);
#endif

void silk_NSQ_c
(
    const silk_encoder_state    *psEncC,                                      /* I    Encoder State                   */
    silk_nsq_state              *NSQ,                                         /* I/O  NSQ state                       */
    SideInfoIndices             *psIndices,                                   /* I/O  Quantization Indices            */
    const opus_int16            x16[],                                        /* I    Input                           */
    opus_int8                   pulses[],                                     /* O    Quantized pulse signal          */
    const opus_int16            *PredCoef_Q12,                                /* I    Short term prediction coefs     */
    const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],      /* I    Long term prediction coefs      */
    const opus_int16            AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I    Noise shaping coefs             */
    const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],            /* I    Long term shaping coefs         */
    const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                     /* I    Spectral tilt                   */
    const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                   /* I    Low frequency shaping coefs     */
    const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                    /* I    Quantization step sizes         */
    const opus_int              pitchL[ MAX_NB_SUBFR ],                       /* I    Pitch lags                      */
    const opus_int              Lambda_Q10,                                   /* I    Rate/distortion tradeoff        */
    const opus_int              LTP_scale_Q14                                 /* I    LTP state scaling               */
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
    SAVE_STACK;

    NSQ->rand_seed = psIndices->Seed;

    /* Set unvoiced lag to the previous one, overwrite later for voiced */
    lag = NSQ->lagPrev;

    silk_assert( NSQ->prev_gain_Q16 != 0 );

    offset_Q10 = silk_Quantization_Offsets_Q10[ psIndices->signalType >> 1 ][ psIndices->quantOffsetType ];

    if( psIndices->NLSFInterpCoef_Q2 == 4 ) {
        LSF_interpolation_flag = 0;
    } else {
        LSF_interpolation_flag = 1;
    }

    ALLOC( sLTP_Q15, psEncC->ltp_mem_length + psEncC->frame_length, opus_int32 );
    ALLOC( sLTP, psEncC->ltp_mem_length + psEncC->frame_length, opus_int16 );
    ALLOC( x_sc_Q10, psEncC->subfr_length, opus_int32 );
    /* Set up pointers to start of sub frame */
    NSQ->sLTP_shp_buf_idx = psEncC->ltp_mem_length;
    NSQ->sLTP_buf_idx     = psEncC->ltp_mem_length;
    pxq                   = &NSQ->xq[ psEncC->ltp_mem_length ];
    for( k = 0; k < psEncC->nb_subfr; k++ ) {
        A_Q12      = &PredCoef_Q12[ (( k >> 1 ) | ( 1 - LSF_interpolation_flag )) * MAX_LPC_ORDER ];
        B_Q14      = &LTPCoef_Q14[ k * LTP_ORDER ];
        AR_shp_Q13 = &AR_Q13[ k * MAX_SHAPE_LPC_ORDER ];

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
                celt_assert( start_idx > 0 );

                silk_LPC_analysis_filter( &sLTP[ start_idx ], &NSQ->xq[ start_idx + k * psEncC->subfr_length ],
                    A_Q12, psEncC->ltp_mem_length - start_idx, psEncC->predictLPCOrder, psEncC->arch );

                NSQ->rewhite_flag = 1;
                NSQ->sLTP_buf_idx = psEncC->ltp_mem_length;
            }
        }

        silk_nsq_scale_states( psEncC, NSQ, x16, x_sc_Q10, sLTP, sLTP_Q15, k, LTP_scale_Q14, Gains_Q16, pitchL, psIndices->signalType );

        silk_noise_shape_quantizer( NSQ, psIndices->signalType, x_sc_Q10, pulses, pxq, sLTP_Q15, A_Q12, B_Q14,
            AR_shp_Q13, lag, HarmShapeFIRPacked_Q14, Tilt_Q14[ k ], LF_shp_Q14[ k ], Gains_Q16[ k ], Lambda_Q10,
            offset_Q10, psEncC->subfr_length, psEncC->shapingLPCOrder, psEncC->predictLPCOrder, psEncC->arch );

        x16    += psEncC->subfr_length;
        pulses += psEncC->subfr_length;
        pxq    += psEncC->subfr_length;
    }

    /* Update lagPrev for next frame */
    NSQ->lagPrev = pitchL[ psEncC->nb_subfr - 1 ];

    /* Save quantized speech and noise shaping signals */
    silk_memmove( NSQ->xq,           &NSQ->xq[           psEncC->frame_length ], psEncC->ltp_mem_length * sizeof( opus_int16 ) );
    silk_memmove( NSQ->sLTP_shp_Q14, &NSQ->sLTP_shp_Q14[ psEncC->frame_length ], psEncC->ltp_mem_length * sizeof( opus_int32 ) );
    RESTORE_STACK;
}

/******************************/
/* silk_noise_shape_quantizer */
/******************************/

#if !defined(OPUS_X86_MAY_HAVE_SSE4_1)
static OPUS_INLINE
#endif
void silk_noise_shape_quantizer(
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
    opus_int            Lambda_Q10,             /* I                                    */
    opus_int            offset_Q10,             /* I                                    */
    opus_int            length,                 /* I    Input length                    */
    opus_int            shapingLPCOrder,        /* I    Noise shaping AR filter order   */
    opus_int            predictLPCOrder,        /* I    Prediction filter order         */
    int                 arch                    /* I    Architecture                    */
)
{
    opus_int     i;
    opus_int32   LTP_pred_Q13, LPC_pred_Q10, n_AR_Q12, n_LTP_Q13;
    opus_int32   n_LF_Q12, r_Q10, rr_Q10, q1_Q0, q1_Q10, q2_Q10, rd1_Q20, rd2_Q20;
    opus_int32   exc_Q14, LPC_exc_Q14, xq_Q14, Gain_Q10;
    opus_int32   tmp1, tmp2, sLF_AR_shp_Q14;
    opus_int32   *psLPC_Q14, *shp_lag_ptr, *pred_lag_ptr;
#ifdef silk_short_prediction_create_arch_coef
    opus_int32   a_Q12_arch[MAX_LPC_ORDER];
#endif

    shp_lag_ptr  = &NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2 ];
    pred_lag_ptr = &sLTP_Q15[ NSQ->sLTP_buf_idx - lag + LTP_ORDER / 2 ];
    Gain_Q10     = silk_RSHIFT( Gain_Q16, 6 );

    /* Set up short term AR state */
    psLPC_Q14 = &NSQ->sLPC_Q14[ NSQ_LPC_BUF_LENGTH - 1 ];

#ifdef silk_short_prediction_create_arch_coef
    silk_short_prediction_create_arch_coef(a_Q12_arch, a_Q12, predictLPCOrder);
#endif

    for( i = 0; i < length; i++ ) {
        /* Generate dither */
        NSQ->rand_seed = silk_RAND( NSQ->rand_seed );

        /* Short-term prediction */
        LPC_pred_Q10 = silk_noise_shape_quantizer_short_prediction(psLPC_Q14, a_Q12, a_Q12_arch, predictLPCOrder, arch);

        /* Long-term prediction */
        if( signalType == TYPE_VOICED ) {
            /* Unrolled loop */
            /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
            LTP_pred_Q13 = 2;
            LTP_pred_Q13 = silk_SMLAWB( LTP_pred_Q13, pred_lag_ptr[  0 ], b_Q14[ 0 ] );
            LTP_pred_Q13 = silk_SMLAWB( LTP_pred_Q13, pred_lag_ptr[ -1 ], b_Q14[ 1 ] );
            LTP_pred_Q13 = silk_SMLAWB( LTP_pred_Q13, pred_lag_ptr[ -2 ], b_Q14[ 2 ] );
            LTP_pred_Q13 = silk_SMLAWB( LTP_pred_Q13, pred_lag_ptr[ -3 ], b_Q14[ 3 ] );
            LTP_pred_Q13 = silk_SMLAWB( LTP_pred_Q13, pred_lag_ptr[ -4 ], b_Q14[ 4 ] );
            pred_lag_ptr++;
        } else {
            LTP_pred_Q13 = 0;
        }

        /* Noise shape feedback */
        celt_assert( ( shapingLPCOrder & 1 ) == 0 );   /* check that order is even */
        n_AR_Q12 = silk_NSQ_noise_shape_feedback_loop(&NSQ->sDiff_shp_Q14, NSQ->sAR2_Q14, AR_shp_Q13, shapingLPCOrder, arch);

        n_AR_Q12 = silk_SMLAWB( n_AR_Q12, NSQ->sLF_AR_shp_Q14, Tilt_Q14 );

        n_LF_Q12 = silk_SMULWB( NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - 1 ], LF_shp_Q14 );
        n_LF_Q12 = silk_SMLAWT( n_LF_Q12, NSQ->sLF_AR_shp_Q14, LF_shp_Q14 );

        celt_assert( lag > 0 || signalType != TYPE_VOICED );

        /* Combine prediction and noise shaping signals */
        tmp1 = silk_SUB32_ovflw( silk_LSHIFT32( LPC_pred_Q10, 2 ), n_AR_Q12 );  /* Q12 */
        tmp1 = silk_SUB32_ovflw( tmp1, n_LF_Q12 );                              /* Q12 */
        if( lag > 0 ) {
            /* Symmetric, packed FIR coefficients */
            n_LTP_Q13 = silk_SMULWB( silk_ADD_SAT32( shp_lag_ptr[ 0 ], shp_lag_ptr[ -2 ] ), HarmShapeFIRPacked_Q14 );
            n_LTP_Q13 = silk_SMLAWT( n_LTP_Q13, shp_lag_ptr[ -1 ],                      HarmShapeFIRPacked_Q14 );
            n_LTP_Q13 = silk_LSHIFT( n_LTP_Q13, 1 );
            shp_lag_ptr++;

            tmp2 = silk_SUB32( LTP_pred_Q13, n_LTP_Q13 );                       /* Q13 */
            tmp1 = silk_ADD32_ovflw( tmp2, silk_LSHIFT32( tmp1, 1 ) );          /* Q13 */
            tmp1 = silk_RSHIFT_ROUND( tmp1, 3 );                                /* Q10 */
        } else {
            tmp1 = silk_RSHIFT_ROUND( tmp1, 2 );                                /* Q10 */
        }

        r_Q10 = silk_SUB32( x_sc_Q10[ i ], tmp1 );                              /* residual error Q10 */

        /* Flip sign depending on dither */
        if( NSQ->rand_seed < 0 ) {
            r_Q10 = -r_Q10;
        }
        r_Q10 = silk_LIMIT_32( r_Q10, -(31 << 10), 30 << 10 );

        /* Find two quantization level candidates and measure their rate-distortion */
        q1_Q10 = silk_SUB32( r_Q10, offset_Q10 );
        q1_Q0 = silk_RSHIFT( q1_Q10, 10 );
        if (Lambda_Q10 > 2048) {
            /* For aggressive RDO, the bias becomes more than one pulse. */
            int rdo_offset = Lambda_Q10/2 - 512;
            if (q1_Q10 > rdo_offset) {
                q1_Q0 = silk_RSHIFT( q1_Q10 - rdo_offset, 10 );
            } else if (q1_Q10 < -rdo_offset) {
                q1_Q0 = silk_RSHIFT( q1_Q10 + rdo_offset, 10 );
            } else if (q1_Q10 < 0) {
                q1_Q0 = -1;
            } else {
                q1_Q0 = 0;
            }
        }
        if( q1_Q0 > 0 ) {
            q1_Q10  = silk_SUB32( silk_LSHIFT( q1_Q0, 10 ), QUANT_LEVEL_ADJUST_Q10 );
            q1_Q10  = silk_ADD32( q1_Q10, offset_Q10 );
            q2_Q10  = silk_ADD32( q1_Q10, 1024 );
            rd1_Q20 = silk_SMULBB( q1_Q10, Lambda_Q10 );
            rd2_Q20 = silk_SMULBB( q2_Q10, Lambda_Q10 );
        } else if( q1_Q0 == 0 ) {
            q1_Q10  = offset_Q10;
            q2_Q10  = silk_ADD32( q1_Q10, 1024 - QUANT_LEVEL_ADJUST_Q10 );
            rd1_Q20 = silk_SMULBB( q1_Q10, Lambda_Q10 );
            rd2_Q20 = silk_SMULBB( q2_Q10, Lambda_Q10 );
        } else if( q1_Q0 == -1 ) {
            q2_Q10  = offset_Q10;
            q1_Q10  = silk_SUB32( q2_Q10, 1024 - QUANT_LEVEL_ADJUST_Q10 );
            rd1_Q20 = silk_SMULBB( -q1_Q10, Lambda_Q10 );
            rd2_Q20 = silk_SMULBB(  q2_Q10, Lambda_Q10 );
        } else {            /* Q1_Q0 < -1 */
            q1_Q10  = silk_ADD32( silk_LSHIFT( q1_Q0, 10 ), QUANT_LEVEL_ADJUST_Q10 );
            q1_Q10  = silk_ADD32( q1_Q10, offset_Q10 );
            q2_Q10  = silk_ADD32( q1_Q10, 1024 );
            rd1_Q20 = silk_SMULBB( -q1_Q10, Lambda_Q10 );
            rd2_Q20 = silk_SMULBB( -q2_Q10, Lambda_Q10 );
        }
        rr_Q10  = silk_SUB32( r_Q10, q1_Q10 );
        rd1_Q20 = silk_SMLABB( rd1_Q20, rr_Q10, rr_Q10 );
        rr_Q10  = silk_SUB32( r_Q10, q2_Q10 );
        rd2_Q20 = silk_SMLABB( rd2_Q20, rr_Q10, rr_Q10 );

        if( rd2_Q20 < rd1_Q20 ) {
            q1_Q10 = q2_Q10;
        }

        pulses[ i ] = (opus_int8)silk_RSHIFT_ROUND( q1_Q10, 10 );

        /* Excitation */
        exc_Q14 = silk_LSHIFT( q1_Q10, 4 );
        if ( NSQ->rand_seed < 0 ) {
           exc_Q14 = -exc_Q14;
        }

        /* Add predictions */
        LPC_exc_Q14 = silk_ADD_LSHIFT32( exc_Q14, LTP_pred_Q13, 1 );
        xq_Q14      = silk_ADD32_ovflw( LPC_exc_Q14, silk_LSHIFT32( LPC_pred_Q10, 4 ) );

        /* Scale XQ back to normal level before saving */
        xq[ i ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( silk_SMULWW( xq_Q14, Gain_Q10 ), 8 ) );

        /* Update states */
        psLPC_Q14++;
        *psLPC_Q14 = xq_Q14;
        NSQ->sDiff_shp_Q14 = silk_SUB32_ovflw( xq_Q14, silk_LSHIFT32( x_sc_Q10[ i ], 4 ) );
        sLF_AR_shp_Q14 = silk_SUB32_ovflw( NSQ->sDiff_shp_Q14, silk_LSHIFT32( n_AR_Q12, 2 ) );
        NSQ->sLF_AR_shp_Q14 = sLF_AR_shp_Q14;

        NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx ] = silk_SUB32_ovflw(sLF_AR_shp_Q14, silk_LSHIFT32(n_LF_Q12, 2));
        sLTP_Q15[ NSQ->sLTP_buf_idx ] = silk_LSHIFT( LPC_exc_Q14, 1 );
        NSQ->sLTP_shp_buf_idx++;
        NSQ->sLTP_buf_idx++;

        /* Make dither dependent on quantized signal */
        NSQ->rand_seed = silk_ADD32_ovflw( NSQ->rand_seed, pulses[ i ] );
    }

    /* Update LPC synth buffer */
    silk_memcpy( NSQ->sLPC_Q14, &NSQ->sLPC_Q14[ length ], NSQ_LPC_BUF_LENGTH * sizeof( opus_int32 ) );
}

static OPUS_INLINE void silk_nsq_scale_states(
    const silk_encoder_state *psEncC,           /* I    Encoder State                   */
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                       */
    const opus_int16    x16[],                  /* I    input                           */
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
    opus_int32 gain_adj_Q16, inv_gain_Q31, inv_gain_Q26;

    lag          = pitchL[ subfr ];
    inv_gain_Q31 = silk_INVERSE32_varQ( silk_max( Gains_Q16[ subfr ], 1 ), 47 );
    silk_assert( inv_gain_Q31 != 0 );

    /* Scale input */
    inv_gain_Q26 = silk_RSHIFT_ROUND( inv_gain_Q31, 5 );
    for( i = 0; i < psEncC->subfr_length; i++ ) {
        x_sc_Q10[ i ] = silk_SMULWW( x16[ i ], inv_gain_Q26 );
    }

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
    if( Gains_Q16[ subfr ] != NSQ->prev_gain_Q16 ) {
        gain_adj_Q16 =  silk_DIV32_varQ( NSQ->prev_gain_Q16, Gains_Q16[ subfr ], 16 );

        /* Scale long-term shaping state */
        for( i = NSQ->sLTP_shp_buf_idx - psEncC->ltp_mem_length; i < NSQ->sLTP_shp_buf_idx; i++ ) {
            NSQ->sLTP_shp_Q14[ i ] = silk_SMULWW( gain_adj_Q16, NSQ->sLTP_shp_Q14[ i ] );
        }

        /* Scale long-term prediction state */
        if( signal_type == TYPE_VOICED && NSQ->rewhite_flag == 0 ) {
            for( i = NSQ->sLTP_buf_idx - lag - LTP_ORDER / 2; i < NSQ->sLTP_buf_idx; i++ ) {
                sLTP_Q15[ i ] = silk_SMULWW( gain_adj_Q16, sLTP_Q15[ i ] );
            }
        }

        NSQ->sLF_AR_shp_Q14 = silk_SMULWW( gain_adj_Q16, NSQ->sLF_AR_shp_Q14 );
        NSQ->sDiff_shp_Q14 = silk_SMULWW( gain_adj_Q16, NSQ->sDiff_shp_Q14 );

        /* Scale short-term prediction and shaping states */
        for( i = 0; i < NSQ_LPC_BUF_LENGTH; i++ ) {
            NSQ->sLPC_Q14[ i ] = silk_SMULWW( gain_adj_Q16, NSQ->sLPC_Q14[ i ] );
        }
        for( i = 0; i < MAX_SHAPE_LPC_ORDER; i++ ) {
            NSQ->sAR2_Q14[ i ] = silk_SMULWW( gain_adj_Q16, NSQ->sAR2_Q14[ i ] );
        }

        /* Save inverse gain */
        NSQ->prev_gain_Q16 = Gains_Q16[ subfr ];
    }
}
