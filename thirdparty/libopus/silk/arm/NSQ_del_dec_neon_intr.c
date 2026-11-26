/***********************************************************************
Copyright (c) 2017 Google Inc.
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

#include <arm_neon.h>
#ifdef OPUS_CHECK_ASM
# include <string.h>
#endif
#include "main.h"
#include "stack_alloc.h"
#include "os_support.h"

/* NEON intrinsics optimization now can only parallelize up to 4 delay decision states.    */
/* If there are more states, C function is called, and this optimization must be expanded. */
#define NEON_MAX_DEL_DEC_STATES 4

typedef struct {
    opus_int32 sLPC_Q14[ MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH ][ NEON_MAX_DEL_DEC_STATES ];
    opus_int32 RandState[ DECISION_DELAY ][     NEON_MAX_DEL_DEC_STATES ];
    opus_int32 Q_Q10[     DECISION_DELAY ][     NEON_MAX_DEL_DEC_STATES ];
    opus_int32 Xq_Q14[    DECISION_DELAY ][     NEON_MAX_DEL_DEC_STATES ];
    opus_int32 Pred_Q15[  DECISION_DELAY ][     NEON_MAX_DEL_DEC_STATES ];
    opus_int32 Shape_Q14[ DECISION_DELAY ][     NEON_MAX_DEL_DEC_STATES ];
    opus_int32 sAR2_Q14[ MAX_SHAPE_LPC_ORDER ][ NEON_MAX_DEL_DEC_STATES ];
    opus_int32 LF_AR_Q14[ NEON_MAX_DEL_DEC_STATES ];
    opus_int32 Diff_Q14[  NEON_MAX_DEL_DEC_STATES ];
    opus_int32 Seed[      NEON_MAX_DEL_DEC_STATES ];
    opus_int32 SeedInit[  NEON_MAX_DEL_DEC_STATES ];
    opus_int32 RD_Q10[    NEON_MAX_DEL_DEC_STATES ];
} NSQ_del_decs_struct;

typedef struct {
    opus_int32 Q_Q10[        NEON_MAX_DEL_DEC_STATES ];
    opus_int32 RD_Q10[       NEON_MAX_DEL_DEC_STATES ];
    opus_int32 xq_Q14[       NEON_MAX_DEL_DEC_STATES ];
    opus_int32 LF_AR_Q14[    NEON_MAX_DEL_DEC_STATES ];
    opus_int32 Diff_Q14[     NEON_MAX_DEL_DEC_STATES ];
    opus_int32 sLTP_shp_Q14[ NEON_MAX_DEL_DEC_STATES ];
    opus_int32 LPC_exc_Q14[  NEON_MAX_DEL_DEC_STATES ];
} NSQ_samples_struct;

static OPUS_INLINE void silk_nsq_del_dec_scale_states_neon(
    const silk_encoder_state *psEncC,               /* I    Encoder State                       */
    silk_nsq_state      *NSQ,                       /* I/O  NSQ state                           */
    NSQ_del_decs_struct psDelDec[],                 /* I/O  Delayed decision states             */
    const opus_int16    x16[],                      /* I    Input                               */
    opus_int32          x_sc_Q10[],                 /* O    Input scaled with 1/Gain in Q10     */
    const opus_int16    sLTP[],                     /* I    Re-whitened LTP state in Q0         */
    opus_int32          sLTP_Q15[],                 /* O    LTP state matching scaled input     */
    opus_int            subfr,                      /* I    Subframe number                     */
    const opus_int      LTP_scale_Q14,              /* I    LTP state scaling                   */
    const opus_int32    Gains_Q16[ MAX_NB_SUBFR ],  /* I                                        */
    const opus_int      pitchL[ MAX_NB_SUBFR ],     /* I    Pitch lag                           */
    const opus_int      signal_type,                /* I    Signal type                         */
    const opus_int      decisionDelay               /* I    Decision delay                      */
);

/******************************************/
/* Noise shape quantizer for one subframe */
/******************************************/
static OPUS_INLINE void silk_noise_shape_quantizer_del_dec_neon(
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                           */
    NSQ_del_decs_struct psDelDec[],             /* I/O  Delayed decision states             */
    opus_int            signalType,             /* I    Signal type                         */
    const opus_int32    x_Q10[],                /* I                                        */
    opus_int8           pulses[],               /* O                                        */
    opus_int16          xq[],                   /* O                                        */
    opus_int32          sLTP_Q15[],             /* I/O  LTP filter state                    */
    opus_int32          delayedGain_Q10[],      /* I/O  Gain delay buffer                   */
    const opus_int16    a_Q12[],                /* I    Short term prediction coefs         */
    const opus_int16    b_Q14[],                /* I    Long term prediction coefs          */
    const opus_int16    AR_shp_Q13[],           /* I    Noise shaping coefs                 */
    opus_int            lag,                    /* I    Pitch lag                           */
    opus_int32          HarmShapeFIRPacked_Q14, /* I                                        */
    opus_int            Tilt_Q14,               /* I    Spectral tilt                       */
    opus_int32          LF_shp_Q14,             /* I                                        */
    opus_int32          Gain_Q16,               /* I                                        */
    opus_int            Lambda_Q10,             /* I                                        */
    opus_int            offset_Q10,             /* I                                        */
    opus_int            length,                 /* I    Input length                        */
    opus_int            subfr,                  /* I    Subframe number                     */
    opus_int            shapingLPCOrder,        /* I    Shaping LPC filter order            */
    opus_int            predictLPCOrder,        /* I    Prediction filter order             */
    opus_int            warping_Q16,            /* I                                        */
    opus_int            nStatesDelayedDecision, /* I    Number of states in decision tree   */
    opus_int            *smpl_buf_idx,          /* I/O  Index to newest samples in buffers  */
    opus_int            decisionDelay           /* I                                        */
);

static OPUS_INLINE void copy_winner_state_kernel(
    const NSQ_del_decs_struct *psDelDec,
    const opus_int            offset,
    const opus_int            last_smple_idx,
    const opus_int            Winner_ind,
    const int32x2_t           gain_lo_s32x2,
    const int32x2_t           gain_hi_s32x2,
    const int32x4_t           shift_s32x4,
    int32x4_t                 t0_s32x4,
    int32x4_t                 t1_s32x4,
    opus_int8 *const          pulses,
    opus_int16                *pxq,
    silk_nsq_state            *NSQ
)
{
    int16x8_t t_s16x8;
    int32x4_t o0_s32x4, o1_s32x4;

    t0_s32x4 = vld1q_lane_s32( &psDelDec->Q_Q10[ last_smple_idx - 0 ][ Winner_ind ], t0_s32x4, 0 );
    t0_s32x4 = vld1q_lane_s32( &psDelDec->Q_Q10[ last_smple_idx - 1 ][ Winner_ind ], t0_s32x4, 1 );
    t0_s32x4 = vld1q_lane_s32( &psDelDec->Q_Q10[ last_smple_idx - 2 ][ Winner_ind ], t0_s32x4, 2 );
    t0_s32x4 = vld1q_lane_s32( &psDelDec->Q_Q10[ last_smple_idx - 3 ][ Winner_ind ], t0_s32x4, 3 );
    t1_s32x4 = vld1q_lane_s32( &psDelDec->Q_Q10[ last_smple_idx - 4 ][ Winner_ind ], t1_s32x4, 0 );
    t1_s32x4 = vld1q_lane_s32( &psDelDec->Q_Q10[ last_smple_idx - 5 ][ Winner_ind ], t1_s32x4, 1 );
    t1_s32x4 = vld1q_lane_s32( &psDelDec->Q_Q10[ last_smple_idx - 6 ][ Winner_ind ], t1_s32x4, 2 );
    t1_s32x4 = vld1q_lane_s32( &psDelDec->Q_Q10[ last_smple_idx - 7 ][ Winner_ind ], t1_s32x4, 3 );
    t_s16x8 = vcombine_s16( vrshrn_n_s32( t0_s32x4, 10 ), vrshrn_n_s32( t1_s32x4, 10 ) );
    vst1_s8( &pulses[ offset ], vmovn_s16( t_s16x8 ) );

    t0_s32x4 = vld1q_lane_s32( &psDelDec->Xq_Q14[ last_smple_idx - 0 ][ Winner_ind ], t0_s32x4, 0 );
    t0_s32x4 = vld1q_lane_s32( &psDelDec->Xq_Q14[ last_smple_idx - 1 ][ Winner_ind ], t0_s32x4, 1 );
    t0_s32x4 = vld1q_lane_s32( &psDelDec->Xq_Q14[ last_smple_idx - 2 ][ Winner_ind ], t0_s32x4, 2 );
    t0_s32x4 = vld1q_lane_s32( &psDelDec->Xq_Q14[ last_smple_idx - 3 ][ Winner_ind ], t0_s32x4, 3 );
    t1_s32x4 = vld1q_lane_s32( &psDelDec->Xq_Q14[ last_smple_idx - 4 ][ Winner_ind ], t1_s32x4, 0 );
    t1_s32x4 = vld1q_lane_s32( &psDelDec->Xq_Q14[ last_smple_idx - 5 ][ Winner_ind ], t1_s32x4, 1 );
    t1_s32x4 = vld1q_lane_s32( &psDelDec->Xq_Q14[ last_smple_idx - 6 ][ Winner_ind ], t1_s32x4, 2 );
    t1_s32x4 = vld1q_lane_s32( &psDelDec->Xq_Q14[ last_smple_idx - 7 ][ Winner_ind ], t1_s32x4, 3 );
    o0_s32x4 = vqdmulhq_lane_s32( t0_s32x4, gain_lo_s32x2, 0 );
    o1_s32x4 = vqdmulhq_lane_s32( t1_s32x4, gain_lo_s32x2, 0 );
    o0_s32x4 = vmlaq_lane_s32( o0_s32x4, t0_s32x4, gain_hi_s32x2, 0 );
    o1_s32x4 = vmlaq_lane_s32( o1_s32x4, t1_s32x4, gain_hi_s32x2, 0 );
    o0_s32x4 = vrshlq_s32( o0_s32x4, shift_s32x4 );
    o1_s32x4 = vrshlq_s32( o1_s32x4, shift_s32x4 );
    vst1_s16( &pxq[ offset + 0 ], vqmovn_s32( o0_s32x4 ) );
    vst1_s16( &pxq[ offset + 4 ], vqmovn_s32( o1_s32x4 ) );

    t0_s32x4 = vld1q_lane_s32( &psDelDec->Shape_Q14[ last_smple_idx - 0 ][ Winner_ind ], t0_s32x4, 0 );
    t0_s32x4 = vld1q_lane_s32( &psDelDec->Shape_Q14[ last_smple_idx - 1 ][ Winner_ind ], t0_s32x4, 1 );
    t0_s32x4 = vld1q_lane_s32( &psDelDec->Shape_Q14[ last_smple_idx - 2 ][ Winner_ind ], t0_s32x4, 2 );
    t0_s32x4 = vld1q_lane_s32( &psDelDec->Shape_Q14[ last_smple_idx - 3 ][ Winner_ind ], t0_s32x4, 3 );
    t1_s32x4 = vld1q_lane_s32( &psDelDec->Shape_Q14[ last_smple_idx - 4 ][ Winner_ind ], t1_s32x4, 0 );
    t1_s32x4 = vld1q_lane_s32( &psDelDec->Shape_Q14[ last_smple_idx - 5 ][ Winner_ind ], t1_s32x4, 1 );
    t1_s32x4 = vld1q_lane_s32( &psDelDec->Shape_Q14[ last_smple_idx - 6 ][ Winner_ind ], t1_s32x4, 2 );
    t1_s32x4 = vld1q_lane_s32( &psDelDec->Shape_Q14[ last_smple_idx - 7 ][ Winner_ind ], t1_s32x4, 3 );
    vst1q_s32( &NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx + offset + 0 ], t0_s32x4 );
    vst1q_s32( &NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx + offset + 4 ], t1_s32x4 );
}

static OPUS_INLINE void copy_winner_state(
    const NSQ_del_decs_struct *psDelDec,
    const opus_int            decisionDelay,
    const opus_int            smpl_buf_idx,
    const opus_int            Winner_ind,
    const opus_int32          gain,
    const opus_int32          shift,
    opus_int8 *const          pulses,
    opus_int16                *pxq,
    silk_nsq_state            *NSQ
)
{
    opus_int        i, last_smple_idx;
    const int32x2_t gain_lo_s32x2 = vdup_n_s32( silk_LSHIFT32( gain & 0x0000FFFF, 15 ) );
    const int32x2_t gain_hi_s32x2 = vdup_n_s32( gain >> 16 );
    const int32x4_t shift_s32x4 = vdupq_n_s32( -shift );
    int32x4_t       t0_s32x4, t1_s32x4;

    t0_s32x4 = t1_s32x4 = vdupq_n_s32( 0 ); /* initialization */
    last_smple_idx = smpl_buf_idx + decisionDelay - 1 + DECISION_DELAY;
    if( last_smple_idx >= DECISION_DELAY ) last_smple_idx -= DECISION_DELAY;
    if( last_smple_idx >= DECISION_DELAY ) last_smple_idx -= DECISION_DELAY;

    for( i = 0; ( i < ( decisionDelay - 7 ) ) && ( last_smple_idx >= 7 ); i += 8, last_smple_idx -= 8 ) {
        copy_winner_state_kernel( psDelDec, i - decisionDelay, last_smple_idx, Winner_ind, gain_lo_s32x2, gain_hi_s32x2, shift_s32x4, t0_s32x4, t1_s32x4, pulses, pxq, NSQ );
    }
    for( ; ( i < decisionDelay ) && ( last_smple_idx >= 0 ); i++, last_smple_idx-- ) {
        pulses[ i - decisionDelay ] = (opus_int8)silk_RSHIFT_ROUND( psDelDec->Q_Q10[ last_smple_idx ][ Winner_ind ], 10 );
        pxq[ i - decisionDelay ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( silk_SMULWW( psDelDec->Xq_Q14[ last_smple_idx ][ Winner_ind ], gain ), shift ) );
        NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - decisionDelay + i ] = psDelDec->Shape_Q14[ last_smple_idx ][ Winner_ind ];
    }

    last_smple_idx += DECISION_DELAY;
    for( ; i < ( decisionDelay - 7 ); i++, last_smple_idx-- ) {
        copy_winner_state_kernel( psDelDec, i - decisionDelay, last_smple_idx, Winner_ind, gain_lo_s32x2, gain_hi_s32x2, shift_s32x4, t0_s32x4, t1_s32x4, pulses, pxq, NSQ );
    }
    for( ; i < decisionDelay; i++, last_smple_idx-- ) {
        pulses[ i - decisionDelay ] = (opus_int8)silk_RSHIFT_ROUND( psDelDec->Q_Q10[ last_smple_idx ][ Winner_ind ], 10 );
        pxq[ i - decisionDelay ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( silk_SMULWW( psDelDec->Xq_Q14[ last_smple_idx ][ Winner_ind ], gain ), shift ) );
        NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - decisionDelay + i ] = psDelDec->Shape_Q14[ last_smple_idx ][ Winner_ind ];
    }
}

void silk_NSQ_del_dec_neon(
    const silk_encoder_state    *psEncC,                                    /* I    Encoder State                   */
    silk_nsq_state              *NSQ,                                       /* I/O  NSQ state                       */
    SideInfoIndices             *psIndices,                                 /* I/O  Quantization Indices            */
    const opus_int16            x16[],                                      /* I    Input                           */
    opus_int8                   pulses[],                                   /* O    Quantized pulse signal          */
    const opus_int16            *PredCoef_Q12,                              /* I    Short term prediction coefs     */
    const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],    /* I    Long term prediction coefs      */
    const opus_int16            AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I Noise shaping coefs              */
    const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],          /* I    Long term shaping coefs         */
    const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                   /* I    Spectral tilt                   */
    const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                 /* I    Low frequency shaping coefs     */
    const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                  /* I    Quantization step sizes         */
    const opus_int              pitchL[ MAX_NB_SUBFR ],                     /* I    Pitch lags                      */
    const opus_int              Lambda_Q10,                                 /* I    Rate/distortion tradeoff        */
    const opus_int              LTP_scale_Q14                               /* I    LTP state scaling               */
)
{
#ifdef OPUS_CHECK_ASM
    silk_nsq_state NSQ_c;
    SideInfoIndices psIndices_c;
    opus_int8 pulses_c[ MAX_FRAME_LENGTH ];
    const opus_int8 *const pulses_a = pulses;

    ( void )pulses_a;
    silk_memcpy( &NSQ_c, NSQ, sizeof( NSQ_c ) );
    silk_memcpy( &psIndices_c, psIndices, sizeof( psIndices_c ) );
    silk_memcpy( pulses_c, pulses, sizeof( pulses_c ) );
    silk_NSQ_del_dec_c( psEncC, &NSQ_c, &psIndices_c, x16, pulses_c, PredCoef_Q12, LTPCoef_Q14, AR_Q13, HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16,
                       pitchL, Lambda_Q10, LTP_scale_Q14 );
#endif

    /* The optimization parallelizes the different delay decision states. */
    if(( psEncC->nStatesDelayedDecision > NEON_MAX_DEL_DEC_STATES ) || ( psEncC->nStatesDelayedDecision <= 2 )) {
        /* NEON intrinsics optimization now can only parallelize up to 4 delay decision states.    */
        /* If there are more states, C function is called, and this optimization must be expanded. */
        /* When the number of delay decision states is less than 3, there are penalties using this */
        /* optimization, and C function is called.                                                 */
        /* When the number of delay decision states is 2, it's better to specialize another        */
        /* structure NSQ_del_dec2_struct and optimize with shorter NEON registers. (Low priority)  */
        silk_NSQ_del_dec_c( psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, HarmShapeGain_Q14,
            Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14 );
    } else {
        opus_int            i, k, lag, start_idx, LSF_interpolation_flag, Winner_ind, subfr;
        opus_int            smpl_buf_idx, decisionDelay;
        const opus_int16    *A_Q12, *B_Q14, *AR_shp_Q13;
        opus_int16          *pxq;
        VARDECL( opus_int32, sLTP_Q15 );
        VARDECL( opus_int16, sLTP );
        opus_int32          HarmShapeFIRPacked_Q14;
        opus_int            offset_Q10;
        opus_int32          RDmin_Q10, Gain_Q10;
        VARDECL( opus_int32, x_sc_Q10 );
        VARDECL( opus_int32, delayedGain_Q10 );
        VARDECL( NSQ_del_decs_struct, psDelDec );
        int32x4_t           t_s32x4;
        SAVE_STACK;

        /* Set unvoiced lag to the previous one, overwrite later for voiced */
        lag = NSQ->lagPrev;

        silk_assert( NSQ->prev_gain_Q16 != 0 );

        /* Initialize delayed decision states */
        ALLOC( psDelDec, 1, NSQ_del_decs_struct );
        OPUS_CLEAR(psDelDec, 1);
        /* Only RandState and RD_Q10 need to be initialized to 0. */
        silk_memset( psDelDec->RandState, 0, sizeof( psDelDec->RandState ) );
        vst1q_s32( psDelDec->RD_Q10, vdupq_n_s32( 0 ) );

        for( k = 0; k < psEncC->nStatesDelayedDecision; k++ ) {
            psDelDec->SeedInit[ k ] = psDelDec->Seed[ k ] = ( k + psIndices->Seed ) & 3;
        }
        vst1q_s32( psDelDec->LF_AR_Q14, vld1q_dup_s32( &NSQ->sLF_AR_shp_Q14 ) );
        vst1q_s32( psDelDec->Diff_Q14, vld1q_dup_s32( &NSQ->sDiff_shp_Q14 ) );
        vst1q_s32( psDelDec->Shape_Q14[ 0 ], vld1q_dup_s32( &NSQ->sLTP_shp_Q14[ psEncC->ltp_mem_length - 1 ] ) );
        for( i = 0; i < NSQ_LPC_BUF_LENGTH; i++ ) {
            vst1q_s32( psDelDec->sLPC_Q14[ i ], vld1q_dup_s32( &NSQ->sLPC_Q14[ i ] ) );
        }
        for( i = 0; i < (opus_int)( sizeof( NSQ->sAR2_Q14 ) / sizeof( NSQ->sAR2_Q14[ 0 ] ) ); i++ ) {
            vst1q_s32( psDelDec->sAR2_Q14[ i ], vld1q_dup_s32( &NSQ->sAR2_Q14[ i ] ) );
        }

        offset_Q10   = silk_Quantization_Offsets_Q10[ psIndices->signalType >> 1 ][ psIndices->quantOffsetType ];
        smpl_buf_idx = 0; /* index of oldest samples */

        decisionDelay = silk_min_int( DECISION_DELAY, psEncC->subfr_length );

        /* For voiced frames limit the decision delay to lower than the pitch lag */
        if( psIndices->signalType == TYPE_VOICED ) {
            opus_int pitch_min = pitchL[ 0 ];
            for( k = 1; k < psEncC->nb_subfr; k++ ) {
                pitch_min = silk_min_int( pitch_min, pitchL[ k ] );
            }
            decisionDelay = silk_min_int( decisionDelay, pitch_min - LTP_ORDER / 2 - 1 );
        } else {
            if( lag > 0 ) {
                decisionDelay = silk_min_int( decisionDelay, lag - LTP_ORDER / 2 - 1 );
            }
        }

        if( psIndices->NLSFInterpCoef_Q2 == 4 ) {
            LSF_interpolation_flag = 0;
        } else {
            LSF_interpolation_flag = 1;
        }

        ALLOC( sLTP_Q15, psEncC->ltp_mem_length + psEncC->frame_length, opus_int32 );
        ALLOC( sLTP, psEncC->ltp_mem_length + psEncC->frame_length, opus_int16 );
        ALLOC( x_sc_Q10, psEncC->subfr_length, opus_int32 );
        ALLOC( delayedGain_Q10, DECISION_DELAY, opus_int32 );
        /* Set up pointers to start of sub frame */
        pxq                   = &NSQ->xq[ psEncC->ltp_mem_length ];
        NSQ->sLTP_shp_buf_idx = psEncC->ltp_mem_length;
        NSQ->sLTP_buf_idx     = psEncC->ltp_mem_length;
        subfr = 0;
        for( k = 0; k < psEncC->nb_subfr; k++ ) {
            A_Q12      = &PredCoef_Q12[ ( ( k >> 1 ) | ( 1 - LSF_interpolation_flag ) ) * MAX_LPC_ORDER ];
            B_Q14      = &LTPCoef_Q14[ k * LTP_ORDER           ];
            AR_shp_Q13 = &AR_Q13[     k * MAX_SHAPE_LPC_ORDER ];

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
                    if( k == 2 ) {
                        /* RESET DELAYED DECISIONS */
                        /* Find winner */
                        int32x4_t RD_Q10_s32x4;
                        RDmin_Q10 = psDelDec->RD_Q10[ 0 ];
                        Winner_ind = 0;
                        for( i = 1; i < psEncC->nStatesDelayedDecision; i++ ) {
                            if( psDelDec->RD_Q10[ i ] < RDmin_Q10 ) {
                                RDmin_Q10 = psDelDec->RD_Q10[ i ];
                                Winner_ind = i;
                            }
                        }
                        psDelDec->RD_Q10[ Winner_ind ] -= ( silk_int32_MAX >> 4 );
                        RD_Q10_s32x4 = vld1q_s32( psDelDec->RD_Q10 );
                        RD_Q10_s32x4 = vaddq_s32( RD_Q10_s32x4, vdupq_n_s32( silk_int32_MAX >> 4 ) );
                        vst1q_s32( psDelDec->RD_Q10, RD_Q10_s32x4 );

                        /* Copy final part of signals from winner state to output and long-term filter states */
                        copy_winner_state( psDelDec, decisionDelay, smpl_buf_idx, Winner_ind, Gains_Q16[ 1 ], 14, pulses, pxq, NSQ );

                        subfr = 0;
                    }

                    /* Rewhiten with new A coefs */
                    start_idx = psEncC->ltp_mem_length - lag - psEncC->predictLPCOrder - LTP_ORDER / 2;
                    silk_assert( start_idx > 0 );

                    silk_LPC_analysis_filter( &sLTP[ start_idx ], &NSQ->xq[ start_idx + k * psEncC->subfr_length ],
                        A_Q12, psEncC->ltp_mem_length - start_idx, psEncC->predictLPCOrder, psEncC->arch );

                    NSQ->sLTP_buf_idx = psEncC->ltp_mem_length;
                    NSQ->rewhite_flag = 1;
                }
            }

            silk_nsq_del_dec_scale_states_neon( psEncC, NSQ, psDelDec, x16, x_sc_Q10, sLTP, sLTP_Q15, k,
                LTP_scale_Q14, Gains_Q16, pitchL, psIndices->signalType, decisionDelay );

            silk_noise_shape_quantizer_del_dec_neon( NSQ, psDelDec, psIndices->signalType, x_sc_Q10, pulses, pxq, sLTP_Q15,
                delayedGain_Q10, A_Q12, B_Q14, AR_shp_Q13, lag, HarmShapeFIRPacked_Q14, Tilt_Q14[ k ], LF_shp_Q14[ k ],
                Gains_Q16[ k ], Lambda_Q10, offset_Q10, psEncC->subfr_length, subfr++, psEncC->shapingLPCOrder,
                psEncC->predictLPCOrder, psEncC->warping_Q16, psEncC->nStatesDelayedDecision, &smpl_buf_idx, decisionDelay );

            x16    += psEncC->subfr_length;
            pulses += psEncC->subfr_length;
            pxq    += psEncC->subfr_length;
        }

        /* Find winner */
        RDmin_Q10 = psDelDec->RD_Q10[ 0 ];
        Winner_ind = 0;
        for( k = 1; k < psEncC->nStatesDelayedDecision; k++ ) {
            if( psDelDec->RD_Q10[ k ] < RDmin_Q10 ) {
                RDmin_Q10 = psDelDec->RD_Q10[ k ];
                Winner_ind = k;
            }
        }

        /* Copy final part of signals from winner state to output and long-term filter states */
        psIndices->Seed = psDelDec->SeedInit[ Winner_ind ];
        Gain_Q10 = silk_RSHIFT32( Gains_Q16[ psEncC->nb_subfr - 1 ], 6 );
        copy_winner_state( psDelDec, decisionDelay, smpl_buf_idx, Winner_ind, Gain_Q10, 8, pulses, pxq, NSQ );

        t_s32x4 = vdupq_n_s32( 0 ); /* initialization */
        for( i = 0; i < ( NSQ_LPC_BUF_LENGTH - 3 ); i += 4 ) {
            t_s32x4 = vld1q_lane_s32( &psDelDec->sLPC_Q14[ i + 0 ][ Winner_ind ], t_s32x4, 0 );
            t_s32x4 = vld1q_lane_s32( &psDelDec->sLPC_Q14[ i + 1 ][ Winner_ind ], t_s32x4, 1 );
            t_s32x4 = vld1q_lane_s32( &psDelDec->sLPC_Q14[ i + 2 ][ Winner_ind ], t_s32x4, 2 );
            t_s32x4 = vld1q_lane_s32( &psDelDec->sLPC_Q14[ i + 3 ][ Winner_ind ], t_s32x4, 3 );
            vst1q_s32( &NSQ->sLPC_Q14[ i ], t_s32x4 );
        }

        for( ; i < NSQ_LPC_BUF_LENGTH; i++ ) {
          NSQ->sLPC_Q14[ i ] = psDelDec->sLPC_Q14[ i ][ Winner_ind ];
        }

        for( i = 0; i < (opus_int)( sizeof( NSQ->sAR2_Q14 ) / sizeof( NSQ->sAR2_Q14[ 0 ] ) - 3 ); i += 4 ) {
            t_s32x4 = vld1q_lane_s32( &psDelDec->sAR2_Q14[ i + 0 ][ Winner_ind ], t_s32x4, 0 );
            t_s32x4 = vld1q_lane_s32( &psDelDec->sAR2_Q14[ i + 1 ][ Winner_ind ], t_s32x4, 1 );
            t_s32x4 = vld1q_lane_s32( &psDelDec->sAR2_Q14[ i + 2 ][ Winner_ind ], t_s32x4, 2 );
            t_s32x4 = vld1q_lane_s32( &psDelDec->sAR2_Q14[ i + 3 ][ Winner_ind ], t_s32x4, 3 );
            vst1q_s32( &NSQ->sAR2_Q14[ i ], t_s32x4 );
        }

        for( ; i < (opus_int)( sizeof( NSQ->sAR2_Q14 ) / sizeof( NSQ->sAR2_Q14[ 0 ] ) ); i++ ) {
          NSQ->sAR2_Q14[ i ] = psDelDec->sAR2_Q14[ i ][ Winner_ind ];
        }

        /* Update states */
        NSQ->sLF_AR_shp_Q14 = psDelDec->LF_AR_Q14[ Winner_ind ];
        NSQ->sDiff_shp_Q14  = psDelDec->Diff_Q14[ Winner_ind ];
        NSQ->lagPrev        = pitchL[ psEncC->nb_subfr - 1 ];

        /* Save quantized speech signal */
        silk_memmove( NSQ->xq,           &NSQ->xq[           psEncC->frame_length ], psEncC->ltp_mem_length * sizeof( opus_int16 ) );
        silk_memmove( NSQ->sLTP_shp_Q14, &NSQ->sLTP_shp_Q14[ psEncC->frame_length ], psEncC->ltp_mem_length * sizeof( opus_int32 ) );
        RESTORE_STACK;
    }

#ifdef OPUS_CHECK_ASM
    silk_assert( !memcmp( &NSQ_c, NSQ, sizeof( NSQ_c ) ) );
    silk_assert( !memcmp( &psIndices_c, psIndices, sizeof( psIndices_c ) ) );
    silk_assert( !memcmp( pulses_c, pulses_a, sizeof( pulses_c ) ) );
#endif
}

/******************************************/
/* Noise shape quantizer for one subframe */
/******************************************/
/* Note: Function silk_short_prediction_create_arch_coef_neon() defined in NSQ_neon.h is actually a hacking C function. */
/*       Therefore here we append "_local" to the NEON function name to avoid confusion.                                */
static OPUS_INLINE void silk_short_prediction_create_arch_coef_neon_local(opus_int32 *out, const opus_int16 *in, opus_int order)
{
  int16x8_t t_s16x8;
  int32x4_t t0_s32x4, t1_s32x4, t2_s32x4, t3_s32x4;
  silk_assert( order == 10 || order == 16 );

  t_s16x8 = vld1q_s16( in + 0 );                                                   /* 7 6 5 4  3 2 1 0    */
  t_s16x8 = vrev64q_s16( t_s16x8 );                                                /* 4 5 6 7  0 1 2 3    */
  t2_s32x4 = vshll_n_s16( vget_high_s16( t_s16x8 ), 15 );                          /* 4 5 6 7             */
  t3_s32x4 = vshll_n_s16( vget_low_s16(  t_s16x8 ), 15 );                          /* 0 1 2 3             */

  if( order == 16 ) {
      t_s16x8 = vld1q_s16( in + 8 );                                               /* F E D C  B A 9 8    */
      t_s16x8 = vrev64q_s16( t_s16x8 );                                            /* C D E F  8 9 A B    */
      t0_s32x4 = vshll_n_s16( vget_high_s16( t_s16x8 ), 15 );                      /* C D E F             */
      t1_s32x4 = vshll_n_s16( vget_low_s16(  t_s16x8 ), 15 );                      /* 8 9 A B             */
  } else {
      int16x4_t t_s16x4;

      t0_s32x4 = vdupq_n_s32( 0 );                                                 /* zero zero zero zero */
      t_s16x4 = vld1_s16( in + 6 );                                                /* 9    8    7    6    */
      t_s16x4 = vrev64_s16( t_s16x4 );                                             /* 6    7    8    9    */
      t1_s32x4 = vshll_n_s16( t_s16x4, 15 );
      t1_s32x4 = vcombine_s32( vget_low_s32(t0_s32x4), vget_low_s32( t1_s32x4 ) ); /* 8    9    zero zero */
  }
  vst1q_s32( out +  0, t0_s32x4 );
  vst1q_s32( out +  4, t1_s32x4 );
  vst1q_s32( out +  8, t2_s32x4 );
  vst1q_s32( out + 12, t3_s32x4 );
}

static OPUS_INLINE int32x4_t silk_SMLAWB_lane0_neon(
    const int32x4_t out_s32x4,
    const int32x4_t in_s32x4,
    const int32x2_t coef_s32x2
)
{
    return vaddq_s32( out_s32x4, vqdmulhq_lane_s32( in_s32x4, coef_s32x2, 0 ) );
}

static OPUS_INLINE int32x4_t silk_SMLAWB_lane1_neon(
    const int32x4_t out_s32x4,
    const int32x4_t in_s32x4,
    const int32x2_t coef_s32x2
)
{
    return vaddq_s32( out_s32x4, vqdmulhq_lane_s32( in_s32x4, coef_s32x2, 1 ) );
}

/* Note: This function has different return value than silk_noise_shape_quantizer_short_prediction_neon(). */
/*       Therefore here we append "_local" to the function name to avoid confusion.                        */
static OPUS_INLINE int32x4_t silk_noise_shape_quantizer_short_prediction_neon_local(const opus_int32 *buf32, const opus_int32 *a_Q12_arch, opus_int order)
{
    const int32x4_t a_Q12_arch0_s32x4 = vld1q_s32( a_Q12_arch + 0 );
    const int32x4_t a_Q12_arch1_s32x4 = vld1q_s32( a_Q12_arch + 4 );
    const int32x4_t a_Q12_arch2_s32x4 = vld1q_s32( a_Q12_arch + 8 );
    const int32x4_t a_Q12_arch3_s32x4 = vld1q_s32( a_Q12_arch + 12 );
    int32x4_t LPC_pred_Q14_s32x4;

    silk_assert( order == 10 || order == 16 );
    /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
    LPC_pred_Q14_s32x4 = vdupq_n_s32( silk_RSHIFT( order, 1 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane0_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 +  0 * NEON_MAX_DEL_DEC_STATES ), vget_low_s32(  a_Q12_arch0_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane1_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 +  1 * NEON_MAX_DEL_DEC_STATES ), vget_low_s32(  a_Q12_arch0_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane0_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 +  2 * NEON_MAX_DEL_DEC_STATES ), vget_high_s32( a_Q12_arch0_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane1_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 +  3 * NEON_MAX_DEL_DEC_STATES ), vget_high_s32( a_Q12_arch0_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane0_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 +  4 * NEON_MAX_DEL_DEC_STATES ), vget_low_s32(  a_Q12_arch1_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane1_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 +  5 * NEON_MAX_DEL_DEC_STATES ), vget_low_s32(  a_Q12_arch1_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane0_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 +  6 * NEON_MAX_DEL_DEC_STATES ), vget_high_s32( a_Q12_arch1_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane1_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 +  7 * NEON_MAX_DEL_DEC_STATES ), vget_high_s32( a_Q12_arch1_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane0_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 +  8 * NEON_MAX_DEL_DEC_STATES ), vget_low_s32(  a_Q12_arch2_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane1_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 +  9 * NEON_MAX_DEL_DEC_STATES ), vget_low_s32(  a_Q12_arch2_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane0_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 + 10 * NEON_MAX_DEL_DEC_STATES ), vget_high_s32( a_Q12_arch2_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane1_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 + 11 * NEON_MAX_DEL_DEC_STATES ), vget_high_s32( a_Q12_arch2_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane0_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 + 12 * NEON_MAX_DEL_DEC_STATES ), vget_low_s32(  a_Q12_arch3_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane1_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 + 13 * NEON_MAX_DEL_DEC_STATES ), vget_low_s32(  a_Q12_arch3_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane0_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 + 14 * NEON_MAX_DEL_DEC_STATES ), vget_high_s32( a_Q12_arch3_s32x4 ) );
    LPC_pred_Q14_s32x4 = silk_SMLAWB_lane1_neon( LPC_pred_Q14_s32x4, vld1q_s32( buf32 + 15 * NEON_MAX_DEL_DEC_STATES ), vget_high_s32( a_Q12_arch3_s32x4 ) );

    return LPC_pred_Q14_s32x4;
}

static OPUS_INLINE void silk_noise_shape_quantizer_del_dec_neon(
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                           */
    NSQ_del_decs_struct psDelDec[],             /* I/O  Delayed decision states             */
    opus_int            signalType,             /* I    Signal type                         */
    const opus_int32    x_Q10[],                /* I                                        */
    opus_int8           pulses[],               /* O                                        */
    opus_int16          xq[],                   /* O                                        */
    opus_int32          sLTP_Q15[],             /* I/O  LTP filter state                    */
    opus_int32          delayedGain_Q10[],      /* I/O  Gain delay buffer                   */
    const opus_int16    a_Q12[],                /* I    Short term prediction coefs         */
    const opus_int16    b_Q14[],                /* I    Long term prediction coefs          */
    const opus_int16    AR_shp_Q13[],           /* I    Noise shaping coefs                 */
    opus_int            lag,                    /* I    Pitch lag                           */
    opus_int32          HarmShapeFIRPacked_Q14, /* I                                        */
    opus_int            Tilt_Q14,               /* I    Spectral tilt                       */
    opus_int32          LF_shp_Q14,             /* I                                        */
    opus_int32          Gain_Q16,               /* I                                        */
    opus_int            Lambda_Q10,             /* I                                        */
    opus_int            offset_Q10,             /* I                                        */
    opus_int            length,                 /* I    Input length                        */
    opus_int            subfr,                  /* I    Subframe number                     */
    opus_int            shapingLPCOrder,        /* I    Shaping LPC filter order            */
    opus_int            predictLPCOrder,        /* I    Prediction filter order             */
    opus_int            warping_Q16,            /* I                                        */
    opus_int            nStatesDelayedDecision, /* I    Number of states in decision tree   */
    opus_int            *smpl_buf_idx,          /* I/O  Index to newest samples in buffers  */
    opus_int            decisionDelay           /* I                                        */
)
{
    opus_int     i, j, k, Winner_ind, RDmin_ind, RDmax_ind, last_smple_idx;
    opus_int32   Winner_rand_state;
    opus_int32   LTP_pred_Q14, n_LTP_Q14;
    opus_int32   RDmin_Q10, RDmax_Q10;
    opus_int32   Gain_Q10;
    opus_int32   *pred_lag_ptr, *shp_lag_ptr;
    opus_int32   a_Q12_arch[MAX_LPC_ORDER];
    const int32x2_t warping_Q16_s32x2 = vdup_n_s32( silk_LSHIFT32( warping_Q16, 16 ) >> 1 );
    const opus_int32 LF_shp_Q29 = silk_LSHIFT32( LF_shp_Q14, 16 ) >> 1;
    opus_int32 AR_shp_Q28[ MAX_SHAPE_LPC_ORDER ];
    const uint32x4_t rand_multiplier_u32x4 = vdupq_n_u32( RAND_MULTIPLIER );
    const uint32x4_t rand_increment_u32x4 = vdupq_n_u32( RAND_INCREMENT );

    VARDECL( NSQ_samples_struct, psSampleState );
    SAVE_STACK;

    silk_assert( nStatesDelayedDecision > 0 );
    silk_assert( ( shapingLPCOrder & 1 ) == 0 );   /* check that order is even */
    ALLOC( psSampleState, 2, NSQ_samples_struct );
    OPUS_CLEAR(psSampleState, 2);

    shp_lag_ptr  = &NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2 ];
    pred_lag_ptr = &sLTP_Q15[ NSQ->sLTP_buf_idx - lag + LTP_ORDER / 2 ];
    Gain_Q10     = silk_RSHIFT( Gain_Q16, 6 );

    for( i = 0; i < ( MAX_SHAPE_LPC_ORDER - 7 ); i += 8 ) {
      const int16x8_t t_s16x8 = vld1q_s16( AR_shp_Q13 +  i );
      vst1q_s32( AR_shp_Q28 + i + 0, vshll_n_s16( vget_low_s16(  t_s16x8 ), 15 ) );
      vst1q_s32( AR_shp_Q28 + i + 4, vshll_n_s16( vget_high_s16( t_s16x8 ), 15 ) );
    }

    for( ; i < MAX_SHAPE_LPC_ORDER; i++ ) {
      AR_shp_Q28[i] = silk_LSHIFT32( AR_shp_Q13[i], 15 );
    }

    silk_short_prediction_create_arch_coef_neon_local( a_Q12_arch, a_Q12, predictLPCOrder );

    for( i = 0; i < length; i++ ) {
        int32x4_t Seed_s32x4, LPC_pred_Q14_s32x4;
        int32x4_t sign_s32x4, tmp1_s32x4, tmp2_s32x4;
        int32x4_t n_AR_Q14_s32x4, n_LF_Q14_s32x4;
        int32x2_t AR_shp_Q28_s32x2;
        int16x4_t r_Q10_s16x4, rr_Q10_s16x4;

        /* Perform common calculations used in all states */

        /* Long-term prediction */
        if( signalType == TYPE_VOICED ) {
            /* Unrolled loop */
            /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
            LTP_pred_Q14 = 2;
            LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[  0 ], b_Q14[ 0 ] );
            LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[ -1 ], b_Q14[ 1 ] );
            LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[ -2 ], b_Q14[ 2 ] );
            LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[ -3 ], b_Q14[ 3 ] );
            LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[ -4 ], b_Q14[ 4 ] );
            LTP_pred_Q14 = silk_LSHIFT( LTP_pred_Q14, 1 );                          /* Q13 -> Q14 */
            pred_lag_ptr++;
        } else {
            LTP_pred_Q14 = 0;
        }

        /* Long-term shaping */
        if( lag > 0 ) {
            /* Symmetric, packed FIR coefficients */
            n_LTP_Q14 = silk_SMULWB( silk_ADD32( shp_lag_ptr[ 0 ], shp_lag_ptr[ -2 ] ), HarmShapeFIRPacked_Q14 );
            n_LTP_Q14 = silk_SMLAWT( n_LTP_Q14, shp_lag_ptr[ -1 ],                      HarmShapeFIRPacked_Q14 );
            n_LTP_Q14 = silk_SUB_LSHIFT32( LTP_pred_Q14, n_LTP_Q14, 2 );            /* Q12 -> Q14 */
            shp_lag_ptr++;
        } else {
            n_LTP_Q14 = 0;
        }

        /* Generate dither */
        Seed_s32x4 = vld1q_s32( psDelDec->Seed );
        Seed_s32x4 = vreinterpretq_s32_u32( vmlaq_u32( rand_increment_u32x4, vreinterpretq_u32_s32( Seed_s32x4 ), rand_multiplier_u32x4 ) );
        vst1q_s32( psDelDec->Seed, Seed_s32x4 );

        /* Short-term prediction */
        LPC_pred_Q14_s32x4 = silk_noise_shape_quantizer_short_prediction_neon_local(psDelDec->sLPC_Q14[ NSQ_LPC_BUF_LENGTH - 16 + i ], a_Q12_arch, predictLPCOrder);
        LPC_pred_Q14_s32x4 = vshlq_n_s32( LPC_pred_Q14_s32x4, 4 ); /* Q10 -> Q14 */

        /* Noise shape feedback */
        /* Output of lowpass section */
        tmp2_s32x4 = silk_SMLAWB_lane0_neon( vld1q_s32( psDelDec->Diff_Q14 ), vld1q_s32( psDelDec->sAR2_Q14[ 0 ] ), warping_Q16_s32x2 );
        /* Output of allpass section */
        tmp1_s32x4 = vsubq_s32( vld1q_s32( psDelDec->sAR2_Q14[ 1 ] ), tmp2_s32x4 );
        tmp1_s32x4 = silk_SMLAWB_lane0_neon( vld1q_s32( psDelDec->sAR2_Q14[ 0 ] ), tmp1_s32x4, warping_Q16_s32x2 );
        vst1q_s32( psDelDec->sAR2_Q14[ 0 ], tmp2_s32x4 );
        AR_shp_Q28_s32x2 = vld1_s32( AR_shp_Q28 );
        n_AR_Q14_s32x4 = vaddq_s32( vdupq_n_s32( silk_RSHIFT( shapingLPCOrder, 1 ) ), vqdmulhq_lane_s32( tmp2_s32x4, AR_shp_Q28_s32x2, 0 ) );

        /* Loop over allpass sections */
        for( j = 2; j < shapingLPCOrder; j += 2 ) {
            /* Output of allpass section */
            tmp2_s32x4 = vsubq_s32( vld1q_s32( psDelDec->sAR2_Q14[ j + 0 ] ), tmp1_s32x4 );
            tmp2_s32x4 = silk_SMLAWB_lane0_neon( vld1q_s32( psDelDec->sAR2_Q14[ j - 1 ] ), tmp2_s32x4, warping_Q16_s32x2 );
            vst1q_s32( psDelDec->sAR2_Q14[ j - 1 ], tmp1_s32x4 );
            n_AR_Q14_s32x4 = vaddq_s32( n_AR_Q14_s32x4, vqdmulhq_lane_s32( tmp1_s32x4, AR_shp_Q28_s32x2, 1 ) );
            /* Output of allpass section */
            tmp1_s32x4 = vsubq_s32( vld1q_s32( psDelDec->sAR2_Q14[ j + 1 ] ), tmp2_s32x4 );
            tmp1_s32x4 = silk_SMLAWB_lane0_neon( vld1q_s32( psDelDec->sAR2_Q14[ j + 0 ] ), tmp1_s32x4, warping_Q16_s32x2 );
            vst1q_s32( psDelDec->sAR2_Q14[ j + 0 ], tmp2_s32x4 );
            AR_shp_Q28_s32x2 = vld1_s32( &AR_shp_Q28[ j ] );
            n_AR_Q14_s32x4 = vaddq_s32( n_AR_Q14_s32x4, vqdmulhq_lane_s32( tmp2_s32x4, AR_shp_Q28_s32x2, 0 ) );
        }
        vst1q_s32( psDelDec->sAR2_Q14[ shapingLPCOrder - 1 ], tmp1_s32x4 );
        n_AR_Q14_s32x4 = vaddq_s32( n_AR_Q14_s32x4, vqdmulhq_lane_s32( tmp1_s32x4, AR_shp_Q28_s32x2, 1 ) );
        n_AR_Q14_s32x4 = vshlq_n_s32( n_AR_Q14_s32x4, 1 );                                                                                        /* Q11 -> Q12 */
        n_AR_Q14_s32x4 = vaddq_s32( n_AR_Q14_s32x4, vqdmulhq_n_s32( vld1q_s32( psDelDec->LF_AR_Q14 ), silk_LSHIFT32( Tilt_Q14, 16 ) >> 1 ) );     /* Q12 */
        n_AR_Q14_s32x4 = vshlq_n_s32( n_AR_Q14_s32x4, 2 );                                                                                        /* Q12 -> Q14 */
        n_LF_Q14_s32x4 = vqdmulhq_n_s32( vld1q_s32( psDelDec->Shape_Q14[ *smpl_buf_idx ] ), LF_shp_Q29 );                                         /* Q12 */
        n_LF_Q14_s32x4 = vaddq_s32( n_LF_Q14_s32x4, vqdmulhq_n_s32( vld1q_s32( psDelDec->LF_AR_Q14 ), silk_LSHIFT32( LF_shp_Q14 >> 16 , 15 ) ) ); /* Q12 */
        n_LF_Q14_s32x4 = vshlq_n_s32( n_LF_Q14_s32x4, 2 );                                                                                        /* Q12 -> Q14 */

        /* Input minus prediction plus noise feedback                       */
        /* r = x[ i ] - LTP_pred - LPC_pred + n_AR + n_Tilt + n_LF + n_LTP  */
        tmp1_s32x4 = vaddq_s32( n_AR_Q14_s32x4, n_LF_Q14_s32x4 );               /* Q14 */
        tmp2_s32x4 = vaddq_s32( vdupq_n_s32( n_LTP_Q14 ), LPC_pred_Q14_s32x4 ); /* Q13 */
        tmp1_s32x4 = vsubq_s32( tmp2_s32x4, tmp1_s32x4 );                       /* Q13 */
        tmp1_s32x4 = vrshrq_n_s32( tmp1_s32x4, 4 );                             /* Q10 */
        tmp1_s32x4 = vsubq_s32( vdupq_n_s32( x_Q10[ i ] ), tmp1_s32x4 );        /* residual error Q10 */

        /* Flip sign depending on dither */
        sign_s32x4 = vreinterpretq_s32_u32( vcltq_s32( Seed_s32x4, vdupq_n_s32( 0 ) ) );
        tmp1_s32x4 = veorq_s32( tmp1_s32x4, sign_s32x4 );
        tmp1_s32x4 = vsubq_s32( tmp1_s32x4, sign_s32x4 );
        tmp1_s32x4 = vmaxq_s32( tmp1_s32x4, vdupq_n_s32( -( 31 << 10 ) ) );
        tmp1_s32x4 = vminq_s32( tmp1_s32x4, vdupq_n_s32( 30 << 10 ) );
        r_Q10_s16x4 = vmovn_s32( tmp1_s32x4 );

        /* Find two quantization level candidates and measure their rate-distortion */
        {
            int16x4_t q1_Q10_s16x4 = vsub_s16( r_Q10_s16x4, vdup_n_s16( offset_Q10 ) );
            int16x4_t q1_Q0_s16x4 = vshr_n_s16( q1_Q10_s16x4, 10 );
            int16x4_t q2_Q10_s16x4;
            int32x4_t rd1_Q10_s32x4, rd2_Q10_s32x4;
            uint32x4_t t_u32x4;

            if( Lambda_Q10 > 2048 ) {
                /* For aggressive RDO, the bias becomes more than one pulse. */
                const int rdo_offset = Lambda_Q10/2 - 512;
                const uint16x4_t greaterThanRdo = vcgt_s16( q1_Q10_s16x4, vdup_n_s16( rdo_offset ) );
                const uint16x4_t lessThanMinusRdo = vclt_s16( q1_Q10_s16x4, vdup_n_s16( -rdo_offset ) );
                int16x4_t signed_offset = vbsl_s16( greaterThanRdo, vdup_n_s16( -rdo_offset ), vdup_n_s16( 0 ) );
                signed_offset = vbsl_s16( lessThanMinusRdo, vdup_n_s16( rdo_offset ), signed_offset );
                /* If Lambda_Q10 > 32767, then q1_Q0, q1_Q10 and q2_Q10 must change to 32-bit. */
                silk_assert( Lambda_Q10 <= 32767 );

                q1_Q0_s16x4 = vreinterpret_s16_u16( vclt_s16( q1_Q10_s16x4, vdup_n_s16( 0 ) ) );
                q1_Q0_s16x4 = vbsl_s16(vorr_u16(greaterThanRdo, lessThanMinusRdo), vadd_s16( q1_Q10_s16x4 , signed_offset), q1_Q0_s16x4);
                q1_Q0_s16x4 = vshr_n_s16( q1_Q0_s16x4, 10 );
            }
            {
                const uint16x4_t equal0_u16x4 = vceq_s16( q1_Q0_s16x4, vdup_n_s16( 0 ) );
                const uint16x4_t equalMinus1_u16x4 = vceq_s16( q1_Q0_s16x4, vdup_n_s16( -1 ) );
                const uint16x4_t lessThanMinus1_u16x4 = vclt_s16( q1_Q0_s16x4, vdup_n_s16( -1 ) );
                int16x4_t tmp1_s16x4, tmp2_s16x4, tmp_summand_s16x4;

                q1_Q10_s16x4 = vshl_n_s16( q1_Q0_s16x4, 10 );
                tmp_summand_s16x4 = vand_s16( vreinterpret_s16_u16(vcge_s16(q1_Q0_s16x4, vdup_n_s16(0))), vdup_n_s16( offset_Q10 - QUANT_LEVEL_ADJUST_Q10 ) );
                tmp1_s16x4 = vadd_s16( q1_Q10_s16x4, tmp_summand_s16x4 );
                tmp_summand_s16x4 = vbsl_s16( lessThanMinus1_u16x4, vdup_n_s16( offset_Q10 + QUANT_LEVEL_ADJUST_Q10 ), vdup_n_s16(0) );
                q1_Q10_s16x4 = vadd_s16( q1_Q10_s16x4,  tmp_summand_s16x4);
                q1_Q10_s16x4 = vbsl_s16( lessThanMinus1_u16x4, q1_Q10_s16x4, tmp1_s16x4 );
                q1_Q10_s16x4 = vbsl_s16( equal0_u16x4, vdup_n_s16( offset_Q10 ), q1_Q10_s16x4 );
                q1_Q10_s16x4 = vbsl_s16( equalMinus1_u16x4, vdup_n_s16( offset_Q10 - ( 1024 - QUANT_LEVEL_ADJUST_Q10 ) ), q1_Q10_s16x4 );
                q2_Q10_s16x4 = vadd_s16( q1_Q10_s16x4, vdup_n_s16( 1024 ) );
                q2_Q10_s16x4 = vbsl_s16( equal0_u16x4, vdup_n_s16( offset_Q10 + 1024 - QUANT_LEVEL_ADJUST_Q10 ), q2_Q10_s16x4 );
                q2_Q10_s16x4 = vbsl_s16( equalMinus1_u16x4, vdup_n_s16( offset_Q10 ), q2_Q10_s16x4 );
                tmp1_s16x4 = q1_Q10_s16x4;
                tmp2_s16x4 = q2_Q10_s16x4;
                tmp1_s16x4 = vbsl_s16( vorr_u16( equalMinus1_u16x4, lessThanMinus1_u16x4 ), vneg_s16( tmp1_s16x4 ), tmp1_s16x4 );
                tmp2_s16x4 = vbsl_s16( lessThanMinus1_u16x4, vneg_s16( tmp2_s16x4 ), tmp2_s16x4 );
                rd1_Q10_s32x4 = vmull_s16( tmp1_s16x4, vdup_n_s16( Lambda_Q10 ) );
                rd2_Q10_s32x4 = vmull_s16( tmp2_s16x4, vdup_n_s16( Lambda_Q10 ) );
            }

            rr_Q10_s16x4 = vsub_s16( r_Q10_s16x4, q1_Q10_s16x4 );
            rd1_Q10_s32x4 = vmlal_s16( rd1_Q10_s32x4, rr_Q10_s16x4, rr_Q10_s16x4 );
            rd1_Q10_s32x4 = vshrq_n_s32( rd1_Q10_s32x4, 10 );

            rr_Q10_s16x4 = vsub_s16( r_Q10_s16x4, q2_Q10_s16x4 );
            rd2_Q10_s32x4 = vmlal_s16( rd2_Q10_s32x4, rr_Q10_s16x4, rr_Q10_s16x4 );
            rd2_Q10_s32x4 = vshrq_n_s32( rd2_Q10_s32x4, 10 );

            tmp2_s32x4 = vld1q_s32( psDelDec->RD_Q10 );
            tmp1_s32x4 = vaddq_s32( tmp2_s32x4, vminq_s32( rd1_Q10_s32x4, rd2_Q10_s32x4 ) );
            tmp2_s32x4 = vaddq_s32( tmp2_s32x4, vmaxq_s32( rd1_Q10_s32x4, rd2_Q10_s32x4 ) );
            vst1q_s32( psSampleState[ 0 ].RD_Q10, tmp1_s32x4 );
            vst1q_s32( psSampleState[ 1 ].RD_Q10, tmp2_s32x4 );
            t_u32x4 = vcltq_s32( rd1_Q10_s32x4, rd2_Q10_s32x4 );
            tmp1_s32x4 = vbslq_s32( t_u32x4, vmovl_s16( q1_Q10_s16x4 ), vmovl_s16( q2_Q10_s16x4 ) );
            tmp2_s32x4 = vbslq_s32( t_u32x4, vmovl_s16( q2_Q10_s16x4 ), vmovl_s16( q1_Q10_s16x4 ) );
            vst1q_s32( psSampleState[ 0 ].Q_Q10, tmp1_s32x4 );
            vst1q_s32( psSampleState[ 1 ].Q_Q10, tmp2_s32x4 );
        }

        {
            /* Update states for best quantization */
            int32x4_t exc_Q14_s32x4, LPC_exc_Q14_s32x4, xq_Q14_s32x4, sLF_AR_shp_Q14_s32x4;

            /* Quantized excitation */
            exc_Q14_s32x4 = vshlq_n_s32( tmp1_s32x4, 4 );
            exc_Q14_s32x4 = veorq_s32( exc_Q14_s32x4, sign_s32x4 );
            exc_Q14_s32x4 = vsubq_s32( exc_Q14_s32x4, sign_s32x4 );

            /* Add predictions */
            LPC_exc_Q14_s32x4 = vaddq_s32( exc_Q14_s32x4, vdupq_n_s32( LTP_pred_Q14 ) );
            xq_Q14_s32x4      = vaddq_s32( LPC_exc_Q14_s32x4, LPC_pred_Q14_s32x4 );

            /* Update states */
            tmp1_s32x4 = vsubq_s32( xq_Q14_s32x4, vshlq_n_s32( vdupq_n_s32( x_Q10[ i ] ), 4 ) );
            vst1q_s32( psSampleState[ 0 ].Diff_Q14, tmp1_s32x4 );
            sLF_AR_shp_Q14_s32x4 = vsubq_s32( tmp1_s32x4, n_AR_Q14_s32x4 );
            vst1q_s32( psSampleState[ 0 ].sLTP_shp_Q14, vsubq_s32( sLF_AR_shp_Q14_s32x4, n_LF_Q14_s32x4 ) );
            vst1q_s32( psSampleState[ 0 ].LF_AR_Q14, sLF_AR_shp_Q14_s32x4 );
            vst1q_s32( psSampleState[ 0 ].LPC_exc_Q14, LPC_exc_Q14_s32x4 );
            vst1q_s32( psSampleState[ 0 ].xq_Q14, xq_Q14_s32x4 );

            /* Quantized excitation */
            exc_Q14_s32x4 = vshlq_n_s32( tmp2_s32x4, 4 );
            exc_Q14_s32x4 = veorq_s32( exc_Q14_s32x4, sign_s32x4 );
            exc_Q14_s32x4 = vsubq_s32( exc_Q14_s32x4, sign_s32x4 );

            /* Add predictions */
            LPC_exc_Q14_s32x4 = vaddq_s32( exc_Q14_s32x4, vdupq_n_s32( LTP_pred_Q14 ) );
            xq_Q14_s32x4      = vaddq_s32( LPC_exc_Q14_s32x4, LPC_pred_Q14_s32x4 );

            /* Update states */
            tmp1_s32x4 = vsubq_s32( xq_Q14_s32x4, vshlq_n_s32( vdupq_n_s32( x_Q10[ i ] ), 4 ) );
            vst1q_s32( psSampleState[ 1 ].Diff_Q14, tmp1_s32x4 );
            sLF_AR_shp_Q14_s32x4 = vsubq_s32( tmp1_s32x4, n_AR_Q14_s32x4 );
            vst1q_s32( psSampleState[ 1 ].sLTP_shp_Q14, vsubq_s32( sLF_AR_shp_Q14_s32x4, n_LF_Q14_s32x4 ) );
            vst1q_s32( psSampleState[ 1 ].LF_AR_Q14, sLF_AR_shp_Q14_s32x4 );
            vst1q_s32( psSampleState[ 1 ].LPC_exc_Q14, LPC_exc_Q14_s32x4 );
            vst1q_s32( psSampleState[ 1 ].xq_Q14, xq_Q14_s32x4 );
        }

        *smpl_buf_idx = *smpl_buf_idx ? ( *smpl_buf_idx - 1 ) : ( DECISION_DELAY - 1);
        last_smple_idx = *smpl_buf_idx + decisionDelay + DECISION_DELAY;
        if( last_smple_idx >= DECISION_DELAY ) last_smple_idx -= DECISION_DELAY;
        if( last_smple_idx >= DECISION_DELAY ) last_smple_idx -= DECISION_DELAY;

        /* Find winner */
        RDmin_Q10 = psSampleState[ 0 ].RD_Q10[ 0 ];
        Winner_ind = 0;
        for( k = 1; k < nStatesDelayedDecision; k++ ) {
            if( psSampleState[ 0 ].RD_Q10[ k ] < RDmin_Q10 ) {
                RDmin_Q10 = psSampleState[ 0 ].RD_Q10[ k ];
                Winner_ind = k;
            }
        }

        /* clear unused part of RD_Q10 to avoid overflows */
        if( nStatesDelayedDecision < NEON_MAX_DEL_DEC_STATES )
        {
            OPUS_CLEAR(psSampleState[0].RD_Q10 + nStatesDelayedDecision, NEON_MAX_DEL_DEC_STATES - nStatesDelayedDecision);
            OPUS_CLEAR(psSampleState[1].RD_Q10 + nStatesDelayedDecision, NEON_MAX_DEL_DEC_STATES - nStatesDelayedDecision);
        }

        /* Increase RD values of expired states */
        {
            uint32x4_t t_u32x4;
            Winner_rand_state = psDelDec->RandState[ last_smple_idx ][ Winner_ind ];
            t_u32x4 = vceqq_s32( vld1q_s32( psDelDec->RandState[ last_smple_idx ] ), vdupq_n_s32( Winner_rand_state ) );
            t_u32x4 = vmvnq_u32( t_u32x4 );
            t_u32x4 = vshrq_n_u32( t_u32x4, 5 );
            tmp1_s32x4 = vld1q_s32( psSampleState[ 0 ].RD_Q10 );
            tmp2_s32x4 = vld1q_s32( psSampleState[ 1 ].RD_Q10 );
            tmp1_s32x4 = vaddq_s32( tmp1_s32x4, vreinterpretq_s32_u32( t_u32x4 ) );
            tmp2_s32x4 = vaddq_s32( tmp2_s32x4, vreinterpretq_s32_u32( t_u32x4 ) );
            vst1q_s32( psSampleState[ 0 ].RD_Q10, tmp1_s32x4 );
            vst1q_s32( psSampleState[ 1 ].RD_Q10, tmp2_s32x4 );

            /* Find worst in first set and best in second set */
            RDmax_Q10 = psSampleState[ 0 ].RD_Q10[ 0 ];
            RDmin_Q10 = psSampleState[ 1 ].RD_Q10[ 0 ];
            RDmax_ind = 0;
            RDmin_ind = 0;
            for( k = 1; k < nStatesDelayedDecision; k++ ) {
                /* find worst in first set */
                if( psSampleState[ 0 ].RD_Q10[ k ] > RDmax_Q10 ) {
                    RDmax_Q10 = psSampleState[ 0 ].RD_Q10[ k ];
                    RDmax_ind = k;
                }
                /* find best in second set */
                if( psSampleState[ 1 ].RD_Q10[ k ] < RDmin_Q10 ) {
                    RDmin_Q10 = psSampleState[ 1 ].RD_Q10[ k ];
                    RDmin_ind = k;
                }
            }
        }

        /* Replace a state if best from second set outperforms worst in first set */
        if( RDmin_Q10 < RDmax_Q10 ) {
            opus_int32 (*ptr)[NEON_MAX_DEL_DEC_STATES] = psDelDec->RandState;
            const int numOthers = (int)( ( sizeof( NSQ_del_decs_struct ) - sizeof( ( (NSQ_del_decs_struct *)0 )->sLPC_Q14 ) )
                / ( NEON_MAX_DEL_DEC_STATES * sizeof( opus_int32 ) ) );
            /* Only ( predictLPCOrder - 1 ) of sLPC_Q14 buffer need to be updated, though the first several     */
            /* useless sLPC_Q14[] will be different comparing with C when predictLPCOrder < NSQ_LPC_BUF_LENGTH. */
            /* Here just update constant ( NSQ_LPC_BUF_LENGTH - 1 ) for simplicity.                             */
            for( j = i + 1; j < i + NSQ_LPC_BUF_LENGTH; j++ ) {
                psDelDec->sLPC_Q14[ j ][ RDmax_ind ] = psDelDec->sLPC_Q14[ j ][ RDmin_ind ];
            }
            for( j = 0; j < numOthers; j++ ) {
                ptr[ j ][ RDmax_ind ] = ptr[ j ][ RDmin_ind ];
            }

            psSampleState[ 0 ].Q_Q10[ RDmax_ind ] = psSampleState[ 1 ].Q_Q10[ RDmin_ind ];
            psSampleState[ 0 ].RD_Q10[ RDmax_ind ] = psSampleState[ 1 ].RD_Q10[ RDmin_ind ];
            psSampleState[ 0 ].xq_Q14[ RDmax_ind ] = psSampleState[ 1 ].xq_Q14[ RDmin_ind ];
            psSampleState[ 0 ].LF_AR_Q14[ RDmax_ind ] = psSampleState[ 1 ].LF_AR_Q14[ RDmin_ind ];
            psSampleState[ 0 ].Diff_Q14[ RDmax_ind ] = psSampleState[ 1 ].Diff_Q14[ RDmin_ind ];
            psSampleState[ 0 ].sLTP_shp_Q14[ RDmax_ind ] = psSampleState[ 1 ].sLTP_shp_Q14[ RDmin_ind ];
            psSampleState[ 0 ].LPC_exc_Q14[ RDmax_ind ] = psSampleState[ 1 ].LPC_exc_Q14[ RDmin_ind ];
        }

        /* Write samples from winner to output and long-term filter states */
        if( subfr > 0 || i >= decisionDelay ) {
            pulses[  i - decisionDelay ] = (opus_int8)silk_RSHIFT_ROUND( psDelDec->Q_Q10[ last_smple_idx ][ Winner_ind ], 10 );
            xq[ i - decisionDelay ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND(
                silk_SMULWW( psDelDec->Xq_Q14[ last_smple_idx ][ Winner_ind ], delayedGain_Q10[ last_smple_idx ] ), 8 ) );
            NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - decisionDelay ] = psDelDec->Shape_Q14[ last_smple_idx ][ Winner_ind ];
            sLTP_Q15[          NSQ->sLTP_buf_idx     - decisionDelay ] = psDelDec->Pred_Q15[  last_smple_idx ][ Winner_ind ];
        }
        NSQ->sLTP_shp_buf_idx++;
        NSQ->sLTP_buf_idx++;

        /* Update states */
        vst1q_s32( psDelDec->LF_AR_Q14, vld1q_s32( psSampleState[ 0 ].LF_AR_Q14 ) );
        vst1q_s32( psDelDec->Diff_Q14, vld1q_s32( psSampleState[ 0 ].Diff_Q14 ) );
        vst1q_s32( psDelDec->sLPC_Q14[ NSQ_LPC_BUF_LENGTH + i ], vld1q_s32( psSampleState[ 0 ].xq_Q14 ) );
        vst1q_s32( psDelDec->Xq_Q14[ *smpl_buf_idx ], vld1q_s32( psSampleState[ 0 ].xq_Q14 ) );
        tmp1_s32x4 = vld1q_s32( psSampleState[ 0 ].Q_Q10 );
        vst1q_s32( psDelDec->Q_Q10[ *smpl_buf_idx ], tmp1_s32x4 );
        vst1q_s32( psDelDec->Pred_Q15[ *smpl_buf_idx ], vshlq_n_s32( vld1q_s32( psSampleState[ 0 ].LPC_exc_Q14 ), 1 ) );
        vst1q_s32( psDelDec->Shape_Q14[ *smpl_buf_idx ], vld1q_s32( psSampleState[ 0 ].sLTP_shp_Q14 ) );
        tmp1_s32x4 = vrshrq_n_s32( tmp1_s32x4, 10 );
        tmp1_s32x4 = vreinterpretq_s32_u32( vaddq_u32( vreinterpretq_u32_s32(
            vld1q_s32( psDelDec->Seed ) ), vreinterpretq_u32_s32( tmp1_s32x4 ) ) );
        vst1q_s32( psDelDec->Seed, tmp1_s32x4 );
        vst1q_s32( psDelDec->RandState[ *smpl_buf_idx ], tmp1_s32x4 );
        vst1q_s32( psDelDec->RD_Q10, vld1q_s32( psSampleState[ 0 ].RD_Q10 ) );
        delayedGain_Q10[ *smpl_buf_idx ] = Gain_Q10;
    }
    /* Update LPC states */
    silk_memcpy( psDelDec->sLPC_Q14[ 0 ], psDelDec->sLPC_Q14[ length ], NEON_MAX_DEL_DEC_STATES * NSQ_LPC_BUF_LENGTH * sizeof( opus_int32 ) );

    RESTORE_STACK;
}

static OPUS_INLINE void silk_SMULWB_8_neon(
    const opus_int16 *a,
    const int32x2_t  b,
    opus_int32       *o
)
{
    const int16x8_t a_s16x8 = vld1q_s16( a );
    int32x4_t o0_s32x4, o1_s32x4;

    o0_s32x4 = vshll_n_s16( vget_low_s16( a_s16x8 ), 15 );
    o1_s32x4 = vshll_n_s16( vget_high_s16( a_s16x8 ), 15 );
    o0_s32x4 = vqdmulhq_lane_s32( o0_s32x4, b, 0 );
    o1_s32x4 = vqdmulhq_lane_s32( o1_s32x4, b, 0 );
    vst1q_s32( o, o0_s32x4 );
    vst1q_s32( o + 4, o1_s32x4 );
}

/* Only works when ( b >= -65536 ) && ( b < 65536 ). */
static OPUS_INLINE void silk_SMULWW_small_b_4_neon(
    opus_int32       *a,
    const int32x2_t  b_s32x2)
{
    int32x4_t o_s32x4;

    o_s32x4 = vld1q_s32( a );
    o_s32x4 = vqdmulhq_lane_s32( o_s32x4, b_s32x2, 0 );
    vst1q_s32( a, o_s32x4 );
}

/* Only works when ( b >= -65536 ) && ( b < 65536 ). */
static OPUS_INLINE void silk_SMULWW_small_b_8_neon(
    opus_int32       *a,
    const int32x2_t  b_s32x2
)
{
    int32x4_t o0_s32x4, o1_s32x4;

    o0_s32x4 = vld1q_s32( a );
    o1_s32x4 = vld1q_s32( a + 4 );
    o0_s32x4 = vqdmulhq_lane_s32( o0_s32x4, b_s32x2, 0 );
    o1_s32x4 = vqdmulhq_lane_s32( o1_s32x4, b_s32x2, 0 );
    vst1q_s32( a, o0_s32x4 );
    vst1q_s32( a + 4, o1_s32x4 );
}

static OPUS_INLINE void silk_SMULWW_4_neon(
    opus_int32       *a,
    const int32x2_t  b_s32x2)
{
    int32x4_t a_s32x4, o_s32x4;

    a_s32x4 = vld1q_s32( a );
    o_s32x4 = vqdmulhq_lane_s32( a_s32x4, b_s32x2, 0 );
    o_s32x4 = vmlaq_lane_s32( o_s32x4, a_s32x4, b_s32x2, 1 );
    vst1q_s32( a, o_s32x4 );
}

static OPUS_INLINE void silk_SMULWW_8_neon(
    opus_int32       *a,
    const int32x2_t  b_s32x2
)
{
    int32x4_t a0_s32x4, a1_s32x4, o0_s32x4, o1_s32x4;

    a0_s32x4 = vld1q_s32( a );
    a1_s32x4 = vld1q_s32( a + 4 );
    o0_s32x4 = vqdmulhq_lane_s32( a0_s32x4, b_s32x2, 0 );
    o1_s32x4 = vqdmulhq_lane_s32( a1_s32x4, b_s32x2, 0 );
    o0_s32x4 = vmlaq_lane_s32( o0_s32x4, a0_s32x4, b_s32x2, 1 );
    o1_s32x4 = vmlaq_lane_s32( o1_s32x4, a1_s32x4, b_s32x2, 1 );
    vst1q_s32( a, o0_s32x4 );
    vst1q_s32( a + 4, o1_s32x4 );
}

static OPUS_INLINE void silk_SMULWW_loop_neon(
    const opus_int16 *a,
    const opus_int32 b,
    opus_int32       *o,
    const opus_int   loop_num
)
{
    opus_int i;
    int32x2_t b_s32x2;

    b_s32x2 = vdup_n_s32( b );
    for( i = 0; i < loop_num - 7; i += 8 ) {
        silk_SMULWB_8_neon( a + i, b_s32x2, o + i );
    }
    for( ; i < loop_num; i++ ) {
        o[ i ] = silk_SMULWW( a[ i ], b );
    }
}

static OPUS_INLINE void silk_nsq_del_dec_scale_states_neon(
    const silk_encoder_state *psEncC,               /* I    Encoder State                       */
    silk_nsq_state      *NSQ,                       /* I/O  NSQ state                           */
    NSQ_del_decs_struct psDelDec[],                 /* I/O  Delayed decision states             */
    const opus_int16    x16[],                      /* I    Input                               */
    opus_int32          x_sc_Q10[],                 /* O    Input scaled with 1/Gain in Q10     */
    const opus_int16    sLTP[],                     /* I    Re-whitened LTP state in Q0         */
    opus_int32          sLTP_Q15[],                 /* O    LTP state matching scaled input     */
    opus_int            subfr,                      /* I    Subframe number                     */
    const opus_int      LTP_scale_Q14,              /* I    LTP state scaling                   */
    const opus_int32    Gains_Q16[ MAX_NB_SUBFR ],  /* I                                        */
    const opus_int      pitchL[ MAX_NB_SUBFR ],     /* I    Pitch lag                           */
    const opus_int      signal_type,                /* I    Signal type                         */
    const opus_int      decisionDelay               /* I    Decision delay                      */
)
{
    opus_int            i, lag;
    opus_int32          gain_adj_Q16, inv_gain_Q31, inv_gain_Q26;

    lag          = pitchL[ subfr ];
    inv_gain_Q31 = silk_INVERSE32_varQ( silk_max( Gains_Q16[ subfr ], 1 ), 47 );
    silk_assert( inv_gain_Q31 != 0 );

    /* Scale input */
    inv_gain_Q26 = silk_RSHIFT_ROUND( inv_gain_Q31, 5 );
    silk_SMULWW_loop_neon( x16, inv_gain_Q26, x_sc_Q10, psEncC->subfr_length );

    /* After rewhitening the LTP state is un-scaled, so scale with inv_gain_Q16 */
    if( NSQ->rewhite_flag ) {
        if( subfr == 0 ) {
            /* Do LTP downscaling */
            inv_gain_Q31 = silk_LSHIFT( silk_SMULWB( inv_gain_Q31, LTP_scale_Q14 ), 2 );
        }
        silk_SMULWW_loop_neon( sLTP + NSQ->sLTP_buf_idx - lag - LTP_ORDER / 2, inv_gain_Q31, sLTP_Q15 + NSQ->sLTP_buf_idx - lag - LTP_ORDER / 2, lag + LTP_ORDER / 2 );
    }

    /* Adjust for changing gain */
    if( Gains_Q16[ subfr ] != NSQ->prev_gain_Q16 ) {
        int32x2_t gain_adj_Q16_s32x2;
        gain_adj_Q16 = silk_DIV32_varQ( NSQ->prev_gain_Q16, Gains_Q16[ subfr ], 16 );

        /* Scale long-term shaping state */
        if( ( gain_adj_Q16 >= -65536 ) && ( gain_adj_Q16 < 65536 ) ) {
            gain_adj_Q16_s32x2 = vdup_n_s32( silk_LSHIFT32( gain_adj_Q16, 15 ) );
            for( i = NSQ->sLTP_shp_buf_idx - psEncC->ltp_mem_length; i < NSQ->sLTP_shp_buf_idx - 7; i += 8 ) {
                silk_SMULWW_small_b_8_neon( NSQ->sLTP_shp_Q14 + i, gain_adj_Q16_s32x2 );
            }
            for( ; i < NSQ->sLTP_shp_buf_idx; i++ ) {
                NSQ->sLTP_shp_Q14[ i ] = silk_SMULWW( gain_adj_Q16, NSQ->sLTP_shp_Q14[ i ] );
            }

            /* Scale long-term prediction state */
            if( signal_type == TYPE_VOICED && NSQ->rewhite_flag == 0 ) {
                for( i = NSQ->sLTP_buf_idx - lag - LTP_ORDER / 2; i < NSQ->sLTP_buf_idx - decisionDelay - 7; i += 8 ) {
                    silk_SMULWW_small_b_8_neon( sLTP_Q15 + i, gain_adj_Q16_s32x2 );
                }
                for( ; i < NSQ->sLTP_buf_idx - decisionDelay; i++ ) {
                    sLTP_Q15[ i ] = silk_SMULWW( gain_adj_Q16, sLTP_Q15[ i ] );
                }
            }

            /* Scale scalar states */
            silk_SMULWW_small_b_4_neon( psDelDec->LF_AR_Q14, gain_adj_Q16_s32x2 );
            silk_SMULWW_small_b_4_neon( psDelDec->Diff_Q14,  gain_adj_Q16_s32x2 );

            /* Scale short-term prediction and shaping states */
            for( i = 0; i < NSQ_LPC_BUF_LENGTH; i++ ) {
                silk_SMULWW_small_b_4_neon( psDelDec->sLPC_Q14[ i ], gain_adj_Q16_s32x2 );
            }

            for( i = 0; i < MAX_SHAPE_LPC_ORDER; i++ ) {
                silk_SMULWW_small_b_4_neon( psDelDec->sAR2_Q14[ i ], gain_adj_Q16_s32x2 );
            }

            for( i = 0; i < DECISION_DELAY; i++ ) {
                silk_SMULWW_small_b_4_neon( psDelDec->Pred_Q15[  i ], gain_adj_Q16_s32x2 );
                silk_SMULWW_small_b_4_neon( psDelDec->Shape_Q14[ i ], gain_adj_Q16_s32x2 );
            }
        } else {
            gain_adj_Q16_s32x2 = vdup_n_s32( silk_LSHIFT32( gain_adj_Q16 & 0x0000FFFF, 15 ) );
            gain_adj_Q16_s32x2 = vset_lane_s32( gain_adj_Q16 >> 16, gain_adj_Q16_s32x2, 1 );
            for( i = NSQ->sLTP_shp_buf_idx - psEncC->ltp_mem_length; i < NSQ->sLTP_shp_buf_idx - 7; i += 8 ) {
                silk_SMULWW_8_neon( NSQ->sLTP_shp_Q14 + i, gain_adj_Q16_s32x2 );
            }
            for( ; i < NSQ->sLTP_shp_buf_idx; i++ ) {
                NSQ->sLTP_shp_Q14[ i ] = silk_SMULWW( gain_adj_Q16, NSQ->sLTP_shp_Q14[ i ] );
            }

            /* Scale long-term prediction state */
            if( signal_type == TYPE_VOICED && NSQ->rewhite_flag == 0 ) {
                for( i = NSQ->sLTP_buf_idx - lag - LTP_ORDER / 2; i < NSQ->sLTP_buf_idx - decisionDelay - 7; i += 8 ) {
                    silk_SMULWW_8_neon( sLTP_Q15 + i, gain_adj_Q16_s32x2 );
                }
                for( ; i < NSQ->sLTP_buf_idx - decisionDelay; i++ ) {
                    sLTP_Q15[ i ] = silk_SMULWW( gain_adj_Q16, sLTP_Q15[ i ] );
                }
            }

            /* Scale scalar states */
            silk_SMULWW_4_neon( psDelDec->LF_AR_Q14, gain_adj_Q16_s32x2 );
            silk_SMULWW_4_neon( psDelDec->Diff_Q14,  gain_adj_Q16_s32x2 );

            /* Scale short-term prediction and shaping states */
            for( i = 0; i < NSQ_LPC_BUF_LENGTH; i++ ) {
                silk_SMULWW_4_neon( psDelDec->sLPC_Q14[ i ], gain_adj_Q16_s32x2 );
            }

            for( i = 0; i < MAX_SHAPE_LPC_ORDER; i++ ) {
                silk_SMULWW_4_neon( psDelDec->sAR2_Q14[ i ], gain_adj_Q16_s32x2 );
            }

            for( i = 0; i < DECISION_DELAY; i++ ) {
                silk_SMULWW_4_neon( psDelDec->Pred_Q15[  i ], gain_adj_Q16_s32x2 );
                silk_SMULWW_4_neon( psDelDec->Shape_Q14[ i ], gain_adj_Q16_s32x2 );
            }
        }

        /* Save inverse gain */
        NSQ->prev_gain_Q16 = Gains_Q16[ subfr ];
    }
}
