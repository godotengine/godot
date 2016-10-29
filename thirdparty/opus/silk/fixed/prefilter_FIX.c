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
#include "stack_alloc.h"
#include "tuning_parameters.h"

#if defined(MIPSr1_ASM)
#include "mips/prefilter_FIX_mipsr1.h"
#endif


#if !defined(OVERRIDE_silk_warped_LPC_analysis_filter_FIX)
#define silk_warped_LPC_analysis_filter_FIX(state, res_Q2, coef_Q13, input, lambda_Q16, length, order, arch) \
    ((void)(arch),silk_warped_LPC_analysis_filter_FIX_c(state, res_Q2, coef_Q13, input, lambda_Q16, length, order))
#endif

/* Prefilter for finding Quantizer input signal */
static OPUS_INLINE void silk_prefilt_FIX(
    silk_prefilter_state_FIX    *P,                         /* I/O  state                               */
    opus_int32                  st_res_Q12[],               /* I    short term residual signal          */
    opus_int32                  xw_Q3[],                    /* O    prefiltered signal                  */
    opus_int32                  HarmShapeFIRPacked_Q12,     /* I    Harmonic shaping coeficients        */
    opus_int                    Tilt_Q14,                   /* I    Tilt shaping coeficient             */
    opus_int32                  LF_shp_Q14,                 /* I    Low-frequancy shaping coeficients   */
    opus_int                    lag,                        /* I    Lag for harmonic shaping            */
    opus_int                    length                      /* I    Length of signals                   */
);

void silk_warped_LPC_analysis_filter_FIX_c(
          opus_int32            state[],                    /* I/O  State [order + 1]                   */
          opus_int32            res_Q2[],                   /* O    Residual signal [length]            */
    const opus_int16            coef_Q13[],                 /* I    Coefficients [order]                */
    const opus_int16            input[],                    /* I    Input signal [length]               */
    const opus_int16            lambda_Q16,                 /* I    Warping factor                      */
    const opus_int              length,                     /* I    Length of input signal              */
    const opus_int              order                       /* I    Filter order (even)                 */
)
{
    opus_int     n, i;
    opus_int32   acc_Q11, tmp1, tmp2;

    /* Order must be even */
    silk_assert( ( order & 1 ) == 0 );

    for( n = 0; n < length; n++ ) {
        /* Output of lowpass section */
        tmp2 = silk_SMLAWB( state[ 0 ], state[ 1 ], lambda_Q16 );
        state[ 0 ] = silk_LSHIFT( input[ n ], 14 );
        /* Output of allpass section */
        tmp1 = silk_SMLAWB( state[ 1 ], state[ 2 ] - tmp2, lambda_Q16 );
        state[ 1 ] = tmp2;
        acc_Q11 = silk_RSHIFT( order, 1 );
        acc_Q11 = silk_SMLAWB( acc_Q11, tmp2, coef_Q13[ 0 ] );
        /* Loop over allpass sections */
        for( i = 2; i < order; i += 2 ) {
            /* Output of allpass section */
            tmp2 = silk_SMLAWB( state[ i ], state[ i + 1 ] - tmp1, lambda_Q16 );
            state[ i ] = tmp1;
            acc_Q11 = silk_SMLAWB( acc_Q11, tmp1, coef_Q13[ i - 1 ] );
            /* Output of allpass section */
            tmp1 = silk_SMLAWB( state[ i + 1 ], state[ i + 2 ] - tmp2, lambda_Q16 );
            state[ i + 1 ] = tmp2;
            acc_Q11 = silk_SMLAWB( acc_Q11, tmp2, coef_Q13[ i ] );
        }
        state[ order ] = tmp1;
        acc_Q11 = silk_SMLAWB( acc_Q11, tmp1, coef_Q13[ order - 1 ] );
        res_Q2[ n ] = silk_LSHIFT( (opus_int32)input[ n ], 2 ) - silk_RSHIFT_ROUND( acc_Q11, 9 );
    }
}

void silk_prefilter_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  Encoder state                                                               */
    const silk_encoder_control_FIX  *psEncCtrl,                             /* I    Encoder control                                                             */
    opus_int32                      xw_Q3[],                                /* O    Weighted signal                                                             */
    const opus_int16                x[]                                     /* I    Speech signal                                                               */
)
{
    silk_prefilter_state_FIX *P = &psEnc->sPrefilt;
    opus_int   j, k, lag;
    opus_int32 tmp_32;
    const opus_int16 *AR1_shp_Q13;
    const opus_int16 *px;
    opus_int32 *pxw_Q3;
    opus_int   HarmShapeGain_Q12, Tilt_Q14;
    opus_int32 HarmShapeFIRPacked_Q12, LF_shp_Q14;
    VARDECL( opus_int32, x_filt_Q12 );
    VARDECL( opus_int32, st_res_Q2 );
    opus_int16 B_Q10[ 2 ];
    SAVE_STACK;

    /* Set up pointers */
    px  = x;
    pxw_Q3 = xw_Q3;
    lag = P->lagPrev;
    ALLOC( x_filt_Q12, psEnc->sCmn.subfr_length, opus_int32 );
    ALLOC( st_res_Q2, psEnc->sCmn.subfr_length, opus_int32 );
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        /* Update Variables that change per sub frame */
        if( psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
            lag = psEncCtrl->pitchL[ k ];
        }

        /* Noise shape parameters */
        HarmShapeGain_Q12 = silk_SMULWB( (opus_int32)psEncCtrl->HarmShapeGain_Q14[ k ], 16384 - psEncCtrl->HarmBoost_Q14[ k ] );
        silk_assert( HarmShapeGain_Q12 >= 0 );
        HarmShapeFIRPacked_Q12  =                          silk_RSHIFT( HarmShapeGain_Q12, 2 );
        HarmShapeFIRPacked_Q12 |= silk_LSHIFT( (opus_int32)silk_RSHIFT( HarmShapeGain_Q12, 1 ), 16 );
        Tilt_Q14    = psEncCtrl->Tilt_Q14[   k ];
        LF_shp_Q14  = psEncCtrl->LF_shp_Q14[ k ];
        AR1_shp_Q13 = &psEncCtrl->AR1_Q13[   k * MAX_SHAPE_LPC_ORDER ];

        /* Short term FIR filtering*/
        silk_warped_LPC_analysis_filter_FIX( P->sAR_shp, st_res_Q2, AR1_shp_Q13, px,
            psEnc->sCmn.warping_Q16, psEnc->sCmn.subfr_length, psEnc->sCmn.shapingLPCOrder, psEnc->sCmn.arch );

        /* Reduce (mainly) low frequencies during harmonic emphasis */
        B_Q10[ 0 ] = silk_RSHIFT_ROUND( psEncCtrl->GainsPre_Q14[ k ], 4 );
        tmp_32 = silk_SMLABB( SILK_FIX_CONST( INPUT_TILT, 26 ), psEncCtrl->HarmBoost_Q14[ k ], HarmShapeGain_Q12 );   /* Q26 */
        tmp_32 = silk_SMLABB( tmp_32, psEncCtrl->coding_quality_Q14, SILK_FIX_CONST( HIGH_RATE_INPUT_TILT, 12 ) );    /* Q26 */
        tmp_32 = silk_SMULWB( tmp_32, -psEncCtrl->GainsPre_Q14[ k ] );                                                /* Q24 */
        tmp_32 = silk_RSHIFT_ROUND( tmp_32, 14 );                                                                     /* Q10 */
        B_Q10[ 1 ]= silk_SAT16( tmp_32 );
        x_filt_Q12[ 0 ] = silk_MLA( silk_MUL( st_res_Q2[ 0 ], B_Q10[ 0 ] ), P->sHarmHP_Q2, B_Q10[ 1 ] );
        for( j = 1; j < psEnc->sCmn.subfr_length; j++ ) {
            x_filt_Q12[ j ] = silk_MLA( silk_MUL( st_res_Q2[ j ], B_Q10[ 0 ] ), st_res_Q2[ j - 1 ], B_Q10[ 1 ] );
        }
        P->sHarmHP_Q2 = st_res_Q2[ psEnc->sCmn.subfr_length - 1 ];

        silk_prefilt_FIX( P, x_filt_Q12, pxw_Q3, HarmShapeFIRPacked_Q12, Tilt_Q14, LF_shp_Q14, lag, psEnc->sCmn.subfr_length );

        px  += psEnc->sCmn.subfr_length;
        pxw_Q3 += psEnc->sCmn.subfr_length;
    }

    P->lagPrev = psEncCtrl->pitchL[ psEnc->sCmn.nb_subfr - 1 ];
    RESTORE_STACK;
}

#ifndef OVERRIDE_silk_prefilt_FIX
/* Prefilter for finding Quantizer input signal */
static OPUS_INLINE void silk_prefilt_FIX(
    silk_prefilter_state_FIX    *P,                         /* I/O  state                               */
    opus_int32                  st_res_Q12[],               /* I    short term residual signal          */
    opus_int32                  xw_Q3[],                    /* O    prefiltered signal                  */
    opus_int32                  HarmShapeFIRPacked_Q12,     /* I    Harmonic shaping coeficients        */
    opus_int                    Tilt_Q14,                   /* I    Tilt shaping coeficient             */
    opus_int32                  LF_shp_Q14,                 /* I    Low-frequancy shaping coeficients   */
    opus_int                    lag,                        /* I    Lag for harmonic shaping            */
    opus_int                    length                      /* I    Length of signals                   */
)
{
    opus_int   i, idx, LTP_shp_buf_idx;
    opus_int32 n_LTP_Q12, n_Tilt_Q10, n_LF_Q10;
    opus_int32 sLF_MA_shp_Q12, sLF_AR_shp_Q12;
    opus_int16 *LTP_shp_buf;

    /* To speed up use temp variables instead of using the struct */
    LTP_shp_buf     = P->sLTP_shp;
    LTP_shp_buf_idx = P->sLTP_shp_buf_idx;
    sLF_AR_shp_Q12  = P->sLF_AR_shp_Q12;
    sLF_MA_shp_Q12  = P->sLF_MA_shp_Q12;

    for( i = 0; i < length; i++ ) {
        if( lag > 0 ) {
            /* unrolled loop */
            silk_assert( HARM_SHAPE_FIR_TAPS == 3 );
            idx = lag + LTP_shp_buf_idx;
            n_LTP_Q12 = silk_SMULBB(            LTP_shp_buf[ ( idx - HARM_SHAPE_FIR_TAPS / 2 - 1) & LTP_MASK ], HarmShapeFIRPacked_Q12 );
            n_LTP_Q12 = silk_SMLABT( n_LTP_Q12, LTP_shp_buf[ ( idx - HARM_SHAPE_FIR_TAPS / 2    ) & LTP_MASK ], HarmShapeFIRPacked_Q12 );
            n_LTP_Q12 = silk_SMLABB( n_LTP_Q12, LTP_shp_buf[ ( idx - HARM_SHAPE_FIR_TAPS / 2 + 1) & LTP_MASK ], HarmShapeFIRPacked_Q12 );
        } else {
            n_LTP_Q12 = 0;
        }

        n_Tilt_Q10 = silk_SMULWB( sLF_AR_shp_Q12, Tilt_Q14 );
        n_LF_Q10   = silk_SMLAWB( silk_SMULWT( sLF_AR_shp_Q12, LF_shp_Q14 ), sLF_MA_shp_Q12, LF_shp_Q14 );

        sLF_AR_shp_Q12 = silk_SUB32( st_res_Q12[ i ], silk_LSHIFT( n_Tilt_Q10, 2 ) );
        sLF_MA_shp_Q12 = silk_SUB32( sLF_AR_shp_Q12,  silk_LSHIFT( n_LF_Q10,   2 ) );

        LTP_shp_buf_idx = ( LTP_shp_buf_idx - 1 ) & LTP_MASK;
        LTP_shp_buf[ LTP_shp_buf_idx ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( sLF_MA_shp_Q12, 12 ) );

        xw_Q3[i] = silk_RSHIFT_ROUND( silk_SUB32( sLF_MA_shp_Q12, n_LTP_Q12 ), 9 );
    }

    /* Copy temp variable back to state */
    P->sLF_AR_shp_Q12   = sLF_AR_shp_Q12;
    P->sLF_MA_shp_Q12   = sLF_MA_shp_Q12;
    P->sLTP_shp_buf_idx = LTP_shp_buf_idx;
}
#endif /* OVERRIDE_silk_prefilt_FIX */
