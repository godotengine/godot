/***********************************************************************
Copyright (c) 2021 Google Inc.
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

#ifdef OPUS_CHECK_ASM
#include <string.h>
#endif

#include "opus_defines.h"
#include <immintrin.h>

#include "main.h"
#include "stack_alloc.h"
#include "NSQ.h"
#include "celt/x86/x86cpu.h"

/* Returns TRUE if all assumptions met */
static OPUS_INLINE int verify_assumptions(const silk_encoder_state *psEncC)
{
    /* This optimization is based on these assumptions        */
    /* These assumptions are fundamental and hence assert are */
    /* used. Should any assert triggers, we have to re-visit  */
    /* all related code to make sure it still functions the   */
    /* same as the C implementation.                          */
    silk_assert(MAX_DEL_DEC_STATES  <= 4      &&
                MAX_FRAME_LENGTH     % 4 == 0 &&
                MAX_SUB_FRAME_LENGTH % 4 == 0 &&
                LTP_MEM_LENGTH_MS    % 4 == 0 );
    silk_assert(psEncC->fs_kHz ==  8 ||
                psEncC->fs_kHz == 12 ||
                psEncC->fs_kHz == 16 );
    silk_assert(psEncC->nb_subfr <= MAX_NB_SUBFR &&
                psEncC->nb_subfr > 0             );
    silk_assert(psEncC->nStatesDelayedDecision <= MAX_DEL_DEC_STATES &&
                psEncC->nStatesDelayedDecision > 0                   );
    silk_assert(psEncC->ltp_mem_length == psEncC->fs_kHz * LTP_MEM_LENGTH_MS);

    /* Regressions were observed on certain AMD Zen CPUs when      */
    /* nStatesDelayedDecision is 1 or 2. Ideally we should detect  */
    /* these CPUs and enable this optimization on others; however, */
    /* there is no good way to do so under current OPUS framework. */
    return psEncC->nStatesDelayedDecision == 3 ||
           psEncC->nStatesDelayedDecision == 4;
}

/* Intrinsics not defined on MSVC */
#ifdef _MSC_VER
#include <Intsafe.h>
static inline int __builtin_sadd_overflow(opus_int32 a, opus_int32 b, opus_int32* res)
{
    *res = a+b;
    return (*res ^ a) & (*res ^ b) & 0x80000000;
}
static inline int __builtin_ctz(unsigned int x)
{
    DWORD res = 0;
    return _BitScanForward(&res, x) ? res : 32;
}
#endif

static OPUS_INLINE __m128i silk_cvtepi64_epi32_high(__m256i num)
{
    return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(num, _mm256_set_epi32(0, 0, 0, 0, 7, 5, 3, 1)));
}

static OPUS_INLINE opus_int16 silk_sat16(opus_int32 num)
{
    num = num > silk_int16_MAX ? silk_int16_MAX : num;
    num = num < silk_int16_MIN ? silk_int16_MIN : num;
    return num;
}

static OPUS_INLINE opus_int32 silk_sar_round_32(opus_int32 a, int bits)
{
    silk_assert(bits > 0 && bits < 31);
    a += 1 << (bits-1);
    return a >> bits;
}

static OPUS_INLINE opus_int64 silk_sar_round_smulww(opus_int32 a, opus_int32 b, int bits)
{
    silk_assert(bits > 0 && bits < 63);
#ifdef OPUS_CHECK_ASM
    return silk_RSHIFT_ROUND(silk_SMULWW(a, b), bits);
#else
    /* This code is more correct, but it won't overflow like the C code in some rare cases. */
    silk_assert(bits > 0 && bits < 63);
    opus_int64 t = ((opus_int64)a) * ((opus_int64)b);
    bits += 16;
    t += 1ull << (bits-1);
    return t >> bits;
#endif
}

static OPUS_INLINE opus_int32 silk_add_sat32(opus_int32 a, opus_int32 b)
{
    opus_int32 sum;
    if (__builtin_sadd_overflow(a, b, &sum))
    {
        return a >= 0 ? silk_int32_MAX : silk_int32_MIN;
    }
    return sum;
}

static OPUS_INLINE __m128i silk_mm_srai_round_epi32(__m128i a, int bits)
{
    silk_assert(bits > 0 && bits < 31);
    return _mm_srai_epi32(_mm_add_epi32(a, _mm_set1_epi32(1 << (bits - 1))), bits);
}

/* add/subtract with output saturated */
static OPUS_INLINE __m128i silk_mm_add_sat_epi32(__m128i a, __m128i b)
{
    __m128i r = _mm_add_epi32(a, b);
    __m128i OF = _mm_and_si128(_mm_xor_si128(a, r), _mm_xor_si128(b, r));           /* OF = (sum ^ a) & (sum ^ b)   */
    __m128i SAT = _mm_add_epi32(_mm_srli_epi32(a, 31), _mm_set1_epi32(0x7FFFFFFF)); /* SAT = (a >> 31) + 0x7FFFFFFF */
    return _mm_blendv_epi8(r, SAT, _mm_srai_epi32(OF, 31));
}
static OPUS_INLINE __m128i silk_mm_sub_sat_epi32(__m128i a, __m128i b)
{
    __m128i r = _mm_sub_epi32(a, b);
    __m128i OF = _mm_andnot_si128(_mm_xor_si128(b, r), _mm_xor_si128(a, r));        /* OF = (sum ^ a) & (sum ^ ~b) = (sum ^ a) & ~(sum ^ b) */
    __m128i SAT = _mm_add_epi32(_mm_srli_epi32(a, 31), _mm_set1_epi32(0x7FFFFFFF)); /* SAT = (a >> 31) + 0x7FFFFFFF                         */
    return _mm_blendv_epi8(r, SAT, _mm_srai_epi32(OF, 31));
}
static OPUS_INLINE __m256i silk_mm256_sub_sat_epi32(__m256i a, __m256i b)
{
    __m256i r = _mm256_sub_epi32(a, b);
    __m256i OF = _mm256_andnot_si256(_mm256_xor_si256(b, r), _mm256_xor_si256(a, r));        /* OF = (sum ^ a) & (sum ^ ~b) = (sum ^ a) & ~(sum ^ b) */
    __m256i SAT = _mm256_add_epi32(_mm256_srli_epi32(a, 31), _mm256_set1_epi32(0x7FFFFFFF)); /* SAT = (a >> 31) + 0x7FFFFFFF                         */
    return _mm256_blendv_epi8(r, SAT, _mm256_srai_epi32(OF, 31));
}

static OPUS_INLINE __m128i silk_mm_limit_epi32(__m128i num, opus_int32 limit1, opus_int32 limit2)
{
    opus_int32 lo = limit1 < limit2 ? limit1 : limit2;
    opus_int32 hi = limit1 > limit2 ? limit1 : limit2;

    num = _mm_min_epi32(num, _mm_set1_epi32(hi));
    num = _mm_max_epi32(num, _mm_set1_epi32(lo));
    return num;
}

/* cond < 0 ? -num : num */
static OPUS_INLINE __m128i silk_mm_sign_epi32(__m128i num, __m128i cond)
{
    return _mm_sign_epi32(num, _mm_or_si128(cond, _mm_set1_epi32(1)));
}
static OPUS_INLINE __m256i silk_mm256_sign_epi32(__m256i num, __m256i cond)
{
    return _mm256_sign_epi32(num, _mm256_or_si256(cond, _mm256_set1_epi32(1)));
}

/* (a32 * b32) >> 16 */
static OPUS_INLINE __m128i silk_mm_smulww_epi32(__m128i a, opus_int32 b)
{
    return silk_cvtepi64_epi32_high(_mm256_slli_epi64(_mm256_mul_epi32(_mm256_cvtepi32_epi64(a), _mm256_set1_epi32(b)), 16));
}

/* (a32 * (opus_int32)((opus_int16)(b32))) >> 16 output have to be 32bit int */
static OPUS_INLINE __m128i silk_mm_smulwb_epi32(__m128i a, opus_int32 b)
{
    return silk_cvtepi64_epi32_high(_mm256_mul_epi32(_mm256_cvtepi32_epi64(a), _mm256_set1_epi32(silk_LSHIFT(b, 16))));
}

/* (opus_int32)((opus_int16)(a3))) * (opus_int32)((opus_int16)(b32)) output have to be 32bit int */
static OPUS_INLINE __m256i silk_mm256_smulbb_epi32(__m256i a, __m256i b)
{
    const char FF = (char)0xFF;
    __m256i msk = _mm256_set_epi8(
        FF, FF, FF, FF, FF, FF, FF, FF, 13, 12, 9, 8, 5, 4, 1, 0,
        FF, FF, FF, FF, FF, FF, FF, FF, 13, 12, 9, 8, 5, 4, 1, 0);
    __m256i lo = _mm256_mullo_epi16(a, b);
    __m256i hi = _mm256_mulhi_epi16(a, b);
    lo = _mm256_shuffle_epi8(lo, msk);
    hi = _mm256_shuffle_epi8(hi, msk);
    return _mm256_unpacklo_epi16(lo, hi);
}

static OPUS_INLINE __m256i silk_mm256_reverse_epi32(__m256i v)
{
    v = _mm256_shuffle_epi32(v, 0x1B);
    v = _mm256_permute4x64_epi64(v, 0x4E);
    return v;
}

static OPUS_INLINE opus_int32 silk_mm256_hsum_epi32(__m256i v)
{
    __m128i sum = _mm_add_epi32(_mm256_extracti128_si256(v, 1), _mm256_extracti128_si256(v, 0));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
    return _mm_cvtsi128_si32(sum);
}

static OPUS_INLINE __m128i silk_mm_hmin_epi32(__m128i num)
{
    num = _mm_min_epi32(num, _mm_shuffle_epi32(num, 0x4E)); /* 0123 -> 2301 */
    num = _mm_min_epi32(num, _mm_shuffle_epi32(num, 0xB1)); /* 0123 -> 1032 */
    return num;
}

static OPUS_INLINE __m128i silk_mm_hmax_epi32(__m128i num)
{
    num = _mm_max_epi32(num, _mm_shuffle_epi32(num, 0x4E)); /* 0123 -> 2310 */
    num = _mm_max_epi32(num, _mm_shuffle_epi32(num, 0xB1)); /* 0123 -> 1032 */
    return num;
}

static OPUS_INLINE __m128i silk_mm_mask_hmin_epi32(__m128i num, __m128i mask)
{
    num = _mm_blendv_epi8(num, _mm_set1_epi32(silk_int32_MAX), mask);
    return silk_mm_hmin_epi32(num);
}

static OPUS_INLINE __m128i silk_mm_mask_hmax_epi32(__m128i num, __m128i mask)
{
    num = _mm_blendv_epi8(num, _mm_set1_epi32(silk_int32_MIN), mask);
    return silk_mm_hmax_epi32(num);
}

static OPUS_INLINE __m128i silk_mm256_rand_epi32(__m128i seed)
{
    seed = _mm_mullo_epi32(seed, _mm_set1_epi32(RAND_MULTIPLIER));
    seed = _mm_add_epi32(seed, _mm_set1_epi32(RAND_INCREMENT));
    return seed;
}

static OPUS_INLINE opus_int32 silk_index_of_first_equal_epi32(__m128i a, __m128i b)
{
    unsigned int mask = _mm_movemask_epi8(_mm_cmpeq_epi32(a, b)) & 0x1111;
    silk_assert(mask != 0);
    return __builtin_ctz(mask) >> 2;
}

static __m128i silk_index_to_selector(opus_int32 index)
{
    silk_assert(index < 4);
    index <<= 2;
    return _mm_set_epi8(
        index + 3, index + 2, index + 1, index + 0,
        index + 3, index + 2, index + 1, index + 0,
        index + 3, index + 2, index + 1, index + 0,
        index + 3, index + 2, index + 1, index + 0);
}

static opus_int32 silk_select_winner(__m128i num, __m128i selector)
{
    return _mm_cvtsi128_si32(_mm_shuffle_epi8(num, selector));
}

typedef struct
{
    __m128i RandState;
    __m128i Q_Q10;
    __m128i Xq_Q14;
    __m128i Pred_Q15;
    __m128i Shape_Q14;
} NSQ_del_dec_sample_struct;

typedef struct
{
    __m128i sLPC_Q14[MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH];
    __m128i LF_AR_Q14;
    __m128i Seed;
    __m128i SeedInit;
    __m128i RD_Q10;
    __m128i Diff_Q14;
    __m128i sAR2_Q14[MAX_SHAPE_LPC_ORDER];
    NSQ_del_dec_sample_struct Samples[DECISION_DELAY];
} NSQ_del_dec_struct;

static OPUS_INLINE void silk_nsq_del_dec_scale_states_avx2(
    const silk_encoder_state *psEncC,          /* I    Encoder State                   */
    silk_nsq_state *NSQ,                       /* I/O  NSQ state                       */
    NSQ_del_dec_struct *psDelDec,              /* I/O  Delayed decision states         */
    const opus_int16 x16[],                    /* I    Input                           */
    opus_int32 x_sc_Q10[MAX_SUB_FRAME_LENGTH], /* O    Input scaled with 1/Gain in Q10 */
    const opus_int16 sLTP[],                   /* I    Re-whitened LTP state in Q0     */
    opus_int32 sLTP_Q15[],                     /* O    LTP state matching scaled input */
    opus_int subfr,                            /* I    Subframe number                 */
    const opus_int LTP_scale_Q14,              /* I    LTP state scaling               */
    const opus_int32 Gains_Q16[MAX_NB_SUBFR],  /* I                                    */
    const opus_int pitchL[MAX_NB_SUBFR],       /* I    Pitch lag                       */
    const opus_int signal_type,                /* I    Signal type                     */
    const opus_int decisionDelay               /* I    Decision delay                  */
);

/*******************************************/
/* LPC analysis filter                     */
/* NB! State is kept internally and the    */
/* filter always starts with zero state    */
/* first d output samples are set to zero  */
/*******************************************/
static OPUS_INLINE void silk_LPC_analysis_filter_avx2(
    opus_int16                  *out,               /* O    Output signal                           */
    const opus_int16            *in,                /* I    Input signal                            */
    const opus_int16            *B,                 /* I    MA prediction coefficients, Q12 [order] */
    const opus_int32            len,                /* I    Signal length                           */
    const opus_int32            order               /* I    Filter order                            */
);

/******************************************/
/* Noise shape quantizer for one subframe */
/******************************************/
static OPUS_INLINE void silk_noise_shape_quantizer_del_dec_avx2(
    silk_nsq_state *NSQ,                        /* I/O  NSQ state                          */
    NSQ_del_dec_struct psDelDec[],              /* I/O  Delayed decision states            */
    opus_int signalType,                        /* I    Signal type                        */
    const opus_int32 x_Q10[],                   /* I                                       */
    opus_int8 pulses[],                         /* O                                       */
    opus_int16 xq[],                            /* O                                       */
    opus_int32 sLTP_Q15[],                      /* I/O  LTP filter state                   */
    opus_int32 delayedGain_Q10[DECISION_DELAY], /* I/O  Gain delay buffer                  */
    const opus_int16 a_Q12[],                   /* I    Short term prediction coefs        */
    const opus_int16 b_Q14[],                   /* I    Long term prediction coefs         */
    const opus_int16 AR_shp_Q13[],              /* I    Noise shaping coefs                */
    opus_int lag,                               /* I    Pitch lag                          */
    opus_int32 HarmShapeFIRPacked_Q14,          /* I                                       */
    opus_int Tilt_Q14,                          /* I    Spectral tilt                      */
    opus_int32 LF_shp_Q14,                      /* I                                       */
    opus_int32 Gain_Q16,                        /* I                                       */
    opus_int Lambda_Q10,                        /* I                                       */
    opus_int offset_Q10,                        /* I                                       */
    opus_int length,                            /* I    Input length                       */
    opus_int subfr,                             /* I    Subframe number                    */
    opus_int shapingLPCOrder,                   /* I    Shaping LPC filter order           */
    opus_int predictLPCOrder,                   /* I    Prediction filter order            */
    opus_int warping_Q16,                       /* I                                       */
    __m128i MaskDelDec,                         /* I    Mask of states in decision tree    */
    opus_int *smpl_buf_idx,                     /* I/O  Index to newest samples in buffers */
    opus_int decisionDelay                      /* I                                       */
);

void silk_NSQ_del_dec_avx2(
    const silk_encoder_state *psEncC,                            /* I    Encoder State               */
    silk_nsq_state *NSQ,                                         /* I/O  NSQ state                   */
    SideInfoIndices *psIndices,                                  /* I/O  Quantization Indices        */
    const opus_int16 x16[],                                      /* I    Input                       */
    opus_int8 pulses[],                                          /* O    Quantized pulse signal      */
    const opus_int16 *PredCoef_Q12,                              /* I    Short term prediction coefs */
    const opus_int16 LTPCoef_Q14[LTP_ORDER * MAX_NB_SUBFR],      /* I    Long term prediction coefs  */
    const opus_int16 AR_Q13[MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER], /* I    Noise shaping coefs         */
    const opus_int HarmShapeGain_Q14[MAX_NB_SUBFR],              /* I    Long term shaping coefs     */
    const opus_int Tilt_Q14[MAX_NB_SUBFR],                       /* I    Spectral tilt               */
    const opus_int32 LF_shp_Q14[MAX_NB_SUBFR],                   /* I    Low frequency shaping coefs */
    const opus_int32 Gains_Q16[MAX_NB_SUBFR],                    /* I    Quantization step sizes     */
    const opus_int32 pitchL[MAX_NB_SUBFR],                       /* I    Pitch lags                  */
    const opus_int Lambda_Q10,                                   /* I    Rate/distortion tradeoff    */
    const opus_int LTP_scale_Q14                                 /* I    LTP state scaling           */
)
{
#ifdef OPUS_CHECK_ASM
    silk_nsq_state NSQ_c;
    SideInfoIndices psIndices_c;
    opus_int8 pulses_c[MAX_FRAME_LENGTH];
    const opus_int8 *const pulses_a = pulses;

    silk_memcpy(&NSQ_c, NSQ, sizeof(NSQ_c));
    silk_memcpy(&psIndices_c, psIndices, sizeof(psIndices_c));
    silk_memcpy(pulses_c, pulses, sizeof(pulses_c));
    silk_NSQ_del_dec_c(psEncC, &NSQ_c, &psIndices_c, x16, pulses_c, PredCoef_Q12, LTPCoef_Q14, AR_Q13, HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16,
                       pitchL, Lambda_Q10, LTP_scale_Q14);
#endif

    if (!verify_assumptions(psEncC))
    {
        silk_NSQ_del_dec_c(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14);
        return;
    }

    opus_int i, k, lag, start_idx, LSF_interpolation_flag, Winner_ind, subfr;
    opus_int last_smple_idx, smpl_buf_idx, decisionDelay;
    const opus_int16 *A_Q12, *B_Q14, *AR_shp_Q13;
    opus_int16 *pxq;
    VARDECL(opus_int32, sLTP_Q15);
    VARDECL(opus_int16, sLTP);
    opus_int32 HarmShapeFIRPacked_Q14;
    opus_int offset_Q10;
    opus_int32 Gain_Q10;
    opus_int32 x_sc_Q10[MAX_SUB_FRAME_LENGTH];
    opus_int32 delayedGain_Q10[DECISION_DELAY];
    NSQ_del_dec_struct psDelDec = {0};
    NSQ_del_dec_sample_struct *psSample;
    __m128i RDmin_Q10, MaskDelDec, Winner_selector;
    SAVE_STACK;

    MaskDelDec = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(0xFFFFFF00ul << ((psEncC->nStatesDelayedDecision - 1) << 3)));

    /* Set unvoiced lag to the previous one, overwrite later for voiced */
    lag = NSQ->lagPrev;

    silk_assert(NSQ->prev_gain_Q16 != 0);
    psDelDec.Seed = _mm_and_si128(
        _mm_add_epi32(_mm_set_epi32(3, 2, 1, 0), _mm_set1_epi32(psIndices->Seed)),
        _mm_set1_epi32(3));
    psDelDec.SeedInit = psDelDec.Seed;
    psDelDec.RD_Q10 = _mm_setzero_si128();
    psDelDec.LF_AR_Q14 = _mm_set1_epi32(NSQ->sLF_AR_shp_Q14);
    psDelDec.Diff_Q14 = _mm_set1_epi32(NSQ->sDiff_shp_Q14);
    psDelDec.Samples[0].Shape_Q14 = _mm_set1_epi32(NSQ->sLTP_shp_Q14[psEncC->ltp_mem_length - 1]);
    for (i = 0; i < NSQ_LPC_BUF_LENGTH; i++)
    {
        psDelDec.sLPC_Q14[i] = _mm_set1_epi32(NSQ->sLPC_Q14[i]);
    }
    for (i = 0; i < MAX_SHAPE_LPC_ORDER; i++)
    {
        psDelDec.sAR2_Q14[i] = _mm_set1_epi32(NSQ->sAR2_Q14[i]);
    }

    offset_Q10 = silk_Quantization_Offsets_Q10[psIndices->signalType >> 1][psIndices->quantOffsetType];
    smpl_buf_idx = 0; /* index of oldest samples */

    decisionDelay = silk_min_int(DECISION_DELAY, psEncC->subfr_length);

    /* For voiced frames limit the decision delay to lower than the pitch lag */
    if (psIndices->signalType == TYPE_VOICED)
    {
        for (k = 0; k < psEncC->nb_subfr; k++)
        {
            decisionDelay = silk_min_int(decisionDelay, pitchL[k] - LTP_ORDER / 2 - 1);
        }
    }
    else
    {
        if (lag > 0)
        {
            decisionDelay = silk_min_int(decisionDelay, lag - LTP_ORDER / 2 - 1);
        }
    }

    if (psIndices->NLSFInterpCoef_Q2 == 4)
    {
        LSF_interpolation_flag = 0;
    }
    else
    {
        LSF_interpolation_flag = 1;
    }

    ALLOC(sLTP_Q15, psEncC->ltp_mem_length + psEncC->frame_length, opus_int32);
    ALLOC(sLTP, psEncC->ltp_mem_length + psEncC->frame_length, opus_int16);
    /* Set up pointers to start of sub frame */
    pxq = &NSQ->xq[psEncC->ltp_mem_length];
    NSQ->sLTP_shp_buf_idx = psEncC->ltp_mem_length;
    NSQ->sLTP_buf_idx = psEncC->ltp_mem_length;
    subfr = 0;
    for (k = 0; k < psEncC->nb_subfr; k++)
    {
        A_Q12 = &PredCoef_Q12[((k >> 1) | (1 ^ LSF_interpolation_flag)) * MAX_LPC_ORDER];
        B_Q14 = &LTPCoef_Q14[k * LTP_ORDER];
        AR_shp_Q13 = &AR_Q13[k * MAX_SHAPE_LPC_ORDER];

        /* Noise shape parameters */
        silk_assert(HarmShapeGain_Q14[k] >= 0);
        HarmShapeFIRPacked_Q14  =                          silk_RSHIFT( HarmShapeGain_Q14[ k ], 2 );
        HarmShapeFIRPacked_Q14 |= silk_LSHIFT( (opus_int32)silk_RSHIFT( HarmShapeGain_Q14[ k ], 1 ), 16 );

        NSQ->rewhite_flag = 0;
        if (psIndices->signalType == TYPE_VOICED)
        {
            /* Voiced */
            lag = pitchL[k];

            /* Re-whitening */
            if ((k & (3 ^ (LSF_interpolation_flag << 1))) == 0)
            {
                if (k == 2)
                {
                    /* RESET DELAYED DECISIONS */
                    /* Find winner */
                    RDmin_Q10 = silk_mm_mask_hmin_epi32(psDelDec.RD_Q10, MaskDelDec);
                    Winner_ind = silk_index_of_first_equal_epi32(RDmin_Q10, psDelDec.RD_Q10);
                    Winner_selector = silk_index_to_selector(Winner_ind);
                    psDelDec.RD_Q10 = _mm_add_epi32(
                        psDelDec.RD_Q10,
                        _mm_blendv_epi8(
                            _mm_set1_epi32(silk_int32_MAX >> 4),
                            _mm_setzero_si128(),
                            _mm_cvtepi8_epi32(_mm_cvtsi32_si128(0xFFU << (unsigned)(Winner_ind << 3)))));

                    /* Copy final part of signals from winner state to output and long-term filter states */
                    last_smple_idx = smpl_buf_idx + decisionDelay;
                    for (i = 0; i < decisionDelay; i++)
                    {
                        last_smple_idx = (last_smple_idx + DECISION_DELAY - 1) % DECISION_DELAY;
                        psSample = &psDelDec.Samples[last_smple_idx];
                        pulses[i - decisionDelay] =
                            (opus_int8)silk_sar_round_32(silk_select_winner(psSample->Q_Q10, Winner_selector), 10);
                        pxq[i - decisionDelay] =
                            silk_sat16((opus_int32)silk_sar_round_smulww(silk_select_winner(psSample->Xq_Q14, Winner_selector), Gains_Q16[1], 14));
                        NSQ->sLTP_shp_Q14[NSQ->sLTP_shp_buf_idx - decisionDelay + i] =
                            silk_select_winner(psSample->Shape_Q14, Winner_selector);
                    }

                    subfr = 0;
                }

                /* Rewhiten with new A coefs */
                start_idx = psEncC->ltp_mem_length - lag - psEncC->predictLPCOrder - LTP_ORDER / 2;
                silk_assert(start_idx > 0);

                silk_LPC_analysis_filter_avx2(&sLTP[start_idx], &NSQ->xq[start_idx + k * psEncC->subfr_length],
                                              A_Q12, psEncC->ltp_mem_length - start_idx, psEncC->predictLPCOrder);

                NSQ->sLTP_buf_idx = psEncC->ltp_mem_length;
                NSQ->rewhite_flag = 1;
            }
        }

        silk_nsq_del_dec_scale_states_avx2(psEncC, NSQ, &psDelDec, x16, x_sc_Q10, sLTP, sLTP_Q15, k,
                                           LTP_scale_Q14, Gains_Q16, pitchL, psIndices->signalType, decisionDelay);

        silk_noise_shape_quantizer_del_dec_avx2(NSQ, &psDelDec, psIndices->signalType, x_sc_Q10, pulses, pxq, sLTP_Q15,
                                                delayedGain_Q10, A_Q12, B_Q14, AR_shp_Q13, lag, HarmShapeFIRPacked_Q14, Tilt_Q14[k], LF_shp_Q14[k],
                                                Gains_Q16[k], Lambda_Q10, offset_Q10, psEncC->subfr_length, subfr++, psEncC->shapingLPCOrder,
                                                psEncC->predictLPCOrder, psEncC->warping_Q16, MaskDelDec, &smpl_buf_idx, decisionDelay);

        x16 += psEncC->subfr_length;
        pulses += psEncC->subfr_length;
        pxq += psEncC->subfr_length;
    }

    /* Find winner */
    RDmin_Q10 = silk_mm_mask_hmin_epi32(psDelDec.RD_Q10, MaskDelDec);
    Winner_selector = silk_index_to_selector(silk_index_of_first_equal_epi32(RDmin_Q10, psDelDec.RD_Q10));

    /* Copy final part of signals from winner state to output and long-term filter states */
    psIndices->Seed = silk_select_winner(psDelDec.SeedInit, Winner_selector);
    last_smple_idx = smpl_buf_idx + decisionDelay;
    Gain_Q10 = Gains_Q16[psEncC->nb_subfr - 1] >> 6;
    for (i = 0; i < decisionDelay; i++)
    {
        last_smple_idx = (last_smple_idx + DECISION_DELAY - 1) % DECISION_DELAY;
        psSample = &psDelDec.Samples[last_smple_idx];

        pulses[i - decisionDelay] =
            (opus_int8)silk_sar_round_32(silk_select_winner(psSample->Q_Q10, Winner_selector), 10);
        pxq[i - decisionDelay] =
            silk_sat16((opus_int32)silk_sar_round_smulww(silk_select_winner(psSample->Xq_Q14, Winner_selector), Gain_Q10, 8));
        NSQ->sLTP_shp_Q14[NSQ->sLTP_shp_buf_idx - decisionDelay + i] =
            silk_select_winner(psSample->Shape_Q14, Winner_selector);
    }
    for (i = 0; i < NSQ_LPC_BUF_LENGTH; i++)
    {
        NSQ->sLPC_Q14[i] = silk_select_winner(psDelDec.sLPC_Q14[i], Winner_selector);
    }
    for (i = 0; i < MAX_SHAPE_LPC_ORDER; i++)
    {
        NSQ->sAR2_Q14[i] = silk_select_winner(psDelDec.sAR2_Q14[i], Winner_selector);
    }

    /* Update states */
    NSQ->sLF_AR_shp_Q14 = silk_select_winner(psDelDec.LF_AR_Q14, Winner_selector);
    NSQ->sDiff_shp_Q14 = silk_select_winner(psDelDec.Diff_Q14, Winner_selector);
    NSQ->lagPrev = pitchL[psEncC->nb_subfr - 1];

    /* Save quantized speech signal */
    silk_memmove(NSQ->xq, &NSQ->xq[psEncC->frame_length], psEncC->ltp_mem_length * sizeof(opus_int16));
    silk_memmove(NSQ->sLTP_shp_Q14, &NSQ->sLTP_shp_Q14[psEncC->frame_length], psEncC->ltp_mem_length * sizeof(opus_int32));

#ifdef OPUS_CHECK_ASM
    silk_assert(!memcmp(&NSQ_c, NSQ, sizeof(NSQ_c)));
    silk_assert(!memcmp(&psIndices_c, psIndices, sizeof(psIndices_c)));
    silk_assert(!memcmp(pulses_c, pulses_a, sizeof(pulses_c)));
#endif

    RESTORE_STACK;
}

static OPUS_INLINE __m128i silk_noise_shape_quantizer_short_prediction_x4(const __m128i *buf32, const opus_int16 *coef16, opus_int order)
{
    __m256i out;
    silk_assert(order == 10 || order == 16);

    /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
    out = _mm256_set1_epi32(order >> 1);
    out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-0]), _mm256_set1_epi32(silk_LSHIFT(coef16[0], 16)))); /* High DWORD */
    out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-1]), _mm256_set1_epi32(silk_LSHIFT(coef16[1], 16)))); /* High DWORD */
    out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-2]), _mm256_set1_epi32(silk_LSHIFT(coef16[2], 16)))); /* High DWORD */
    out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-3]), _mm256_set1_epi32(silk_LSHIFT(coef16[3], 16)))); /* High DWORD */
    out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-4]), _mm256_set1_epi32(silk_LSHIFT(coef16[4], 16)))); /* High DWORD */
    out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-5]), _mm256_set1_epi32(silk_LSHIFT(coef16[5], 16)))); /* High DWORD */
    out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-6]), _mm256_set1_epi32(silk_LSHIFT(coef16[6], 16)))); /* High DWORD */
    out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-7]), _mm256_set1_epi32(silk_LSHIFT(coef16[7], 16)))); /* High DWORD */
    out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-8]), _mm256_set1_epi32(silk_LSHIFT(coef16[8], 16)))); /* High DWORD */
    out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-9]), _mm256_set1_epi32(silk_LSHIFT(coef16[9], 16)))); /* High DWORD */

    if (order == 16)
    {
        out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-10]), _mm256_set1_epi32(silk_LSHIFT(coef16[10], 16)))); /* High DWORD */
        out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-11]), _mm256_set1_epi32(silk_LSHIFT(coef16[11], 16)))); /* High DWORD */
        out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-12]), _mm256_set1_epi32(silk_LSHIFT(coef16[12], 16)))); /* High DWORD */
        out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-13]), _mm256_set1_epi32(silk_LSHIFT(coef16[13], 16)))); /* High DWORD */
        out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-14]), _mm256_set1_epi32(silk_LSHIFT(coef16[14], 16)))); /* High DWORD */
        out = _mm256_add_epi32(out, _mm256_mul_epi32(_mm256_cvtepi32_epi64(buf32[-15]), _mm256_set1_epi32(silk_LSHIFT(coef16[15], 16)))); /* High DWORD */
    }
    return silk_cvtepi64_epi32_high(out);
}

/******************************************/
/* Noise shape quantizer for one subframe */
/******************************************/
static OPUS_INLINE void silk_noise_shape_quantizer_del_dec_avx2(
    silk_nsq_state *NSQ,                        /* I/O  NSQ state                          */
    NSQ_del_dec_struct *psDelDec,               /* I/O  Delayed decision states            */
    opus_int signalType,                        /* I    Signal type                        */
    const opus_int32 x_Q10[],                   /* I                                       */
    opus_int8 pulses[],                         /* O                                       */
    opus_int16 xq[],                            /* O                                       */
    opus_int32 sLTP_Q15[],                      /* I/O  LTP filter state                   */
    opus_int32 delayedGain_Q10[DECISION_DELAY], /* I/O  Gain delay buffer                  */
    const opus_int16 a_Q12[],                   /* I    Short term prediction coefs        */
    const opus_int16 b_Q14[],                   /* I    Long term prediction coefs         */
    const opus_int16 AR_shp_Q13[],              /* I    Noise shaping coefs                */
    opus_int lag,                               /* I    Pitch lag                          */
    opus_int32 HarmShapeFIRPacked_Q14,          /* I                                       */
    opus_int Tilt_Q14,                          /* I    Spectral tilt                      */
    opus_int32 LF_shp_Q14,                      /* I                                       */
    opus_int32 Gain_Q16,                        /* I                                       */
    opus_int Lambda_Q10,                        /* I                                       */
    opus_int offset_Q10,                        /* I                                       */
    opus_int length,                            /* I    Input length                       */
    opus_int subfr,                             /* I    Subframe number                    */
    opus_int shapingLPCOrder,                   /* I    Shaping LPC filter order           */
    opus_int predictLPCOrder,                   /* I    Prediction filter order            */
    opus_int warping_Q16,                       /* I                                       */
    __m128i MaskDelDec,                         /* I    Mask of states in decision tree    */
    opus_int *smpl_buf_idx,                     /* I/O  Index to newest samples in buffers */
    opus_int decisionDelay                      /* I                                       */
)
{
    int i;
    opus_int32 *shp_lag_ptr = &NSQ->sLTP_shp_Q14[NSQ->sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2];
    opus_int32 *pred_lag_ptr = &sLTP_Q15[NSQ->sLTP_buf_idx - lag + LTP_ORDER / 2];
    opus_int32 Gain_Q10 = Gain_Q16 >> 6;

    for (i = 0; i < length; i++)
    {
        /* Perform common calculations used in all states */
        /* NSQ_sample_struct */
        /* Low  128 bits => 1st set */
        /* High 128 bits => 2nd set */
        int j;
        __m256i SS_Q_Q10;
        __m256i SS_RD_Q10;
        __m256i SS_xq_Q14;
        __m256i SS_LF_AR_Q14;
        __m256i SS_Diff_Q14;
        __m256i SS_sLTP_shp_Q14;
        __m256i SS_LPC_exc_Q14;
        __m256i exc_Q14;
        __m256i q_Q10, rr_Q10, rd_Q10;
        __m256i mask;
        __m128i LPC_pred_Q14, n_AR_Q14;
        __m128i RDmin_Q10, RDmax_Q10;
        __m128i n_LF_Q14;
        __m128i r_Q10, q1_Q0, q1_Q10, q2_Q10;
        __m128i Winner_rand_state, Winner_selector;
        __m128i tmp0, tmp1;
        NSQ_del_dec_sample_struct *psLastSample, *psSample;
        opus_int32 RDmin_ind, RDmax_ind, last_smple_idx;
        opus_int32 LTP_pred_Q14, n_LTP_Q14;

        /* Long-term prediction */
        if (signalType == TYPE_VOICED)
        {
            /* Unrolled loop */
            /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
            LTP_pred_Q14 = 2;
            LTP_pred_Q14 += silk_SMULWB(pred_lag_ptr[-0], b_Q14[0]);
            LTP_pred_Q14 += silk_SMULWB(pred_lag_ptr[-1], b_Q14[1]);
            LTP_pred_Q14 += silk_SMULWB(pred_lag_ptr[-2], b_Q14[2]);
            LTP_pred_Q14 += silk_SMULWB(pred_lag_ptr[-3], b_Q14[3]);
            LTP_pred_Q14 += silk_SMULWB(pred_lag_ptr[-4], b_Q14[4]);
            LTP_pred_Q14 = silk_LSHIFT(LTP_pred_Q14, 1); /* Q13 -> Q14 */
            pred_lag_ptr++;
        }
        else
        {
            LTP_pred_Q14 = 0;
        }

        /* Long-term shaping */
        if (lag > 0)
        {
            /* Symmetric, packed FIR coefficients */
            n_LTP_Q14 = silk_add_sat32(shp_lag_ptr[0], shp_lag_ptr[-2]);
            n_LTP_Q14 = silk_SMULWB(n_LTP_Q14, HarmShapeFIRPacked_Q14);
            n_LTP_Q14 = n_LTP_Q14 + silk_SMULWT(shp_lag_ptr[-1], HarmShapeFIRPacked_Q14);
            n_LTP_Q14 = LTP_pred_Q14 - (silk_LSHIFT(n_LTP_Q14, 2)); /* Q12 -> Q14 */
            shp_lag_ptr++;
        }
        else
        {
            n_LTP_Q14 = 0;
        }

        /* BEGIN Updating Delayed Decision States */

        /* Generate dither */
        psDelDec->Seed = silk_mm256_rand_epi32(psDelDec->Seed);

        /* Short-term prediction */
        LPC_pred_Q14 = silk_noise_shape_quantizer_short_prediction_x4(&psDelDec->sLPC_Q14[NSQ_LPC_BUF_LENGTH - 1 + i], a_Q12, predictLPCOrder);
        LPC_pred_Q14 = _mm_slli_epi32(LPC_pred_Q14, 4); /* Q10 -> Q14 */

        /* Noise shape feedback */
        silk_assert(shapingLPCOrder > 0);
        silk_assert((shapingLPCOrder & 1) == 0); /* check that order is even */
        /* Output of lowpass section */
        tmp0 = _mm_add_epi32(psDelDec->Diff_Q14, silk_mm_smulwb_epi32(psDelDec->sAR2_Q14[0], warping_Q16));
        n_AR_Q14 = _mm_set1_epi32(shapingLPCOrder >> 1);
        for (j = 0; j < shapingLPCOrder - 1; j++)
        {
            /* Output of allpass section */
            tmp1 = psDelDec->sAR2_Q14[j];
            psDelDec->sAR2_Q14[j] = tmp0;
            n_AR_Q14 = _mm_add_epi32(n_AR_Q14, silk_mm_smulwb_epi32(tmp0, AR_shp_Q13[j]));
            tmp0 = _mm_add_epi32(tmp1, silk_mm_smulwb_epi32(_mm_sub_epi32(psDelDec->sAR2_Q14[j + 1], tmp0), warping_Q16));
        }
        psDelDec->sAR2_Q14[shapingLPCOrder - 1] = tmp0;
        n_AR_Q14 = _mm_add_epi32(n_AR_Q14, silk_mm_smulwb_epi32(tmp0, AR_shp_Q13[shapingLPCOrder - 1]));

        n_AR_Q14 = _mm_slli_epi32(n_AR_Q14, 1);                                                  /* Q11 -> Q12 */
        n_AR_Q14 = _mm_add_epi32(n_AR_Q14, silk_mm_smulwb_epi32(psDelDec->LF_AR_Q14, Tilt_Q14)); /* Q12 */
        n_AR_Q14 = _mm_slli_epi32(n_AR_Q14, 2);                                                  /* Q12 -> Q14 */

        tmp0 = silk_mm_smulwb_epi32(psDelDec->Samples[*smpl_buf_idx].Shape_Q14, LF_shp_Q14); /* Q12 */
        tmp1 = silk_mm_smulwb_epi32(psDelDec->LF_AR_Q14, LF_shp_Q14 >> 16);                  /* Q12 */
        n_LF_Q14 = _mm_add_epi32(tmp0, tmp1);                                                /* Q12 */
        n_LF_Q14 = _mm_slli_epi32(n_LF_Q14, 2);                                              /* Q12 -> Q14 */

        /* Input minus prediction plus noise feedback                       */
        /* r = x[ i ] - LTP_pred - LPC_pred + n_AR + n_Tilt + n_LF + n_LTP  */
        tmp0 = silk_mm_add_sat_epi32(n_AR_Q14, n_LF_Q14);              /* Q14 */
        tmp1 = _mm_add_epi32(_mm_set1_epi32(n_LTP_Q14), LPC_pred_Q14); /* Q13 */
        tmp0 = silk_mm_sub_sat_epi32(tmp1, tmp0);                      /* Q13 */
        tmp0 = silk_mm_srai_round_epi32(tmp0, 4);                      /* Q10 */

        r_Q10 = _mm_sub_epi32(_mm_set1_epi32(x_Q10[i]), tmp0); /* residual error Q10 */

        /* Flip sign depending on dither */
        r_Q10 = silk_mm_sign_epi32(r_Q10, psDelDec->Seed);
        r_Q10 = silk_mm_limit_epi32(r_Q10, -(31 << 10), 30 << 10);

        /* Find two quantization level candidates and measure their rate-distortion */
        q1_Q10 = _mm_sub_epi32(r_Q10, _mm_set1_epi32(offset_Q10));
        q1_Q0 = _mm_srai_epi32(q1_Q10, 10);
        if (Lambda_Q10 > 2048)
        {
            /* For aggressive RDO, the bias becomes more than one pulse. */
            tmp0 = _mm_sub_epi32(_mm_abs_epi32(q1_Q10), _mm_set1_epi32(Lambda_Q10 / 2 - 512)); /* rdo_offset */
            q1_Q0 = _mm_srai_epi32(q1_Q10, 31);
            tmp1 = _mm_cmpgt_epi32(tmp0, _mm_setzero_si128());
            tmp0 = _mm_srai_epi32(silk_mm_sign_epi32(tmp0, q1_Q10), 10);
            q1_Q0 = _mm_blendv_epi8(q1_Q0, tmp0, tmp1);
        }

        tmp0 = _mm_sign_epi32(_mm_set1_epi32(QUANT_LEVEL_ADJUST_Q10), q1_Q0);
        q1_Q10 = _mm_sub_epi32(_mm_slli_epi32(q1_Q0, 10), tmp0);
        q1_Q10 = _mm_add_epi32(q1_Q10, _mm_set1_epi32(offset_Q10));

        /* check if q1_Q0 is 0 or -1 */
        tmp0 = _mm_add_epi32(_mm_srli_epi32(q1_Q0, 31), q1_Q0);
        tmp1 = _mm_cmpeq_epi32(tmp0, _mm_setzero_si128());
        tmp0 = _mm_blendv_epi8(_mm_set1_epi32(1024), _mm_set1_epi32(1024 - QUANT_LEVEL_ADJUST_Q10), tmp1);
        q2_Q10 = _mm_add_epi32(q1_Q10, tmp0);
        q_Q10 = _mm256_set_m128i(q2_Q10, q1_Q10);

        rr_Q10 = _mm256_sub_epi32(_mm256_broadcastsi128_si256(r_Q10), q_Q10);
        rd_Q10 = _mm256_abs_epi32(q_Q10);
        rr_Q10 = silk_mm256_smulbb_epi32(rr_Q10, rr_Q10);
        rd_Q10 = silk_mm256_smulbb_epi32(rd_Q10, _mm256_set1_epi32(Lambda_Q10));
        rd_Q10 = _mm256_add_epi32(rd_Q10, rr_Q10);
        rd_Q10 = _mm256_srai_epi32(rd_Q10, 10);

        mask = _mm256_broadcastsi128_si256(_mm_cmplt_epi32(_mm256_extracti128_si256(rd_Q10, 0), _mm256_extracti128_si256(rd_Q10, 1)));
        SS_RD_Q10 = _mm256_add_epi32(
            _mm256_broadcastsi128_si256(psDelDec->RD_Q10),
            _mm256_blendv_epi8(
                _mm256_permute2x128_si256(rd_Q10, rd_Q10, 0x1),
                rd_Q10,
                mask));
        SS_Q_Q10 = _mm256_blendv_epi8(
            _mm256_permute2x128_si256(q_Q10, q_Q10, 0x1),
            q_Q10,
            mask);

        /* Update states for best and second best quantization */

        /* Quantized excitation */
        exc_Q14 = silk_mm256_sign_epi32(_mm256_slli_epi32(SS_Q_Q10, 4), _mm256_broadcastsi128_si256(psDelDec->Seed));

        /* Add predictions */
        exc_Q14 = _mm256_add_epi32(exc_Q14, _mm256_set1_epi32(LTP_pred_Q14));
        SS_LPC_exc_Q14 = _mm256_slli_epi32(exc_Q14, 1);
        SS_xq_Q14 = _mm256_add_epi32(exc_Q14, _mm256_broadcastsi128_si256(LPC_pred_Q14));

        /* Update states */
        SS_Diff_Q14 = _mm256_sub_epi32(SS_xq_Q14, _mm256_set1_epi32(silk_LSHIFT(x_Q10[i], 4)));
        SS_LF_AR_Q14 = _mm256_sub_epi32(SS_Diff_Q14, _mm256_broadcastsi128_si256(n_AR_Q14));
        SS_sLTP_shp_Q14 = silk_mm256_sub_sat_epi32(SS_LF_AR_Q14, _mm256_broadcastsi128_si256(n_LF_Q14));

        /* END Updating Delayed Decision States */

        *smpl_buf_idx = (*smpl_buf_idx + DECISION_DELAY - 1) % DECISION_DELAY;
        last_smple_idx = (*smpl_buf_idx + decisionDelay) % DECISION_DELAY;
        psLastSample = &psDelDec->Samples[last_smple_idx];

        /* Find winner */
        RDmin_Q10 = silk_mm_mask_hmin_epi32(_mm256_castsi256_si128(SS_RD_Q10), MaskDelDec);
        Winner_selector = silk_index_to_selector(silk_index_of_first_equal_epi32(RDmin_Q10, _mm256_castsi256_si128(SS_RD_Q10)));

        /* Increase RD values of expired states */
        Winner_rand_state = _mm_shuffle_epi8(psLastSample->RandState, Winner_selector);

        SS_RD_Q10 = _mm256_blendv_epi8(
            _mm256_add_epi32(SS_RD_Q10, _mm256_set1_epi32(silk_int32_MAX >> 4)),
            SS_RD_Q10,
            _mm256_broadcastsi128_si256(_mm_cmpeq_epi32(psLastSample->RandState, Winner_rand_state)));

        /* find worst in first set */
        RDmax_Q10 = silk_mm_mask_hmax_epi32(_mm256_extracti128_si256(SS_RD_Q10, 0), MaskDelDec);
        /* find best in second set */
        RDmin_Q10 = silk_mm_mask_hmin_epi32(_mm256_extracti128_si256(SS_RD_Q10, 1), MaskDelDec);

        /* Replace a state if best from second set outperforms worst in first set */
        tmp0 = _mm_cmplt_epi32(RDmin_Q10, RDmax_Q10);
        if (!_mm_test_all_zeros(tmp0, tmp0))
        {
            int t;
            RDmax_ind = silk_index_of_first_equal_epi32(RDmax_Q10, _mm256_extracti128_si256(SS_RD_Q10, 0));
            RDmin_ind = silk_index_of_first_equal_epi32(RDmin_Q10, _mm256_extracti128_si256(SS_RD_Q10, 1));
            tmp1 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(0xFFU << (unsigned)(RDmax_ind << 3)));
            tmp0 = _mm_blendv_epi8(
                _mm_set_epi8(0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0),
                silk_index_to_selector(RDmin_ind),
                tmp1);
            for (t = i; t < MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH; t++)
            {
                psDelDec->sLPC_Q14[t] = _mm_shuffle_epi8(psDelDec->sLPC_Q14[t], tmp0);
            }
            psDelDec->Seed = _mm_shuffle_epi8(psDelDec->Seed, tmp0);
            psDelDec->SeedInit = _mm_shuffle_epi8(psDelDec->SeedInit, tmp0);
            for (t = 0; t < MAX_SHAPE_LPC_ORDER; t++)
            {
                psDelDec->sAR2_Q14[t] = _mm_shuffle_epi8(psDelDec->sAR2_Q14[t], tmp0);
            }
            for (t = 0; t < DECISION_DELAY; t++)
            {
                psDelDec->Samples[t].RandState = _mm_shuffle_epi8(psDelDec->Samples[t].RandState, tmp0);
                psDelDec->Samples[t].Q_Q10 = _mm_shuffle_epi8(psDelDec->Samples[t].Q_Q10, tmp0);
                psDelDec->Samples[t].Xq_Q14 = _mm_shuffle_epi8(psDelDec->Samples[t].Xq_Q14, tmp0);
                psDelDec->Samples[t].Pred_Q15 = _mm_shuffle_epi8(psDelDec->Samples[t].Pred_Q15, tmp0);
                psDelDec->Samples[t].Shape_Q14 = _mm_shuffle_epi8(psDelDec->Samples[t].Shape_Q14, tmp0);
            }
            mask = _mm256_castsi128_si256(_mm_blendv_epi8(_mm_set_epi32(0x3, 0x2, 0x1, 0x0), _mm_set1_epi32(RDmin_ind + 4), tmp1));
            SS_Q_Q10 = _mm256_permutevar8x32_epi32(SS_Q_Q10, mask);
            SS_RD_Q10 = _mm256_permutevar8x32_epi32(SS_RD_Q10, mask);
            SS_xq_Q14 = _mm256_permutevar8x32_epi32(SS_xq_Q14, mask);
            SS_LF_AR_Q14 = _mm256_permutevar8x32_epi32(SS_LF_AR_Q14, mask);
            SS_Diff_Q14 = _mm256_permutevar8x32_epi32(SS_Diff_Q14, mask);
            SS_sLTP_shp_Q14 = _mm256_permutevar8x32_epi32(SS_sLTP_shp_Q14, mask);
            SS_LPC_exc_Q14 = _mm256_permutevar8x32_epi32(SS_LPC_exc_Q14, mask);
        }

        /* Write samples from winner to output and long-term filter states */
        if (subfr > 0 || i >= decisionDelay)
        {
            pulses[i - decisionDelay] =
                (opus_int8)silk_sar_round_32(silk_select_winner(psLastSample->Q_Q10, Winner_selector), 10);
            xq[i - decisionDelay] =
                silk_sat16((opus_int32)silk_sar_round_smulww(silk_select_winner(psLastSample->Xq_Q14, Winner_selector), delayedGain_Q10[last_smple_idx], 8));
            NSQ->sLTP_shp_Q14[NSQ->sLTP_shp_buf_idx - decisionDelay] =
                silk_select_winner(psLastSample->Shape_Q14, Winner_selector);
            sLTP_Q15[NSQ->sLTP_buf_idx - decisionDelay] =
                silk_select_winner(psLastSample->Pred_Q15, Winner_selector);
        }
        NSQ->sLTP_shp_buf_idx++;
        NSQ->sLTP_buf_idx++;

        /* Update states */
        psSample = &psDelDec->Samples[*smpl_buf_idx];
        psDelDec->Seed = _mm_add_epi32(psDelDec->Seed, silk_mm_srai_round_epi32(_mm256_castsi256_si128(SS_Q_Q10), 10));
        psDelDec->LF_AR_Q14 = _mm256_castsi256_si128(SS_LF_AR_Q14);
        psDelDec->Diff_Q14 = _mm256_castsi256_si128(SS_Diff_Q14);
        psDelDec->sLPC_Q14[i + NSQ_LPC_BUF_LENGTH] = _mm256_castsi256_si128(SS_xq_Q14);
        psDelDec->RD_Q10 = _mm256_castsi256_si128(SS_RD_Q10);
        psSample->Xq_Q14 = _mm256_castsi256_si128(SS_xq_Q14);
        psSample->Q_Q10 = _mm256_castsi256_si128(SS_Q_Q10);
        psSample->Pred_Q15 = _mm256_castsi256_si128(SS_LPC_exc_Q14);
        psSample->Shape_Q14 = _mm256_castsi256_si128(SS_sLTP_shp_Q14);
        psSample->RandState = psDelDec->Seed;
        delayedGain_Q10[*smpl_buf_idx] = Gain_Q10;
    }
    /* Update LPC states */
    for (i = 0; i < NSQ_LPC_BUF_LENGTH; i++)
    {
        psDelDec->sLPC_Q14[i] = (&psDelDec->sLPC_Q14[length])[i];
    }
}

static OPUS_INLINE void silk_nsq_del_dec_scale_states_avx2(
    const silk_encoder_state *psEncC,          /* I    Encoder State                   */
    silk_nsq_state *NSQ,                       /* I/O  NSQ state                       */
    NSQ_del_dec_struct *psDelDec,              /* I/O  Delayed decision states         */
    const opus_int16 x16[],                    /* I    Input                           */
    opus_int32 x_sc_Q10[MAX_SUB_FRAME_LENGTH], /* O    Input scaled with 1/Gain in Q10 */
    const opus_int16 sLTP[],                   /* I    Re-whitened LTP state in Q0     */
    opus_int32 sLTP_Q15[],                     /* O    LTP state matching scaled input */
    opus_int subfr,                            /* I    Subframe number                 */
    const opus_int LTP_scale_Q14,              /* I    LTP state scaling               */
    const opus_int32 Gains_Q16[MAX_NB_SUBFR],  /* I                                    */
    const opus_int pitchL[MAX_NB_SUBFR],       /* I    Pitch lag                       */
    const opus_int signal_type,                /* I    Signal type                     */
    const opus_int decisionDelay               /* I    Decision delay                  */
)
{
    int i;
    opus_int lag;
    opus_int32 gain_adj_Q16, inv_gain_Q31, inv_gain_Q26;
    NSQ_del_dec_sample_struct *psSample;

    lag = pitchL[subfr];
    inv_gain_Q31 = silk_INVERSE32_varQ(silk_max(Gains_Q16[subfr], 1), 47);
    silk_assert(inv_gain_Q31 != 0);

    /* Scale input */
    inv_gain_Q26 = silk_sar_round_32(inv_gain_Q31, 5);
    for (i = 0; i < psEncC->subfr_length; i+=4)
    {
        __m256i x = _mm256_cvtepi16_epi64(_mm_loadu_si64(&x16[i]));
        x = _mm256_slli_epi64(_mm256_mul_epi32(x, _mm256_set1_epi32(inv_gain_Q26)), 16);
        _mm_storeu_si128((__m128i*)&x_sc_Q10[i], silk_cvtepi64_epi32_high(x));
    }

    /* After rewhitening the LTP state is un-scaled, so scale with inv_gain_Q16 */
    if (NSQ->rewhite_flag)
    {
        if (subfr == 0)
        {
            /* Do LTP downscaling */
            inv_gain_Q31 = silk_LSHIFT(silk_SMULWB(inv_gain_Q31, LTP_scale_Q14), 2);
        }
        for (i = NSQ->sLTP_buf_idx - lag - LTP_ORDER / 2; i < NSQ->sLTP_buf_idx; i++)
        {
            silk_assert(i < MAX_FRAME_LENGTH);
            sLTP_Q15[i] = silk_SMULWB(inv_gain_Q31, sLTP[i]);
        }
    }

    /* Adjust for changing gain */
    if (Gains_Q16[subfr] != NSQ->prev_gain_Q16)
    {
        gain_adj_Q16 = silk_DIV32_varQ(NSQ->prev_gain_Q16, Gains_Q16[subfr], 16);

        /* Scale long-term shaping state */
        for (i = NSQ->sLTP_shp_buf_idx - psEncC->ltp_mem_length; i < NSQ->sLTP_shp_buf_idx; i+=4)
        {
	    opus_int32 *p = &NSQ->sLTP_shp_Q14[i];
            _mm_storeu_si128((__m128i*)p, silk_mm_smulww_epi32(_mm_loadu_si128((__m128i*)p), gain_adj_Q16));
        }

        /* Scale long-term prediction state */
        if (signal_type == TYPE_VOICED && NSQ->rewhite_flag == 0)
        {
            for (i = NSQ->sLTP_buf_idx - lag - LTP_ORDER / 2; i < NSQ->sLTP_buf_idx - decisionDelay; i++)
            {
                sLTP_Q15[i] = ((opus_int64)sLTP_Q15[i]) * ((opus_int64)gain_adj_Q16) >> 16;
            }
        }

        /* Scale scalar states */
        psDelDec->LF_AR_Q14 = silk_mm_smulww_epi32(psDelDec->LF_AR_Q14, gain_adj_Q16);
        psDelDec->Diff_Q14 = silk_mm_smulww_epi32(psDelDec->Diff_Q14, gain_adj_Q16);

        /* Scale short-term prediction and shaping states */
        for (i = 0; i < NSQ_LPC_BUF_LENGTH; i++)
        {
            psDelDec->sLPC_Q14[i] = silk_mm_smulww_epi32(psDelDec->sLPC_Q14[i], gain_adj_Q16);
        }
        for (i = 0; i < DECISION_DELAY; i++)
        {
            psSample = &psDelDec->Samples[i];
            psSample->Pred_Q15 = silk_mm_smulww_epi32(psSample->Pred_Q15, gain_adj_Q16);
            psSample->Shape_Q14 = silk_mm_smulww_epi32(psSample->Shape_Q14, gain_adj_Q16);
        }
        for (i = 0; i < MAX_SHAPE_LPC_ORDER; i++)
        {
            psDelDec->sAR2_Q14[i] = silk_mm_smulww_epi32(psDelDec->sAR2_Q14[i], gain_adj_Q16);
        }

        /* Save inverse gain */
        NSQ->prev_gain_Q16 = Gains_Q16[subfr];
    }
}

static OPUS_INLINE void silk_LPC_analysis_filter_avx2(
    opus_int16                  *out,               /* O    Output signal                           */
    const opus_int16            *in,                /* I    Input signal                            */
    const opus_int16            *B,                 /* I    MA prediction coefficients, Q12 [order] */
    const opus_int32            len,                /* I    Signal length                           */
    const opus_int32            order               /* I    Filter order                            */
)
{
    int i;
    opus_int32       out32_Q12, out32;
    silk_assert(order == 10 || order == 16);

    for(i = order; i < len; i++ )
    {
        const opus_int16 *in_ptr = &in[ i ];
        /* Allowing wrap around so that two wraps can cancel each other. The rare
           cases where the result wraps around can only be triggered by invalid streams*/

        __m256i in_v = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)&in_ptr[-8]));
        __m256i B_v  = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)&      B[0]));
        __m256i sum = _mm256_mullo_epi32(in_v, silk_mm256_reverse_epi32(B_v));
        if (order > 10)
        {
            in_v = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)&in_ptr[-16]));
            B_v  = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)&B       [8]));
            B_v  = silk_mm256_reverse_epi32(B_v);
        }
        else
        {
            in_v = _mm256_cvtepi16_epi32(_mm_loadu_si32(&in_ptr[-10]));
            B_v  = _mm256_cvtepi16_epi32(_mm_loadu_si32(&B       [8]));
            B_v  = _mm256_shuffle_epi32(B_v, 0x01);
        }
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(in_v, B_v));

        out32_Q12 = silk_mm256_hsum_epi32(sum);

        /* Subtract prediction */
        out32_Q12 = silk_SUB32_ovflw( silk_LSHIFT( (opus_int32)*in_ptr, 12 ), out32_Q12 );

        /* Scale to Q0 */
        out32 = silk_sar_round_32(out32_Q12, 12);

        /* Saturate output */
        out[ i ] = silk_sat16(out32);
    }

    /* Set first d output samples to zero */
    silk_memset( out, 0, order * sizeof( opus_int16 ) );
}
