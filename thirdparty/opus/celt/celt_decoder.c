/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2010 Xiph.Org Foundation
   Copyright (c) 2008 Gregory Maxwell
   Written by Jean-Marc Valin and Gregory Maxwell */
/*
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

#define CELT_DECODER_C

#include "cpu_support.h"
#include "os_support.h"
#include "mdct.h"
#include <math.h>
#include "celt.h"
#include "pitch.h"
#include "bands.h"
#include "modes.h"
#include "entcode.h"
#include "quant_bands.h"
#include "rate.h"
#include "stack_alloc.h"
#include "mathops.h"
#include "float_cast.h"
#include <stdarg.h>
#include "celt_lpc.h"
#include "vq.h"

#ifdef ENABLE_DEEP_PLC
#include "lpcnet.h"
#include "lpcnet_private.h"
#endif

/* The maximum pitch lag to allow in the pitch-based PLC. It's possible to save
   CPU time in the PLC pitch search by making this smaller than MAX_PERIOD. The
   current value corresponds to a pitch of 66.67 Hz. */
#define PLC_PITCH_LAG_MAX (720)
/* The minimum pitch lag to allow in the pitch-based PLC. This corresponds to a
   pitch of 480 Hz. */
#define PLC_PITCH_LAG_MIN (100)

/**********************************************************************/
/*                                                                    */
/*                             DECODER                                */
/*                                                                    */
/**********************************************************************/
#define DECODE_BUFFER_SIZE 2048

#define PLC_UPDATE_FRAMES 4
#define PLC_UPDATE_SAMPLES (PLC_UPDATE_FRAMES*FRAME_SIZE)

/** Decoder state
 @brief Decoder state
 */
struct OpusCustomDecoder {
   const OpusCustomMode *mode;
   int overlap;
   int channels;
   int stream_channels;

   int downsample;
   int start, end;
   int signalling;
   int disable_inv;
   int complexity;
   int arch;

   /* Everything beyond this point gets cleared on a reset */
#define DECODER_RESET_START rng

   opus_uint32 rng;
   int error;
   int last_pitch_index;
   int loss_duration;
   int skip_plc;
   int postfilter_period;
   int postfilter_period_old;
   opus_val16 postfilter_gain;
   opus_val16 postfilter_gain_old;
   int postfilter_tapset;
   int postfilter_tapset_old;
   int prefilter_and_fold;

   celt_sig preemph_memD[2];

#ifdef ENABLE_DEEP_PLC
   opus_int16 plc_pcm[PLC_UPDATE_SAMPLES];
   int plc_fill;
   float plc_preemphasis_mem;
#endif

   celt_sig _decode_mem[1]; /* Size = channels*(DECODE_BUFFER_SIZE+mode->overlap) */
   /* opus_val16 lpc[],  Size = channels*CELT_LPC_ORDER */
   /* opus_val16 oldEBands[], Size = 2*mode->nbEBands */
   /* opus_val16 oldLogE[], Size = 2*mode->nbEBands */
   /* opus_val16 oldLogE2[], Size = 2*mode->nbEBands */
   /* opus_val16 backgroundLogE[], Size = 2*mode->nbEBands */
};

#if defined(ENABLE_HARDENING) || defined(ENABLE_ASSERTIONS)
/* Make basic checks on the CELT state to ensure we don't end
   up writing all over memory. */
void validate_celt_decoder(CELTDecoder *st)
{
#ifndef CUSTOM_MODES
   celt_assert(st->mode == opus_custom_mode_create(48000, 960, NULL));
   celt_assert(st->overlap == 120);
   celt_assert(st->end <= 21);
#else
/* From Section 4.3 in the spec: "The normal CELT layer uses 21 of those bands,
   though Opus Custom (see Section 6.2) may use a different number of bands"

   Check if it's within the maximum number of Bark frequency bands instead */
   celt_assert(st->end <= 25);
#endif
   celt_assert(st->channels == 1 || st->channels == 2);
   celt_assert(st->stream_channels == 1 || st->stream_channels == 2);
   celt_assert(st->downsample > 0);
   celt_assert(st->start == 0 || st->start == 17);
   celt_assert(st->start < st->end);
#ifdef OPUS_ARCHMASK
   celt_assert(st->arch >= 0);
   celt_assert(st->arch <= OPUS_ARCHMASK);
#endif
   celt_assert(st->last_pitch_index <= PLC_PITCH_LAG_MAX);
   celt_assert(st->last_pitch_index >= PLC_PITCH_LAG_MIN || st->last_pitch_index == 0);
   celt_assert(st->postfilter_period < MAX_PERIOD);
   celt_assert(st->postfilter_period >= COMBFILTER_MINPERIOD || st->postfilter_period == 0);
   celt_assert(st->postfilter_period_old < MAX_PERIOD);
   celt_assert(st->postfilter_period_old >= COMBFILTER_MINPERIOD || st->postfilter_period_old == 0);
   celt_assert(st->postfilter_tapset <= 2);
   celt_assert(st->postfilter_tapset >= 0);
   celt_assert(st->postfilter_tapset_old <= 2);
   celt_assert(st->postfilter_tapset_old >= 0);
}
#endif

int celt_decoder_get_size(int channels)
{
   const CELTMode *mode = opus_custom_mode_create(48000, 960, NULL);
   return opus_custom_decoder_get_size(mode, channels);
}

OPUS_CUSTOM_NOSTATIC int opus_custom_decoder_get_size(const CELTMode *mode, int channels)
{
   int size = sizeof(struct CELTDecoder)
            + (channels*(DECODE_BUFFER_SIZE+mode->overlap)-1)*sizeof(celt_sig)
            + channels*CELT_LPC_ORDER*sizeof(opus_val16)
            + 4*2*mode->nbEBands*sizeof(opus_val16);
   return size;
}

#ifdef CUSTOM_MODES
CELTDecoder *opus_custom_decoder_create(const CELTMode *mode, int channels, int *error)
{
   int ret;
   CELTDecoder *st = (CELTDecoder *)opus_alloc(opus_custom_decoder_get_size(mode, channels));
   ret = opus_custom_decoder_init(st, mode, channels);
   if (ret != OPUS_OK)
   {
      opus_custom_decoder_destroy(st);
      st = NULL;
   }
   if (error)
      *error = ret;
   return st;
}
#endif /* CUSTOM_MODES */

int celt_decoder_init(CELTDecoder *st, opus_int32 sampling_rate, int channels)
{
   int ret;
   ret = opus_custom_decoder_init(st, opus_custom_mode_create(48000, 960, NULL), channels);
   if (ret != OPUS_OK)
      return ret;
   st->downsample = resampling_factor(sampling_rate);
   if (st->downsample==0)
      return OPUS_BAD_ARG;
   else
      return OPUS_OK;
}

OPUS_CUSTOM_NOSTATIC int opus_custom_decoder_init(CELTDecoder *st, const CELTMode *mode, int channels)
{
   if (channels < 0 || channels > 2)
      return OPUS_BAD_ARG;

   if (st==NULL)
      return OPUS_ALLOC_FAIL;

   OPUS_CLEAR((char*)st, opus_custom_decoder_get_size(mode, channels));

   st->mode = mode;
   st->overlap = mode->overlap;
   st->stream_channels = st->channels = channels;

   st->downsample = 1;
   st->start = 0;
   st->end = st->mode->effEBands;
   st->signalling = 1;
#ifndef DISABLE_UPDATE_DRAFT
   st->disable_inv = channels == 1;
#else
   st->disable_inv = 0;
#endif
   st->arch = opus_select_arch();

   opus_custom_decoder_ctl(st, OPUS_RESET_STATE);

   return OPUS_OK;
}

#ifdef CUSTOM_MODES
void opus_custom_decoder_destroy(CELTDecoder *st)
{
   opus_free(st);
}
#endif /* CUSTOM_MODES */

#ifndef CUSTOM_MODES
/* Special case for stereo with no downsampling and no accumulation. This is
   quite common and we can make it faster by processing both channels in the
   same loop, reducing overhead due to the dependency loop in the IIR filter. */
static void deemphasis_stereo_simple(celt_sig *in[], opus_val16 *pcm, int N, const opus_val16 coef0,
      celt_sig *mem)
{
   celt_sig * OPUS_RESTRICT x0;
   celt_sig * OPUS_RESTRICT x1;
   celt_sig m0, m1;
   int j;
   x0=in[0];
   x1=in[1];
   m0 = mem[0];
   m1 = mem[1];
   for (j=0;j<N;j++)
   {
      celt_sig tmp0, tmp1;
      /* Add VERY_SMALL to x[] first to reduce dependency chain. */
      tmp0 = x0[j] + VERY_SMALL + m0;
      tmp1 = x1[j] + VERY_SMALL + m1;
      m0 = MULT16_32_Q15(coef0, tmp0);
      m1 = MULT16_32_Q15(coef0, tmp1);
      pcm[2*j  ] = SCALEOUT(SIG2WORD16(tmp0));
      pcm[2*j+1] = SCALEOUT(SIG2WORD16(tmp1));
   }
   mem[0] = m0;
   mem[1] = m1;
}
#endif

#ifndef RESYNTH
static
#endif
void deemphasis(celt_sig *in[], opus_val16 *pcm, int N, int C, int downsample, const opus_val16 *coef,
      celt_sig *mem, int accum)
{
   int c;
   int Nd;
   int apply_downsampling=0;
   opus_val16 coef0;
   VARDECL(celt_sig, scratch);
   SAVE_STACK;
#ifndef CUSTOM_MODES
   /* Short version for common case. */
   if (downsample == 1 && C == 2 && !accum)
   {
      deemphasis_stereo_simple(in, pcm, N, coef[0], mem);
      return;
   }
#endif
#ifndef FIXED_POINT
   (void)accum;
   celt_assert(accum==0);
#endif
   ALLOC(scratch, N, celt_sig);
   coef0 = coef[0];
   Nd = N/downsample;
   c=0; do {
      int j;
      celt_sig * OPUS_RESTRICT x;
      opus_val16  * OPUS_RESTRICT y;
      celt_sig m = mem[c];
      x =in[c];
      y = pcm+c;
#ifdef CUSTOM_MODES
      if (coef[1] != 0)
      {
         opus_val16 coef1 = coef[1];
         opus_val16 coef3 = coef[3];
         for (j=0;j<N;j++)
         {
            celt_sig tmp = x[j] + m + VERY_SMALL;
            m = MULT16_32_Q15(coef0, tmp)
                          - MULT16_32_Q15(coef1, x[j]);
            tmp = SHL32(MULT16_32_Q15(coef3, tmp), 2);
            scratch[j] = tmp;
         }
         apply_downsampling=1;
      } else
#endif
      if (downsample>1)
      {
         /* Shortcut for the standard (non-custom modes) case */
         for (j=0;j<N;j++)
         {
            celt_sig tmp = x[j] + VERY_SMALL + m;
            m = MULT16_32_Q15(coef0, tmp);
            scratch[j] = tmp;
         }
         apply_downsampling=1;
      } else {
         /* Shortcut for the standard (non-custom modes) case */
#ifdef FIXED_POINT
         if (accum)
         {
            for (j=0;j<N;j++)
            {
               celt_sig tmp = x[j] + m + VERY_SMALL;
               m = MULT16_32_Q15(coef0, tmp);
               y[j*C] = SAT16(ADD32(y[j*C], SCALEOUT(SIG2WORD16(tmp))));
            }
         } else
#endif
         {
            for (j=0;j<N;j++)
            {
               celt_sig tmp = x[j] + VERY_SMALL + m;
               m = MULT16_32_Q15(coef0, tmp);
               y[j*C] = SCALEOUT(SIG2WORD16(tmp));
            }
         }
      }
      mem[c] = m;

      if (apply_downsampling)
      {
         /* Perform down-sampling */
#ifdef FIXED_POINT
         if (accum)
         {
            for (j=0;j<Nd;j++)
               y[j*C] = SAT16(ADD32(y[j*C], SCALEOUT(SIG2WORD16(scratch[j*downsample]))));
         } else
#endif
         {
            for (j=0;j<Nd;j++)
               y[j*C] = SCALEOUT(SIG2WORD16(scratch[j*downsample]));
         }
      }
   } while (++c<C);
   RESTORE_STACK;
}

#ifndef RESYNTH
static
#endif
void celt_synthesis(const CELTMode *mode, celt_norm *X, celt_sig * out_syn[],
                    opus_val16 *oldBandE, int start, int effEnd, int C, int CC,
                    int isTransient, int LM, int downsample,
                    int silence, int arch)
{
   int c, i;
   int M;
   int b;
   int B;
   int N, NB;
   int shift;
   int nbEBands;
   int overlap;
   VARDECL(celt_sig, freq);
   SAVE_STACK;

   overlap = mode->overlap;
   nbEBands = mode->nbEBands;
   N = mode->shortMdctSize<<LM;
   ALLOC(freq, N, celt_sig); /**< Interleaved signal MDCTs */
   M = 1<<LM;

   if (isTransient)
   {
      B = M;
      NB = mode->shortMdctSize;
      shift = mode->maxLM;
   } else {
      B = 1;
      NB = mode->shortMdctSize<<LM;
      shift = mode->maxLM-LM;
   }

   if (CC==2&&C==1)
   {
      /* Copying a mono streams to two channels */
      celt_sig *freq2;
      denormalise_bands(mode, X, freq, oldBandE, start, effEnd, M,
            downsample, silence);
      /* Store a temporary copy in the output buffer because the IMDCT destroys its input. */
      freq2 = out_syn[1]+overlap/2;
      OPUS_COPY(freq2, freq, N);
      for (b=0;b<B;b++)
         clt_mdct_backward(&mode->mdct, &freq2[b], out_syn[0]+NB*b, mode->window, overlap, shift, B, arch);
      for (b=0;b<B;b++)
         clt_mdct_backward(&mode->mdct, &freq[b], out_syn[1]+NB*b, mode->window, overlap, shift, B, arch);
   } else if (CC==1&&C==2)
   {
      /* Downmixing a stereo stream to mono */
      celt_sig *freq2;
      freq2 = out_syn[0]+overlap/2;
      denormalise_bands(mode, X, freq, oldBandE, start, effEnd, M,
            downsample, silence);
      /* Use the output buffer as temp array before downmixing. */
      denormalise_bands(mode, X+N, freq2, oldBandE+nbEBands, start, effEnd, M,
            downsample, silence);
      for (i=0;i<N;i++)
         freq[i] = ADD32(HALF32(freq[i]), HALF32(freq2[i]));
      for (b=0;b<B;b++)
         clt_mdct_backward(&mode->mdct, &freq[b], out_syn[0]+NB*b, mode->window, overlap, shift, B, arch);
   } else {
      /* Normal case (mono or stereo) */
      c=0; do {
         denormalise_bands(mode, X+c*N, freq, oldBandE+c*nbEBands, start, effEnd, M,
               downsample, silence);
         for (b=0;b<B;b++)
            clt_mdct_backward(&mode->mdct, &freq[b], out_syn[c]+NB*b, mode->window, overlap, shift, B, arch);
      } while (++c<CC);
   }
   /* Saturate IMDCT output so that we can't overflow in the pitch postfilter
      or in the */
   c=0; do {
      for (i=0;i<N;i++)
         out_syn[c][i] = SATURATE(out_syn[c][i], SIG_SAT);
   } while (++c<CC);
   RESTORE_STACK;
}

static void tf_decode(int start, int end, int isTransient, int *tf_res, int LM, ec_dec *dec)
{
   int i, curr, tf_select;
   int tf_select_rsv;
   int tf_changed;
   int logp;
   opus_uint32 budget;
   opus_uint32 tell;

   budget = dec->storage*8;
   tell = ec_tell(dec);
   logp = isTransient ? 2 : 4;
   tf_select_rsv = LM>0 && tell+logp+1<=budget;
   budget -= tf_select_rsv;
   tf_changed = curr = 0;
   for (i=start;i<end;i++)
   {
      if (tell+logp<=budget)
      {
         curr ^= ec_dec_bit_logp(dec, logp);
         tell = ec_tell(dec);
         tf_changed |= curr;
      }
      tf_res[i] = curr;
      logp = isTransient ? 4 : 5;
   }
   tf_select = 0;
   if (tf_select_rsv &&
     tf_select_table[LM][4*isTransient+0+tf_changed] !=
     tf_select_table[LM][4*isTransient+2+tf_changed])
   {
      tf_select = ec_dec_bit_logp(dec, 1);
   }
   for (i=start;i<end;i++)
   {
      tf_res[i] = tf_select_table[LM][4*isTransient+2*tf_select+tf_res[i]];
   }
}

static int celt_plc_pitch_search(celt_sig *decode_mem[2], int C, int arch)
{
   int pitch_index;
   VARDECL( opus_val16, lp_pitch_buf );
   SAVE_STACK;
   ALLOC( lp_pitch_buf, DECODE_BUFFER_SIZE>>1, opus_val16 );
   pitch_downsample(decode_mem, lp_pitch_buf,
         DECODE_BUFFER_SIZE, C, arch);
   pitch_search(lp_pitch_buf+(PLC_PITCH_LAG_MAX>>1), lp_pitch_buf,
         DECODE_BUFFER_SIZE-PLC_PITCH_LAG_MAX,
         PLC_PITCH_LAG_MAX-PLC_PITCH_LAG_MIN, &pitch_index, arch);
   pitch_index = PLC_PITCH_LAG_MAX-pitch_index;
   RESTORE_STACK;
   return pitch_index;
}

static void prefilter_and_fold(CELTDecoder * OPUS_RESTRICT st, int N)
{
   int c;
   int CC;
   int i;
   int overlap;
   celt_sig *decode_mem[2];
   const OpusCustomMode *mode;
   VARDECL(opus_val32, etmp);
   mode = st->mode;
   overlap = st->overlap;
   CC = st->channels;
   ALLOC(etmp, overlap, opus_val32);
   c=0; do {
      decode_mem[c] = st->_decode_mem + c*(DECODE_BUFFER_SIZE+overlap);
   } while (++c<CC);

   c=0; do {
      /* Apply the pre-filter to the MDCT overlap for the next frame because
         the post-filter will be re-applied in the decoder after the MDCT
         overlap. */
      comb_filter(etmp, decode_mem[c]+DECODE_BUFFER_SIZE-N,
         st->postfilter_period_old, st->postfilter_period, overlap,
         -st->postfilter_gain_old, -st->postfilter_gain,
         st->postfilter_tapset_old, st->postfilter_tapset, NULL, 0, st->arch);

      /* Simulate TDAC on the concealed audio so that it blends with the
         MDCT of the next frame. */
      for (i=0;i<overlap/2;i++)
      {
         decode_mem[c][DECODE_BUFFER_SIZE-N+i] =
            MULT16_32_Q15(mode->window[i], etmp[overlap-1-i])
            + MULT16_32_Q15(mode->window[overlap-i-1], etmp[i]);
      }
   } while (++c<CC);
}

#ifdef ENABLE_DEEP_PLC

#define SINC_ORDER 48
/* h=cos(pi/2*abs(sin([-24:24]/48*pi*23./24)).^2);
   b=sinc([-24:24]/3*1.02).*h;
   b=b/sum(b); */
static const float sinc_filter[SINC_ORDER+1] = {
    4.2931e-05f, -0.000190293f, -0.000816132f, -0.000637162f, 0.00141662f, 0.00354764f, 0.00184368f, -0.00428274f,
    -0.00856105f, -0.0034003f, 0.00930201f, 0.0159616f, 0.00489785f, -0.0169649f, -0.0259484f, -0.00596856f,
    0.0286551f, 0.0405872f, 0.00649994f, -0.0509284f, -0.0716655f, -0.00665212f,  0.134336f,  0.278927f,
    0.339995f,  0.278927f,  0.134336f, -0.00665212f, -0.0716655f, -0.0509284f, 0.00649994f, 0.0405872f,
    0.0286551f, -0.00596856f, -0.0259484f, -0.0169649f, 0.00489785f, 0.0159616f, 0.00930201f, -0.0034003f,
    -0.00856105f, -0.00428274f, 0.00184368f, 0.00354764f, 0.00141662f, -0.000637162f, -0.000816132f, -0.000190293f,
    4.2931e-05f
};

void update_plc_state(LPCNetPLCState *lpcnet, celt_sig *decode_mem[2], float *plc_preemphasis_mem, int CC)
{
   int i;
   int tmp_read_post, tmp_fec_skip;
   int offset;
   celt_sig buf48k[DECODE_BUFFER_SIZE];
   opus_int16 buf16k[PLC_UPDATE_SAMPLES];
   if (CC == 1) OPUS_COPY(buf48k, decode_mem[0], DECODE_BUFFER_SIZE);
   else {
      for (i=0;i<DECODE_BUFFER_SIZE;i++) {
         buf48k[i] = .5*(decode_mem[0][i] + decode_mem[1][i]);
      }
   }
   /* Down-sample the last 40 ms. */
   for (i=1;i<DECODE_BUFFER_SIZE;i++) buf48k[i] += PREEMPHASIS*buf48k[i-1];
   *plc_preemphasis_mem = buf48k[DECODE_BUFFER_SIZE-1];
   offset = DECODE_BUFFER_SIZE-SINC_ORDER-1 - 3*(PLC_UPDATE_SAMPLES-1);
   celt_assert(3*(PLC_UPDATE_SAMPLES-1) + SINC_ORDER + offset == DECODE_BUFFER_SIZE-1);
   for (i=0;i<PLC_UPDATE_SAMPLES;i++) {
      int j;
      float sum = 0;
      for (j=0;j<SINC_ORDER+1;j++) {
         sum += buf48k[3*i + j + offset]*sinc_filter[j];
      }
      buf16k[i] = float2int(MIN32(32767.f, MAX32(-32767.f, sum)));
   }
   tmp_read_post = lpcnet->fec_read_pos;
   tmp_fec_skip = lpcnet->fec_skip;
   for (i=0;i<PLC_UPDATE_FRAMES;i++) {
      lpcnet_plc_update(lpcnet, &buf16k[FRAME_SIZE*i]);
   }
   lpcnet->fec_read_pos = tmp_read_post;
   lpcnet->fec_skip = tmp_fec_skip;
}
#endif

static void celt_decode_lost(CELTDecoder * OPUS_RESTRICT st, int N, int LM
#ifdef ENABLE_DEEP_PLC
      ,LPCNetPLCState *lpcnet
#endif
      )
{
   int c;
   int i;
   const int C = st->channels;
   celt_sig *decode_mem[2];
   celt_sig *out_syn[2];
   opus_val16 *lpc;
   opus_val16 *oldBandE, *oldLogE, *oldLogE2, *backgroundLogE;
   const OpusCustomMode *mode;
   int nbEBands;
   int overlap;
   int start;
   int loss_duration;
   int noise_based;
   const opus_int16 *eBands;
   SAVE_STACK;

   mode = st->mode;
   nbEBands = mode->nbEBands;
   overlap = mode->overlap;
   eBands = mode->eBands;

   c=0; do {
      decode_mem[c] = st->_decode_mem + c*(DECODE_BUFFER_SIZE+overlap);
      out_syn[c] = decode_mem[c]+DECODE_BUFFER_SIZE-N;
   } while (++c<C);
   lpc = (opus_val16*)(st->_decode_mem+(DECODE_BUFFER_SIZE+overlap)*C);
   oldBandE = lpc+C*CELT_LPC_ORDER;
   oldLogE = oldBandE + 2*nbEBands;
   oldLogE2 = oldLogE + 2*nbEBands;
   backgroundLogE = oldLogE2  + 2*nbEBands;

   loss_duration = st->loss_duration;
   start = st->start;
#ifdef ENABLE_DEEP_PLC
   noise_based = start != 0 || (lpcnet->fec_fill_pos == 0 && (st->skip_plc || loss_duration >= 80));
#else
   noise_based = loss_duration >= 40 || start != 0 || st->skip_plc;
#endif
   if (noise_based)
   {
      /* Noise-based PLC/CNG */
      VARDECL(celt_norm, X);
      opus_uint32 seed;
      int end;
      int effEnd;
      opus_val16 decay;
      end = st->end;
      effEnd = IMAX(start, IMIN(end, mode->effEBands));

      ALLOC(X, C*N, celt_norm);   /**< Interleaved normalised MDCTs */
      c=0; do {
         OPUS_MOVE(decode_mem[c], decode_mem[c]+N,
               DECODE_BUFFER_SIZE-N+overlap);
      } while (++c<C);

      if (st->prefilter_and_fold) {
         prefilter_and_fold(st, N);
      }

      /* Energy decay */
      decay = loss_duration==0 ? QCONST16(1.5f, DB_SHIFT) : QCONST16(.5f, DB_SHIFT);
      c=0; do
      {
         for (i=start;i<end;i++)
            oldBandE[c*nbEBands+i] = MAX16(backgroundLogE[c*nbEBands+i], oldBandE[c*nbEBands+i] - decay);
      } while (++c<C);
      seed = st->rng;
      for (c=0;c<C;c++)
      {
         for (i=start;i<effEnd;i++)
         {
            int j;
            int boffs;
            int blen;
            boffs = N*c+(eBands[i]<<LM);
            blen = (eBands[i+1]-eBands[i])<<LM;
            for (j=0;j<blen;j++)
            {
               seed = celt_lcg_rand(seed);
               X[boffs+j] = (celt_norm)((opus_int32)seed>>20);
            }
            renormalise_vector(X+boffs, blen, Q15ONE, st->arch);
         }
      }
      st->rng = seed;

      celt_synthesis(mode, X, out_syn, oldBandE, start, effEnd, C, C, 0, LM, st->downsample, 0, st->arch);
      st->prefilter_and_fold = 0;
      /* Skip regular PLC until we get two consecutive packets. */
      st->skip_plc = 1;
   } else {
      int exc_length;
      /* Pitch-based PLC */
      const opus_val16 *window;
      opus_val16 *exc;
      opus_val16 fade = Q15ONE;
      int pitch_index;
      VARDECL(opus_val16, _exc);
      VARDECL(opus_val16, fir_tmp);

      if (loss_duration == 0)
      {
#ifdef ENABLE_DEEP_PLC
        if (lpcnet->loaded) update_plc_state(lpcnet, decode_mem, &st->plc_preemphasis_mem, C);
#endif
         st->last_pitch_index = pitch_index = celt_plc_pitch_search(decode_mem, C, st->arch);
      } else {
         pitch_index = st->last_pitch_index;
         fade = QCONST16(.8f,15);
      }

      /* We want the excitation for 2 pitch periods in order to look for a
         decaying signal, but we can't get more than MAX_PERIOD. */
      exc_length = IMIN(2*pitch_index, MAX_PERIOD);

      ALLOC(_exc, MAX_PERIOD+CELT_LPC_ORDER, opus_val16);
      ALLOC(fir_tmp, exc_length, opus_val16);
      exc = _exc+CELT_LPC_ORDER;
      window = mode->window;
      c=0; do {
         opus_val16 decay;
         opus_val16 attenuation;
         opus_val32 S1=0;
         celt_sig *buf;
         int extrapolation_offset;
         int extrapolation_len;
         int j;

         buf = decode_mem[c];
         for (i=0;i<MAX_PERIOD+CELT_LPC_ORDER;i++)
            exc[i-CELT_LPC_ORDER] = SROUND16(buf[DECODE_BUFFER_SIZE-MAX_PERIOD-CELT_LPC_ORDER+i], SIG_SHIFT);

         if (loss_duration == 0)
         {
            opus_val32 ac[CELT_LPC_ORDER+1];
            /* Compute LPC coefficients for the last MAX_PERIOD samples before
               the first loss so we can work in the excitation-filter domain. */
            _celt_autocorr(exc, ac, window, overlap,
                   CELT_LPC_ORDER, MAX_PERIOD, st->arch);
            /* Add a noise floor of -40 dB. */
#ifdef FIXED_POINT
            ac[0] += SHR32(ac[0],13);
#else
            ac[0] *= 1.0001f;
#endif
            /* Use lag windowing to stabilize the Levinson-Durbin recursion. */
            for (i=1;i<=CELT_LPC_ORDER;i++)
            {
               /*ac[i] *= exp(-.5*(2*M_PI*.002*i)*(2*M_PI*.002*i));*/
#ifdef FIXED_POINT
               ac[i] -= MULT16_32_Q15(2*i*i, ac[i]);
#else
               ac[i] -= ac[i]*(0.008f*0.008f)*i*i;
#endif
            }
            _celt_lpc(lpc+c*CELT_LPC_ORDER, ac, CELT_LPC_ORDER);
#ifdef FIXED_POINT
         /* For fixed-point, apply bandwidth expansion until we can guarantee that
            no overflow can happen in the IIR filter. This means:
            32768*sum(abs(filter)) < 2^31 */
         while (1) {
            opus_val16 tmp=Q15ONE;
            opus_val32 sum=QCONST16(1., SIG_SHIFT);
            for (i=0;i<CELT_LPC_ORDER;i++)
               sum += ABS16(lpc[c*CELT_LPC_ORDER+i]);
            if (sum < 65535) break;
            for (i=0;i<CELT_LPC_ORDER;i++)
            {
               tmp = MULT16_16_Q15(QCONST16(.99f,15), tmp);
               lpc[c*CELT_LPC_ORDER+i] = MULT16_16_Q15(lpc[c*CELT_LPC_ORDER+i], tmp);
            }
         }
#endif
         }
         /* Initialize the LPC history with the samples just before the start
            of the region for which we're computing the excitation. */
         {
            /* Compute the excitation for exc_length samples before the loss. We need the copy
               because celt_fir() cannot filter in-place. */
            celt_fir(exc+MAX_PERIOD-exc_length, lpc+c*CELT_LPC_ORDER,
                  fir_tmp, exc_length, CELT_LPC_ORDER, st->arch);
            OPUS_COPY(exc+MAX_PERIOD-exc_length, fir_tmp, exc_length);
         }

         /* Check if the waveform is decaying, and if so how fast.
            We do this to avoid adding energy when concealing in a segment
            with decaying energy. */
         {
            opus_val32 E1=1, E2=1;
            int decay_length;
#ifdef FIXED_POINT
            int shift = IMAX(0,2*celt_zlog2(celt_maxabs16(&exc[MAX_PERIOD-exc_length], exc_length))-20);
#endif
            decay_length = exc_length>>1;
            for (i=0;i<decay_length;i++)
            {
               opus_val16 e;
               e = exc[MAX_PERIOD-decay_length+i];
               E1 += SHR32(MULT16_16(e, e), shift);
               e = exc[MAX_PERIOD-2*decay_length+i];
               E2 += SHR32(MULT16_16(e, e), shift);
            }
            E1 = MIN32(E1, E2);
            decay = celt_sqrt(frac_div32(SHR32(E1, 1), E2));
         }

         /* Move the decoder memory one frame to the left to give us room to
            add the data for the new frame. We ignore the overlap that extends
            past the end of the buffer, because we aren't going to use it. */
         OPUS_MOVE(buf, buf+N, DECODE_BUFFER_SIZE-N);

         /* Extrapolate from the end of the excitation with a period of
            "pitch_index", scaling down each period by an additional factor of
            "decay". */
         extrapolation_offset = MAX_PERIOD-pitch_index;
         /* We need to extrapolate enough samples to cover a complete MDCT
            window (including overlap/2 samples on both sides). */
         extrapolation_len = N+overlap;
         /* We also apply fading if this is not the first loss. */
         attenuation = MULT16_16_Q15(fade, decay);
         for (i=j=0;i<extrapolation_len;i++,j++)
         {
            opus_val16 tmp;
            if (j >= pitch_index) {
               j -= pitch_index;
               attenuation = MULT16_16_Q15(attenuation, decay);
            }
            buf[DECODE_BUFFER_SIZE-N+i] =
                  SHL32(EXTEND32(MULT16_16_Q15(attenuation,
                        exc[extrapolation_offset+j])), SIG_SHIFT);
            /* Compute the energy of the previously decoded signal whose
               excitation we're copying. */
            tmp = SROUND16(
                  buf[DECODE_BUFFER_SIZE-MAX_PERIOD-N+extrapolation_offset+j],
                  SIG_SHIFT);
            S1 += SHR32(MULT16_16(tmp, tmp), 10);
         }
         {
            opus_val16 lpc_mem[CELT_LPC_ORDER];
            /* Copy the last decoded samples (prior to the overlap region) to
               synthesis filter memory so we can have a continuous signal. */
            for (i=0;i<CELT_LPC_ORDER;i++)
               lpc_mem[i] = SROUND16(buf[DECODE_BUFFER_SIZE-N-1-i], SIG_SHIFT);
            /* Apply the synthesis filter to convert the excitation back into
               the signal domain. */
            celt_iir(buf+DECODE_BUFFER_SIZE-N, lpc+c*CELT_LPC_ORDER,
                  buf+DECODE_BUFFER_SIZE-N, extrapolation_len, CELT_LPC_ORDER,
                  lpc_mem, st->arch);
#ifdef FIXED_POINT
            for (i=0; i < extrapolation_len; i++)
               buf[DECODE_BUFFER_SIZE-N+i] = SATURATE(buf[DECODE_BUFFER_SIZE-N+i], SIG_SAT);
#endif
         }

         /* Check if the synthesis energy is higher than expected, which can
            happen with the signal changes during our window. If so,
            attenuate. */
         {
            opus_val32 S2=0;
            for (i=0;i<extrapolation_len;i++)
            {
               opus_val16 tmp = SROUND16(buf[DECODE_BUFFER_SIZE-N+i], SIG_SHIFT);
               S2 += SHR32(MULT16_16(tmp, tmp), 10);
            }
            /* This checks for an "explosion" in the synthesis. */
#ifdef FIXED_POINT
            if (!(S1 > SHR32(S2,2)))
#else
            /* The float test is written this way to catch NaNs in the output
               of the IIR filter at the same time. */
            if (!(S1 > 0.2f*S2))
#endif
            {
               for (i=0;i<extrapolation_len;i++)
                  buf[DECODE_BUFFER_SIZE-N+i] = 0;
            } else if (S1 < S2)
            {
               opus_val16 ratio = celt_sqrt(frac_div32(SHR32(S1,1)+1,S2+1));
               for (i=0;i<overlap;i++)
               {
                  opus_val16 tmp_g = Q15ONE
                        - MULT16_16_Q15(window[i], Q15ONE-ratio);
                  buf[DECODE_BUFFER_SIZE-N+i] =
                        MULT16_32_Q15(tmp_g, buf[DECODE_BUFFER_SIZE-N+i]);
               }
               for (i=overlap;i<extrapolation_len;i++)
               {
                  buf[DECODE_BUFFER_SIZE-N+i] =
                        MULT16_32_Q15(ratio, buf[DECODE_BUFFER_SIZE-N+i]);
               }
            }
         }

      } while (++c<C);

#ifdef ENABLE_DEEP_PLC
      if (lpcnet->loaded && (st->complexity >= 5 || lpcnet->fec_fill_pos > 0)) {
         float overlap_mem;
         int samples_needed16k;
         celt_sig *buf;
         VARDECL(float, buf_copy);
         buf = decode_mem[0];
         ALLOC(buf_copy, C*overlap, float);
         c=0; do {
            OPUS_COPY(buf_copy+c*overlap, &decode_mem[c][DECODE_BUFFER_SIZE-N], overlap);
         } while (++c<C);

         /* Need enough samples from the PLC to cover the frame size, resampling delay,
            and the overlap at the end. */
         samples_needed16k = (N+SINC_ORDER+overlap)/3;
         if (loss_duration == 0) {
            st->plc_fill = 0;
         }
         while (st->plc_fill < samples_needed16k) {
            lpcnet_plc_conceal(lpcnet, &st->plc_pcm[st->plc_fill]);
            st->plc_fill += FRAME_SIZE;
         }
         /* Resample to 48 kHz. */
         for (i=0;i<(N+overlap)/3;i++) {
            int j;
            float sum;
            for (sum=0, j=0;j<17;j++) sum += 3*st->plc_pcm[i+j]*sinc_filter[3*j];
            buf[DECODE_BUFFER_SIZE-N+3*i] = sum;
            for (sum=0, j=0;j<16;j++) sum += 3*st->plc_pcm[i+j+1]*sinc_filter[3*j+2];
            buf[DECODE_BUFFER_SIZE-N+3*i+1] = sum;
            for (sum=0, j=0;j<16;j++) sum += 3*st->plc_pcm[i+j+1]*sinc_filter[3*j+1];
            buf[DECODE_BUFFER_SIZE-N+3*i+2] = sum;
         }
         OPUS_MOVE(st->plc_pcm, &st->plc_pcm[N/3], st->plc_fill-N/3);
         st->plc_fill -= N/3;
         for (i=0;i<N;i++) {
            float tmp = buf[DECODE_BUFFER_SIZE-N+i];
            buf[DECODE_BUFFER_SIZE-N+i] -= PREEMPHASIS*st->plc_preemphasis_mem;
            st->plc_preemphasis_mem = tmp;
         }
         overlap_mem = st->plc_preemphasis_mem;
         for (i=0;i<overlap;i++) {
            float tmp = buf[DECODE_BUFFER_SIZE+i];
            buf[DECODE_BUFFER_SIZE+i] -= PREEMPHASIS*overlap_mem;
            overlap_mem = tmp;
         }
         /* For now, we just do mono PLC. */
         if (C==2) OPUS_COPY(decode_mem[1], decode_mem[0], DECODE_BUFFER_SIZE+overlap);
         c=0; do {
            /* Cross-fade with 48-kHz non-neural PLC for the first 2.5 ms to avoid a discontinuity. */
            if (loss_duration == 0) {
               for (i=0;i<overlap;i++) decode_mem[c][DECODE_BUFFER_SIZE-N+i] = (1-window[i])*buf_copy[c*overlap+i] + (window[i])*decode_mem[c][DECODE_BUFFER_SIZE-N+i];
            }
         } while (++c<C);
      }
#endif
      st->prefilter_and_fold = 1;
   }

   /* Saturate to soemthing large to avoid wrap-around. */
   st->loss_duration = IMIN(10000, loss_duration+(1<<LM));

   RESTORE_STACK;
}

int celt_decode_with_ec_dred(CELTDecoder * OPUS_RESTRICT st, const unsigned char *data,
      int len, opus_val16 * OPUS_RESTRICT pcm, int frame_size, ec_dec *dec, int accum
#ifdef ENABLE_DEEP_PLC
      ,LPCNetPLCState *lpcnet
#endif
      )
{
   int c, i, N;
   int spread_decision;
   opus_int32 bits;
   ec_dec _dec;
   VARDECL(celt_norm, X);
   VARDECL(int, fine_quant);
   VARDECL(int, pulses);
   VARDECL(int, cap);
   VARDECL(int, offsets);
   VARDECL(int, fine_priority);
   VARDECL(int, tf_res);
   VARDECL(unsigned char, collapse_masks);
   celt_sig *decode_mem[2];
   celt_sig *out_syn[2];
   opus_val16 *lpc;
   opus_val16 *oldBandE, *oldLogE, *oldLogE2, *backgroundLogE;

   int shortBlocks;
   int isTransient;
   int intra_ener;
   const int CC = st->channels;
   int LM, M;
   int start;
   int end;
   int effEnd;
   int codedBands;
   int alloc_trim;
   int postfilter_pitch;
   opus_val16 postfilter_gain;
   int intensity=0;
   int dual_stereo=0;
   opus_int32 total_bits;
   opus_int32 balance;
   opus_int32 tell;
   int dynalloc_logp;
   int postfilter_tapset;
   int anti_collapse_rsv;
   int anti_collapse_on=0;
   int silence;
   int C = st->stream_channels;
   const OpusCustomMode *mode;
   int nbEBands;
   int overlap;
   const opus_int16 *eBands;
   opus_val16 max_background_increase;
   ALLOC_STACK;

   VALIDATE_CELT_DECODER(st);
   mode = st->mode;
   nbEBands = mode->nbEBands;
   overlap = mode->overlap;
   eBands = mode->eBands;
   start = st->start;
   end = st->end;
   frame_size *= st->downsample;

   lpc = (opus_val16*)(st->_decode_mem+(DECODE_BUFFER_SIZE+overlap)*CC);
   oldBandE = lpc+CC*CELT_LPC_ORDER;
   oldLogE = oldBandE + 2*nbEBands;
   oldLogE2 = oldLogE + 2*nbEBands;
   backgroundLogE = oldLogE2  + 2*nbEBands;

#ifdef CUSTOM_MODES
   if (st->signalling && data!=NULL)
   {
      int data0=data[0];
      /* Convert "standard mode" to Opus header */
      if (mode->Fs==48000 && mode->shortMdctSize==120)
      {
         data0 = fromOpus(data0);
         if (data0<0)
            return OPUS_INVALID_PACKET;
      }
      st->end = end = IMAX(1, mode->effEBands-2*(data0>>5));
      LM = (data0>>3)&0x3;
      C = 1 + ((data0>>2)&0x1);
      data++;
      len--;
      if (LM>mode->maxLM)
         return OPUS_INVALID_PACKET;
      if (frame_size < mode->shortMdctSize<<LM)
         return OPUS_BUFFER_TOO_SMALL;
      else
         frame_size = mode->shortMdctSize<<LM;
   } else {
#else
   {
#endif
      for (LM=0;LM<=mode->maxLM;LM++)
         if (mode->shortMdctSize<<LM==frame_size)
            break;
      if (LM>mode->maxLM)
         return OPUS_BAD_ARG;
   }
   M=1<<LM;

   if (len<0 || len>1275 || pcm==NULL)
      return OPUS_BAD_ARG;

   N = M*mode->shortMdctSize;
   c=0; do {
      decode_mem[c] = st->_decode_mem + c*(DECODE_BUFFER_SIZE+overlap);
      out_syn[c] = decode_mem[c]+DECODE_BUFFER_SIZE-N;
   } while (++c<CC);

   effEnd = end;
   if (effEnd > mode->effEBands)
      effEnd = mode->effEBands;

   if (data == NULL || len<=1)
   {
      celt_decode_lost(st, N, LM
#ifdef ENABLE_DEEP_PLC
      , lpcnet
#endif
                      );
      deemphasis(out_syn, pcm, N, CC, st->downsample, mode->preemph, st->preemph_memD, accum);
      RESTORE_STACK;
      return frame_size/st->downsample;
   }
#ifdef ENABLE_DEEP_PLC
   else {
      /* FIXME: This is a bit of a hack just to make sure opus_decode_native() knows we're no longer in PLC. */
      if (lpcnet) lpcnet->blend = 0;
   }
#endif

   /* Check if there are at least two packets received consecutively before
    * turning on the pitch-based PLC */
   if (st->loss_duration == 0) st->skip_plc = 0;

   if (dec == NULL)
   {
      ec_dec_init(&_dec,(unsigned char*)data,len);
      dec = &_dec;
   }

   if (C==1)
   {
      for (i=0;i<nbEBands;i++)
         oldBandE[i]=MAX16(oldBandE[i],oldBandE[nbEBands+i]);
   }

   total_bits = len*8;
   tell = ec_tell(dec);

   if (tell >= total_bits)
      silence = 1;
   else if (tell==1)
      silence = ec_dec_bit_logp(dec, 15);
   else
      silence = 0;
   if (silence)
   {
      /* Pretend we've read all the remaining bits */
      tell = len*8;
      dec->nbits_total+=tell-ec_tell(dec);
   }

   postfilter_gain = 0;
   postfilter_pitch = 0;
   postfilter_tapset = 0;
   if (start==0 && tell+16 <= total_bits)
   {
      if(ec_dec_bit_logp(dec, 1))
      {
         int qg, octave;
         octave = ec_dec_uint(dec, 6);
         postfilter_pitch = (16<<octave)+ec_dec_bits(dec, 4+octave)-1;
         qg = ec_dec_bits(dec, 3);
         if (ec_tell(dec)+2<=total_bits)
            postfilter_tapset = ec_dec_icdf(dec, tapset_icdf, 2);
         postfilter_gain = QCONST16(.09375f,15)*(qg+1);
      }
      tell = ec_tell(dec);
   }

   if (LM > 0 && tell+3 <= total_bits)
   {
      isTransient = ec_dec_bit_logp(dec, 3);
      tell = ec_tell(dec);
   }
   else
      isTransient = 0;

   if (isTransient)
      shortBlocks = M;
   else
      shortBlocks = 0;

   /* Decode the global flags (first symbols in the stream) */
   intra_ener = tell+3<=total_bits ? ec_dec_bit_logp(dec, 3) : 0;
   /* If recovering from packet loss, make sure we make the energy prediction safe to reduce the
      risk of getting loud artifacts. */
   if (!intra_ener && st->loss_duration != 0) {
      c=0; do
      {
         opus_val16 safety = 0;
         int missing = IMIN(10, st->loss_duration>>LM);
         if (LM==0) safety = QCONST16(1.5f,DB_SHIFT);
         else if (LM==1) safety = QCONST16(.5f,DB_SHIFT);
         for (i=start;i<end;i++)
         {
            if (oldBandE[c*nbEBands+i] < MAX16(oldLogE[c*nbEBands+i], oldLogE2[c*nbEBands+i])) {
               /* If energy is going down already, continue the trend. */
               opus_val32 slope;
               opus_val32 E0, E1, E2;
               E0 = oldBandE[c*nbEBands+i];
               E1 = oldLogE[c*nbEBands+i];
               E2 = oldLogE2[c*nbEBands+i];
               slope = MAX32(E1 - E0, HALF32(E2 - E0));
               E0 -= MAX32(0, (1+missing)*slope);
               oldBandE[c*nbEBands+i] = MAX32(-QCONST16(20.f,DB_SHIFT), E0);
            } else {
               /* Otherwise take the min of the last frames. */
               oldBandE[c*nbEBands+i] = MIN16(MIN16(oldBandE[c*nbEBands+i], oldLogE[c*nbEBands+i]), oldLogE2[c*nbEBands+i]);
            }
            /* Shorter frames have more natural fluctuations -- play it safe. */
            oldBandE[c*nbEBands+i] -= safety;
         }
      } while (++c<2);
   }
   /* Get band energies */
   unquant_coarse_energy(mode, start, end, oldBandE,
         intra_ener, dec, C, LM);

   ALLOC(tf_res, nbEBands, int);
   tf_decode(start, end, isTransient, tf_res, LM, dec);

   tell = ec_tell(dec);
   spread_decision = SPREAD_NORMAL;
   if (tell+4 <= total_bits)
      spread_decision = ec_dec_icdf(dec, spread_icdf, 5);

   ALLOC(cap, nbEBands, int);

   init_caps(mode,cap,LM,C);

   ALLOC(offsets, nbEBands, int);

   dynalloc_logp = 6;
   total_bits<<=BITRES;
   tell = ec_tell_frac(dec);
   for (i=start;i<end;i++)
   {
      int width, quanta;
      int dynalloc_loop_logp;
      int boost;
      width = C*(eBands[i+1]-eBands[i])<<LM;
      /* quanta is 6 bits, but no more than 1 bit/sample
         and no less than 1/8 bit/sample */
      quanta = IMIN(width<<BITRES, IMAX(6<<BITRES, width));
      dynalloc_loop_logp = dynalloc_logp;
      boost = 0;
      while (tell+(dynalloc_loop_logp<<BITRES) < total_bits && boost < cap[i])
      {
         int flag;
         flag = ec_dec_bit_logp(dec, dynalloc_loop_logp);
         tell = ec_tell_frac(dec);
         if (!flag)
            break;
         boost += quanta;
         total_bits -= quanta;
         dynalloc_loop_logp = 1;
      }
      offsets[i] = boost;
      /* Making dynalloc more likely */
      if (boost>0)
         dynalloc_logp = IMAX(2, dynalloc_logp-1);
   }

   ALLOC(fine_quant, nbEBands, int);
   alloc_trim = tell+(6<<BITRES) <= total_bits ?
         ec_dec_icdf(dec, trim_icdf, 7) : 5;

   bits = (((opus_int32)len*8)<<BITRES) - ec_tell_frac(dec) - 1;
   anti_collapse_rsv = isTransient&&LM>=2&&bits>=((LM+2)<<BITRES) ? (1<<BITRES) : 0;
   bits -= anti_collapse_rsv;

   ALLOC(pulses, nbEBands, int);
   ALLOC(fine_priority, nbEBands, int);

   codedBands = clt_compute_allocation(mode, start, end, offsets, cap,
         alloc_trim, &intensity, &dual_stereo, bits, &balance, pulses,
         fine_quant, fine_priority, C, LM, dec, 0, 0, 0);

   unquant_fine_energy(mode, start, end, oldBandE, fine_quant, dec, C);

   c=0; do {
      OPUS_MOVE(decode_mem[c], decode_mem[c]+N, DECODE_BUFFER_SIZE-N+overlap);
   } while (++c<CC);

   /* Decode fixed codebook */
   ALLOC(collapse_masks, C*nbEBands, unsigned char);

   ALLOC(X, C*N, celt_norm);   /**< Interleaved normalised MDCTs */

   quant_all_bands(0, mode, start, end, X, C==2 ? X+N : NULL, collapse_masks,
         NULL, pulses, shortBlocks, spread_decision, dual_stereo, intensity, tf_res,
         len*(8<<BITRES)-anti_collapse_rsv, balance, dec, LM, codedBands, &st->rng, 0,
         st->arch, st->disable_inv);

   if (anti_collapse_rsv > 0)
   {
      anti_collapse_on = ec_dec_bits(dec, 1);
   }

   unquant_energy_finalise(mode, start, end, oldBandE,
         fine_quant, fine_priority, len*8-ec_tell(dec), dec, C);

   if (anti_collapse_on)
      anti_collapse(mode, X, collapse_masks, LM, C, N,
            start, end, oldBandE, oldLogE, oldLogE2, pulses, st->rng, st->arch);

   if (silence)
   {
      for (i=0;i<C*nbEBands;i++)
         oldBandE[i] = -QCONST16(28.f,DB_SHIFT);
   }
   if (st->prefilter_and_fold) {
      prefilter_and_fold(st, N);
   }
   celt_synthesis(mode, X, out_syn, oldBandE, start, effEnd,
                  C, CC, isTransient, LM, st->downsample, silence, st->arch);

   c=0; do {
      st->postfilter_period=IMAX(st->postfilter_period, COMBFILTER_MINPERIOD);
      st->postfilter_period_old=IMAX(st->postfilter_period_old, COMBFILTER_MINPERIOD);
      comb_filter(out_syn[c], out_syn[c], st->postfilter_period_old, st->postfilter_period, mode->shortMdctSize,
            st->postfilter_gain_old, st->postfilter_gain, st->postfilter_tapset_old, st->postfilter_tapset,
            mode->window, overlap, st->arch);
      if (LM!=0)
         comb_filter(out_syn[c]+mode->shortMdctSize, out_syn[c]+mode->shortMdctSize, st->postfilter_period, postfilter_pitch, N-mode->shortMdctSize,
               st->postfilter_gain, postfilter_gain, st->postfilter_tapset, postfilter_tapset,
               mode->window, overlap, st->arch);

   } while (++c<CC);
   st->postfilter_period_old = st->postfilter_period;
   st->postfilter_gain_old = st->postfilter_gain;
   st->postfilter_tapset_old = st->postfilter_tapset;
   st->postfilter_period = postfilter_pitch;
   st->postfilter_gain = postfilter_gain;
   st->postfilter_tapset = postfilter_tapset;
   if (LM!=0)
   {
      st->postfilter_period_old = st->postfilter_period;
      st->postfilter_gain_old = st->postfilter_gain;
      st->postfilter_tapset_old = st->postfilter_tapset;
   }

   if (C==1)
      OPUS_COPY(&oldBandE[nbEBands], oldBandE, nbEBands);

   if (!isTransient)
   {
      OPUS_COPY(oldLogE2, oldLogE, 2*nbEBands);
      OPUS_COPY(oldLogE, oldBandE, 2*nbEBands);
   } else {
      for (i=0;i<2*nbEBands;i++)
         oldLogE[i] = MIN16(oldLogE[i], oldBandE[i]);
   }
   /* In normal circumstances, we only allow the noise floor to increase by
      up to 2.4 dB/second, but when we're in DTX we give the weight of
      all missing packets to the update packet. */
   max_background_increase = IMIN(160, st->loss_duration+M)*QCONST16(0.001f,DB_SHIFT);
   for (i=0;i<2*nbEBands;i++)
      backgroundLogE[i] = MIN16(backgroundLogE[i] + max_background_increase, oldBandE[i]);
   /* In case start or end were to change */
   c=0; do
   {
      for (i=0;i<start;i++)
      {
         oldBandE[c*nbEBands+i]=0;
         oldLogE[c*nbEBands+i]=oldLogE2[c*nbEBands+i]=-QCONST16(28.f,DB_SHIFT);
      }
      for (i=end;i<nbEBands;i++)
      {
         oldBandE[c*nbEBands+i]=0;
         oldLogE[c*nbEBands+i]=oldLogE2[c*nbEBands+i]=-QCONST16(28.f,DB_SHIFT);
      }
   } while (++c<2);
   st->rng = dec->rng;

   deemphasis(out_syn, pcm, N, CC, st->downsample, mode->preemph, st->preemph_memD, accum);
   st->loss_duration = 0;
   st->prefilter_and_fold = 0;
   RESTORE_STACK;
   if (ec_tell(dec) > 8*len)
      return OPUS_INTERNAL_ERROR;
   if(ec_get_error(dec))
      st->error = 1;
   return frame_size/st->downsample;
}

int celt_decode_with_ec(CELTDecoder * OPUS_RESTRICT st, const unsigned char *data,
      int len, opus_val16 * OPUS_RESTRICT pcm, int frame_size, ec_dec *dec, int accum)
{
   return celt_decode_with_ec_dred(st, data, len, pcm, frame_size, dec, accum
#ifdef ENABLE_DEEP_PLC
       , NULL
#endif
       );
}

#ifdef CUSTOM_MODES

#ifdef FIXED_POINT
int opus_custom_decode(CELTDecoder * OPUS_RESTRICT st, const unsigned char *data, int len, opus_int16 * OPUS_RESTRICT pcm, int frame_size)
{
   return celt_decode_with_ec(st, data, len, pcm, frame_size, NULL, 0);
}

#ifndef DISABLE_FLOAT_API
int opus_custom_decode_float(CELTDecoder * OPUS_RESTRICT st, const unsigned char *data, int len, float * OPUS_RESTRICT pcm, int frame_size)
{
   int j, ret, C, N;
   VARDECL(opus_int16, out);
   ALLOC_STACK;

   if (pcm==NULL)
      return OPUS_BAD_ARG;

   C = st->channels;
   N = frame_size;

   ALLOC(out, C*N, opus_int16);
   ret=celt_decode_with_ec(st, data, len, out, frame_size, NULL, 0);
   if (ret>0)
      for (j=0;j<C*ret;j++)
         pcm[j]=out[j]*(1.f/32768.f);

   RESTORE_STACK;
   return ret;
}
#endif /* DISABLE_FLOAT_API */

#else

int opus_custom_decode_float(CELTDecoder * OPUS_RESTRICT st, const unsigned char *data, int len, float * OPUS_RESTRICT pcm, int frame_size)
{
   return celt_decode_with_ec(st, data, len, pcm, frame_size, NULL, 0);
}

int opus_custom_decode(CELTDecoder * OPUS_RESTRICT st, const unsigned char *data, int len, opus_int16 * OPUS_RESTRICT pcm, int frame_size)
{
   int j, ret, C, N;
   VARDECL(celt_sig, out);
   ALLOC_STACK;

   if (pcm==NULL)
      return OPUS_BAD_ARG;

   C = st->channels;
   N = frame_size;
   ALLOC(out, C*N, celt_sig);

   ret=celt_decode_with_ec(st, data, len, out, frame_size, NULL, 0);

   if (ret>0)
      for (j=0;j<C*ret;j++)
         pcm[j] = FLOAT2INT16 (out[j]);

   RESTORE_STACK;
   return ret;
}

#endif
#endif /* CUSTOM_MODES */

int opus_custom_decoder_ctl(CELTDecoder * OPUS_RESTRICT st, int request, ...)
{
   va_list ap;

   va_start(ap, request);
   switch (request)
   {
      case OPUS_SET_COMPLEXITY_REQUEST:
      {
          opus_int32 value = va_arg(ap, opus_int32);
          if(value<0 || value>10)
          {
             goto bad_arg;
          }
          st->complexity = value;
      }
      break;
      case OPUS_GET_COMPLEXITY_REQUEST:
      {
          opus_int32 *value = va_arg(ap, opus_int32*);
          if (!value)
          {
             goto bad_arg;
          }
          *value = st->complexity;
      }
      break;
      case CELT_SET_START_BAND_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         if (value<0 || value>=st->mode->nbEBands)
            goto bad_arg;
         st->start = value;
      }
      break;
      case CELT_SET_END_BAND_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         if (value<1 || value>st->mode->nbEBands)
            goto bad_arg;
         st->end = value;
      }
      break;
      case CELT_SET_CHANNELS_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         if (value<1 || value>2)
            goto bad_arg;
         st->stream_channels = value;
      }
      break;
      case CELT_GET_AND_CLEAR_ERROR_REQUEST:
      {
         opus_int32 *value = va_arg(ap, opus_int32*);
         if (value==NULL)
            goto bad_arg;
         *value=st->error;
         st->error = 0;
      }
      break;
      case OPUS_GET_LOOKAHEAD_REQUEST:
      {
         opus_int32 *value = va_arg(ap, opus_int32*);
         if (value==NULL)
            goto bad_arg;
         *value = st->overlap/st->downsample;
      }
      break;
      case OPUS_RESET_STATE:
      {
         int i;
         opus_val16 *lpc, *oldBandE, *oldLogE, *oldLogE2;
         lpc = (opus_val16*)(st->_decode_mem+(DECODE_BUFFER_SIZE+st->overlap)*st->channels);
         oldBandE = lpc+st->channels*CELT_LPC_ORDER;
         oldLogE = oldBandE + 2*st->mode->nbEBands;
         oldLogE2 = oldLogE + 2*st->mode->nbEBands;
         OPUS_CLEAR((char*)&st->DECODER_RESET_START,
               opus_custom_decoder_get_size(st->mode, st->channels)-
               ((char*)&st->DECODER_RESET_START - (char*)st));
         for (i=0;i<2*st->mode->nbEBands;i++)
            oldLogE[i]=oldLogE2[i]=-QCONST16(28.f,DB_SHIFT);
         st->skip_plc = 1;
      }
      break;
      case OPUS_GET_PITCH_REQUEST:
      {
         opus_int32 *value = va_arg(ap, opus_int32*);
         if (value==NULL)
            goto bad_arg;
         *value = st->postfilter_period;
      }
      break;
      case CELT_GET_MODE_REQUEST:
      {
         const CELTMode ** value = va_arg(ap, const CELTMode**);
         if (value==0)
            goto bad_arg;
         *value=st->mode;
      }
      break;
      case CELT_SET_SIGNALLING_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         st->signalling = value;
      }
      break;
      case OPUS_GET_FINAL_RANGE_REQUEST:
      {
         opus_uint32 * value = va_arg(ap, opus_uint32 *);
         if (value==0)
            goto bad_arg;
         *value=st->rng;
      }
      break;
      case OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST:
      {
          opus_int32 value = va_arg(ap, opus_int32);
          if(value<0 || value>1)
          {
             goto bad_arg;
          }
          st->disable_inv = value;
      }
      break;
      case OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST:
      {
          opus_int32 *value = va_arg(ap, opus_int32*);
          if (!value)
          {
             goto bad_arg;
          }
          *value = st->disable_inv;
      }
      break;
      default:
         goto bad_request;
   }
   va_end(ap);
   return OPUS_OK;
bad_arg:
   va_end(ap);
   return OPUS_BAD_ARG;
bad_request:
      va_end(ap);
  return OPUS_UNIMPLEMENTED;
}
