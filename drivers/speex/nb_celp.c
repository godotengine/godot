/* Copyright (C) 2002-2006 Jean-Marc Valin 
   File: nb_celp.c

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "config.h"


#include <math.h>
#include "nb_celp.h"
#include "lpc.h"
#include "lsp.h"
#include "ltp.h"
#include "quant_lsp.h"
#include "cb_search.h"
#include "filters.h"
#include "stack_alloc.h"
#include "vq.h"
#include <speex/speex_bits.h>
#include "vbr.h"
#include "arch.h"
#include "math_approx.h"
#include "os_support.h"
#include <speex/speex_callbacks.h>

#ifdef VORBIS_PSYCHO
#include "vorbis_psy.h"
#endif

#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif

#ifndef NULL
#define NULL 0
#endif

#define SUBMODE(x) st->submodes[st->submodeID]->x

/* Default size for the encoder and decoder stack (can be changed at compile time).
   This does not apply when using variable-size arrays or alloca. */
#ifndef NB_ENC_STACK
#define NB_ENC_STACK (8000*sizeof(spx_sig_t))
#endif

#ifndef NB_DEC_STACK
#define NB_DEC_STACK (4000*sizeof(spx_sig_t))
#endif


#ifdef FIXED_POINT
const spx_word32_t ol_gain_table[32]={18900, 25150, 33468, 44536, 59265, 78865, 104946, 139653, 185838, 247297, 329081, 437913, 582736, 775454, 1031906, 1373169, 1827293, 2431601, 3235761, 4305867, 5729870, 7624808, 10146425, 13501971, 17967238, 23909222, 31816294, 42338330, 56340132, 74972501, 99766822, 132760927};
const spx_word16_t exc_gain_quant_scal3_bound[7]={1841, 3883, 6051, 8062, 10444, 13580, 18560};
const spx_word16_t exc_gain_quant_scal3[8]={1002, 2680, 5086, 7016, 9108, 11781, 15380, 21740};
const spx_word16_t exc_gain_quant_scal1_bound[1]={14385};
const spx_word16_t exc_gain_quant_scal1[2]={11546, 17224};

#define LSP_MARGIN 16
#define LSP_DELTA1 6553
#define LSP_DELTA2 1638

#else

const float exc_gain_quant_scal3_bound[7]={0.112338f, 0.236980f, 0.369316f, 0.492054f, 0.637471f, 0.828874f, 1.132784f};
const float exc_gain_quant_scal3[8]={0.061130f, 0.163546f, 0.310413f, 0.428220f, 0.555887f, 0.719055f, 0.938694f, 1.326874f};
const float exc_gain_quant_scal1_bound[1]={0.87798f};
const float exc_gain_quant_scal1[2]={0.70469f, 1.05127f};

#define LSP_MARGIN .002f
#define LSP_DELTA1 .2f
#define LSP_DELTA2 .05f

#endif

#ifdef VORBIS_PSYCHO
#define EXTRA_BUFFER 100
#else
#define EXTRA_BUFFER 0
#endif


#define sqr(x) ((x)*(x))

extern const spx_word16_t lag_window[];
extern const spx_word16_t lpc_window[];

void *nb_encoder_init(const SpeexMode *m)
{
   EncState *st;
   const SpeexNBMode *mode;
   int i;

   mode=(const SpeexNBMode *)m->mode;
   st = (EncState*)speex_alloc(sizeof(EncState));
   if (!st)
      return NULL;
#if defined(VAR_ARRAYS) || defined (USE_ALLOCA)
   st->stack = NULL;
#else
   st->stack = (char*)speex_alloc_scratch(NB_ENC_STACK);
#endif
   
   st->mode=m;

   st->frameSize = mode->frameSize;
   st->nbSubframes=mode->frameSize/mode->subframeSize;
   st->subframeSize=mode->subframeSize;
   st->windowSize = st->frameSize+st->subframeSize;
   st->lpcSize = mode->lpcSize;
   st->gamma1=mode->gamma1;
   st->gamma2=mode->gamma2;
   st->min_pitch=mode->pitchStart;
   st->max_pitch=mode->pitchEnd;
   st->lpc_floor = mode->lpc_floor;
  
   st->submodes=mode->submodes;
   st->submodeID=st->submodeSelect=mode->defaultSubmode;
   st->bounded_pitch = 1;

   st->encode_submode = 1;

#ifdef VORBIS_PSYCHO
   st->psy = vorbis_psy_init(8000, 256);
   st->curve = (float*)speex_alloc(128*sizeof(float));
   st->old_curve = (float*)speex_alloc(128*sizeof(float));
   st->psy_window = (float*)speex_alloc(256*sizeof(float));
#endif

   st->cumul_gain = 1024;

   /* Allocating input buffer */
   st->winBuf = (spx_word16_t*)speex_alloc((st->windowSize-st->frameSize)*sizeof(spx_word16_t));
   /* Allocating excitation buffer */
   st->excBuf = (spx_word16_t*)speex_alloc((mode->frameSize+mode->pitchEnd+2)*sizeof(spx_word16_t));
   st->exc = st->excBuf + mode->pitchEnd + 2;
   st->swBuf = (spx_word16_t*)speex_alloc((mode->frameSize+mode->pitchEnd+2)*sizeof(spx_word16_t));
   st->sw = st->swBuf + mode->pitchEnd + 2;

   st->window= lpc_window;
   
   /* Create the window for autocorrelation (lag-windowing) */
   st->lagWindow = lag_window;

   st->old_lsp = (spx_lsp_t*)speex_alloc((st->lpcSize)*sizeof(spx_lsp_t));
   st->old_qlsp = (spx_lsp_t*)speex_alloc((st->lpcSize)*sizeof(spx_lsp_t));
   st->first = 1;
   for (i=0;i<st->lpcSize;i++)
      st->old_lsp[i]= DIV32(MULT16_16(QCONST16(3.1415927f, LSP_SHIFT), i+1), st->lpcSize+1);

   st->mem_sp = (spx_mem_t*)speex_alloc((st->lpcSize)*sizeof(spx_mem_t));
   st->mem_sw = (spx_mem_t*)speex_alloc((st->lpcSize)*sizeof(spx_mem_t));
   st->mem_sw_whole = (spx_mem_t*)speex_alloc((st->lpcSize)*sizeof(spx_mem_t));
   st->mem_exc = (spx_mem_t*)speex_alloc((st->lpcSize)*sizeof(spx_mem_t));
   st->mem_exc2 = (spx_mem_t*)speex_alloc((st->lpcSize)*sizeof(spx_mem_t));

   st->pi_gain = (spx_word32_t*)speex_alloc((st->nbSubframes)*sizeof(spx_word32_t));
   st->innov_rms_save = NULL;
   
   st->pitch = (int*)speex_alloc((st->nbSubframes)*sizeof(int));

#ifndef DISABLE_VBR
   st->vbr = (VBRState*)speex_alloc(sizeof(VBRState));
   vbr_init(st->vbr);
   st->vbr_quality = 8;
   st->vbr_enabled = 0;
   st->vbr_max = 0;
   st->vad_enabled = 0;
   st->dtx_enabled = 0;
   st->dtx_count=0;
   st->abr_enabled = 0;
   st->abr_drift = 0;
   st->abr_drift2 = 0;
#endif /* #ifndef DISABLE_VBR */

   st->plc_tuning = 2;
   st->complexity=2;
   st->sampling_rate=8000;
   st->isWideband = 0;
   st->highpass_enabled = 1;
   
#ifdef ENABLE_VALGRIND
   VALGRIND_MAKE_READABLE(st, NB_ENC_STACK);
#endif
   return st;
}

void nb_encoder_destroy(void *state)
{
   EncState *st=(EncState *)state;
   /* Free all allocated memory */
#if !(defined(VAR_ARRAYS) || defined (USE_ALLOCA))
   speex_free_scratch(st->stack);
#endif

   speex_free (st->winBuf);
   speex_free (st->excBuf);
   speex_free (st->old_qlsp);
   speex_free (st->swBuf);

   speex_free (st->old_lsp);
   speex_free (st->mem_sp);
   speex_free (st->mem_sw);
   speex_free (st->mem_sw_whole);
   speex_free (st->mem_exc);
   speex_free (st->mem_exc2);
   speex_free (st->pi_gain);
   speex_free (st->pitch);

#ifndef DISABLE_VBR
   vbr_destroy(st->vbr);
   speex_free (st->vbr);
#endif /* #ifndef DISABLE_VBR */

#ifdef VORBIS_PSYCHO
   vorbis_psy_destroy(st->psy);
   speex_free (st->curve);
   speex_free (st->old_curve);
   speex_free (st->psy_window);
#endif

   /*Free state memory... should be last*/
   speex_free(st);
}

int nb_encode(void *state, void *vin, SpeexBits *bits)
{
   EncState *st;
   int i, sub, roots;
   int ol_pitch;
   spx_word16_t ol_pitch_coef;
   spx_word32_t ol_gain;
   VARDECL(spx_word16_t *ringing);
   VARDECL(spx_word16_t *target);
   VARDECL(spx_sig_t *innov);
   VARDECL(spx_word32_t *exc32);
   VARDECL(spx_mem_t *mem);
   VARDECL(spx_coef_t *bw_lpc1);
   VARDECL(spx_coef_t *bw_lpc2);
   VARDECL(spx_coef_t *lpc);
   VARDECL(spx_lsp_t *lsp);
   VARDECL(spx_lsp_t *qlsp);
   VARDECL(spx_lsp_t *interp_lsp);
   VARDECL(spx_lsp_t *interp_qlsp);
   VARDECL(spx_coef_t *interp_lpc);
   VARDECL(spx_coef_t *interp_qlpc);
   char *stack;
   VARDECL(spx_word16_t *syn_resp);
   VARDECL(spx_word16_t *real_exc);
   
   spx_word32_t ener=0;
   spx_word16_t fine_gain;
   spx_word16_t *in = (spx_word16_t*)vin;

   st=(EncState *)state;
   stack=st->stack;

   ALLOC(lpc, st->lpcSize, spx_coef_t);
   ALLOC(bw_lpc1, st->lpcSize, spx_coef_t);
   ALLOC(bw_lpc2, st->lpcSize, spx_coef_t);
   ALLOC(lsp, st->lpcSize, spx_lsp_t);
   ALLOC(qlsp, st->lpcSize, spx_lsp_t);
   ALLOC(interp_lsp, st->lpcSize, spx_lsp_t);
   ALLOC(interp_qlsp, st->lpcSize, spx_lsp_t);
   ALLOC(interp_lpc, st->lpcSize, spx_coef_t);
   ALLOC(interp_qlpc, st->lpcSize, spx_coef_t);

   /* Move signals 1 frame towards the past */
   SPEEX_MOVE(st->excBuf, st->excBuf+st->frameSize, st->max_pitch+2);
   SPEEX_MOVE(st->swBuf, st->swBuf+st->frameSize, st->max_pitch+2);

   if (st->highpass_enabled)
      highpass(in, in, st->frameSize, (st->isWideband?HIGHPASS_WIDEBAND:HIGHPASS_NARROWBAND)|HIGHPASS_INPUT, st->mem_hp);
   
   {
      VARDECL(spx_word16_t *w_sig);
      VARDECL(spx_word16_t *autocorr);
      ALLOC(w_sig, st->windowSize, spx_word16_t);
      ALLOC(autocorr, st->lpcSize+1, spx_word16_t);
      /* Window for analysis */
      for (i=0;i<st->windowSize-st->frameSize;i++)
         w_sig[i] = EXTRACT16(SHR32(MULT16_16(st->winBuf[i],st->window[i]),SIG_SHIFT));
      for (;i<st->windowSize;i++)
         w_sig[i] = EXTRACT16(SHR32(MULT16_16(in[i-st->windowSize+st->frameSize],st->window[i]),SIG_SHIFT));
      /* Compute auto-correlation */
      _spx_autocorr(w_sig, autocorr, st->lpcSize+1, st->windowSize);
      autocorr[0] = ADD16(autocorr[0],MULT16_16_Q15(autocorr[0],st->lpc_floor)); /* Noise floor in auto-correlation domain */

      /* Lag windowing: equivalent to filtering in the power-spectrum domain */
      for (i=0;i<st->lpcSize+1;i++)
         autocorr[i] = MULT16_16_Q14(autocorr[i],st->lagWindow[i]);

      /* Levinson-Durbin */
      _spx_lpc(lpc, autocorr, st->lpcSize);
      /* LPC to LSPs (x-domain) transform */
      roots=lpc_to_lsp (lpc, st->lpcSize, lsp, 10, LSP_DELTA1, stack);
      /* Check if we found all the roots */
      if (roots!=st->lpcSize)
      {
         /*If we can't find all LSP's, do some damage control and use previous filter*/
         for (i=0;i<st->lpcSize;i++)
         {
            lsp[i]=st->old_lsp[i];
         }
      }
   }




   /* Whole frame analysis (open-loop estimation of pitch and excitation gain) */
   {
      int diff = st->windowSize-st->frameSize;
      if (st->first)
         for (i=0;i<st->lpcSize;i++)
            interp_lsp[i] = lsp[i];
      else
         lsp_interpolate(st->old_lsp, lsp, interp_lsp, st->lpcSize, st->nbSubframes, st->nbSubframes<<1);

      lsp_enforce_margin(interp_lsp, st->lpcSize, LSP_MARGIN);

      /* Compute interpolated LPCs (unquantized) for whole frame*/
      lsp_to_lpc(interp_lsp, interp_lpc, st->lpcSize,stack);


      /*Open-loop pitch*/
      if (!st->submodes[st->submodeID] || (st->complexity>2 && SUBMODE(have_subframe_gain)<3) || SUBMODE(forced_pitch_gain) || SUBMODE(lbr_pitch) != -1 
#ifndef DISABLE_VBR
           || st->vbr_enabled || st->vad_enabled
#endif
                  )
      {
         int nol_pitch[6];
         spx_word16_t nol_pitch_coef[6];
         
         bw_lpc(st->gamma1, interp_lpc, bw_lpc1, st->lpcSize);
         bw_lpc(st->gamma2, interp_lpc, bw_lpc2, st->lpcSize);

         SPEEX_COPY(st->sw, st->winBuf, diff);
         SPEEX_COPY(st->sw+diff, in, st->frameSize-diff);
         filter_mem16(st->sw, bw_lpc1, bw_lpc2, st->sw, st->frameSize, st->lpcSize, st->mem_sw_whole, stack);

         open_loop_nbest_pitch(st->sw, st->min_pitch, st->max_pitch, st->frameSize, 
                               nol_pitch, nol_pitch_coef, 6, stack);
         ol_pitch=nol_pitch[0];
         ol_pitch_coef = nol_pitch_coef[0];
         /*Try to remove pitch multiples*/
         for (i=1;i<6;i++)
         {
#ifdef FIXED_POINT
            if ((nol_pitch_coef[i]>MULT16_16_Q15(nol_pitch_coef[0],27853)) && 
#else
            if ((nol_pitch_coef[i]>.85*nol_pitch_coef[0]) && 
#endif
                (ABS(2*nol_pitch[i]-ol_pitch)<=2 || ABS(3*nol_pitch[i]-ol_pitch)<=3 || 
                 ABS(4*nol_pitch[i]-ol_pitch)<=4 || ABS(5*nol_pitch[i]-ol_pitch)<=5))
            {
               /*ol_pitch_coef=nol_pitch_coef[i];*/
               ol_pitch = nol_pitch[i];
            }
         }
         /*if (ol_pitch>50)
           ol_pitch/=2;*/
         /*ol_pitch_coef = sqrt(ol_pitch_coef);*/

      } else {
         ol_pitch=0;
         ol_pitch_coef=0;
      }
      
      /*Compute "real" excitation*/
      SPEEX_COPY(st->exc, st->winBuf, diff);
      SPEEX_COPY(st->exc+diff, in, st->frameSize-diff);
      fir_mem16(st->exc, interp_lpc, st->exc, st->frameSize, st->lpcSize, st->mem_exc, stack);

      /* Compute open-loop excitation gain */
      {
         spx_word16_t g = compute_rms16(st->exc, st->frameSize);
         if (st->submodeID!=1 && ol_pitch>0)
            ol_gain = MULT16_16(g, MULT16_16_Q14(QCONST16(1.1,14),
                                spx_sqrt(QCONST32(1.,28)-MULT16_32_Q15(QCONST16(.8,15),SHL32(MULT16_16(ol_pitch_coef,ol_pitch_coef),16)))));
         else
            ol_gain = SHL32(EXTEND32(g),SIG_SHIFT);
      }
   }

#ifdef VORBIS_PSYCHO
   SPEEX_MOVE(st->psy_window, st->psy_window+st->frameSize, 256-st->frameSize);
   SPEEX_COPY(&st->psy_window[256-st->frameSize], in, st->frameSize);
   compute_curve(st->psy, st->psy_window, st->curve);
   /*print_vec(st->curve, 128, "curve");*/
   if (st->first)
      SPEEX_COPY(st->old_curve, st->curve, 128);
#endif

   /*VBR stuff*/
#ifndef DISABLE_VBR
   if (st->vbr && (st->vbr_enabled||st->vad_enabled))
   {
      float lsp_dist=0;
      for (i=0;i<st->lpcSize;i++)
         lsp_dist += (st->old_lsp[i] - lsp[i])*(st->old_lsp[i] - lsp[i]);
      lsp_dist /= LSP_SCALING*LSP_SCALING;
      
      if (st->abr_enabled)
      {
         float qual_change=0;
         if (st->abr_drift2 * st->abr_drift > 0)
         {
            /* Only adapt if long-term and short-term drift are the same sign */
            qual_change = -.00001*st->abr_drift/(1+st->abr_count);
            if (qual_change>.05)
               qual_change=.05;
            if (qual_change<-.05)
               qual_change=-.05;
         }
         st->vbr_quality += qual_change;
         if (st->vbr_quality>10)
            st->vbr_quality=10;
         if (st->vbr_quality<0)
            st->vbr_quality=0;
      }

      st->relative_quality = vbr_analysis(st->vbr, in, st->frameSize, ol_pitch, GAIN_SCALING_1*ol_pitch_coef);
      /*if (delta_qual<0)*/
      /*  delta_qual*=.1*(3+st->vbr_quality);*/
      if (st->vbr_enabled) 
      {
         spx_int32_t mode;
         int choice=0;
         float min_diff=100;
         mode = 8;
         while (mode)
         {
            int v1;
            float thresh;
            v1=(int)floor(st->vbr_quality);
            if (v1==10)
               thresh = vbr_nb_thresh[mode][v1];
            else
               thresh = (st->vbr_quality-v1)*vbr_nb_thresh[mode][v1+1] + (1+v1-st->vbr_quality)*vbr_nb_thresh[mode][v1];
            if (st->relative_quality > thresh && 
                st->relative_quality-thresh<min_diff)
            {
               choice = mode;
               min_diff = st->relative_quality-thresh;
            }
            mode--;
         }
         mode=choice;
         if (mode==0)
         {
            if (st->dtx_count==0 || lsp_dist>.05 || !st->dtx_enabled || st->dtx_count>20)
            {
               mode=1;
               st->dtx_count=1;
            } else {
               mode=0;
               st->dtx_count++;
            }
         } else {
            st->dtx_count=0;
         }

         speex_encoder_ctl(state, SPEEX_SET_MODE, &mode);
         if (st->vbr_max>0)
         {
            spx_int32_t rate;
            speex_encoder_ctl(state, SPEEX_GET_BITRATE, &rate);
            if (rate > st->vbr_max)
            {
               rate = st->vbr_max;
               speex_encoder_ctl(state, SPEEX_SET_BITRATE, &rate);
            }
         }
         
         if (st->abr_enabled)
         {
            spx_int32_t bitrate;
            speex_encoder_ctl(state, SPEEX_GET_BITRATE, &bitrate);
            st->abr_drift+=(bitrate-st->abr_enabled);
            st->abr_drift2 = .95*st->abr_drift2 + .05*(bitrate-st->abr_enabled);
            st->abr_count += 1.0;
         }

      } else {
         /*VAD only case*/
         int mode;
         if (st->relative_quality<2)
         {
            if (st->dtx_count==0 || lsp_dist>.05 || !st->dtx_enabled || st->dtx_count>20)
            {
               st->dtx_count=1;
               mode=1;
            } else {
               mode=0;
               st->dtx_count++;
            }
         } else {
            st->dtx_count = 0;
            mode=st->submodeSelect;
         }
         /*speex_encoder_ctl(state, SPEEX_SET_MODE, &mode);*/
         st->submodeID=mode;
      } 
   } else {
      st->relative_quality = -1;
   }
#endif /* #ifndef DISABLE_VBR */

   if (st->encode_submode)
   {
      /* First, transmit a zero for narrowband */
      speex_bits_pack(bits, 0, 1);

      /* Transmit the sub-mode we use for this frame */
      speex_bits_pack(bits, st->submodeID, NB_SUBMODE_BITS);

   }

   /* If null mode (no transmission), just set a couple things to zero*/
   if (st->submodes[st->submodeID] == NULL)
   {
      for (i=0;i<st->frameSize;i++)
         st->exc[i]=st->sw[i]=VERY_SMALL;

      for (i=0;i<st->lpcSize;i++)
         st->mem_sw[i]=0;
      st->first=1;
      st->bounded_pitch = 1;

      SPEEX_COPY(st->winBuf, in+2*st->frameSize-st->windowSize, st->windowSize-st->frameSize);

      /* Clear memory (no need to really compute it) */
      for (i=0;i<st->lpcSize;i++)
         st->mem_sp[i] = 0;
      return 0;

   }

   /* LSP Quantization */
   if (st->first)
   {
      for (i=0;i<st->lpcSize;i++)
         st->old_lsp[i] = lsp[i];
   }


   /*Quantize LSPs*/
#if 1 /*0 for unquantized*/
   SUBMODE(lsp_quant)(lsp, qlsp, st->lpcSize, bits);
#else
   for (i=0;i<st->lpcSize;i++)
     qlsp[i]=lsp[i];
#endif

   /*If we use low bit-rate pitch mode, transmit open-loop pitch*/
   if (SUBMODE(lbr_pitch)!=-1)
   {
      speex_bits_pack(bits, ol_pitch-st->min_pitch, 7);
   } 

   if (SUBMODE(forced_pitch_gain))
   {
      int quant;
      /* This just damps the pitch a bit, because it tends to be too aggressive when forced */
      ol_pitch_coef = MULT16_16_Q15(QCONST16(.9,15), ol_pitch_coef);
#ifdef FIXED_POINT
      quant = PSHR16(MULT16_16_16(15, ol_pitch_coef),GAIN_SHIFT);
#else
      quant = (int)floor(.5+15*ol_pitch_coef*GAIN_SCALING_1);
#endif
      if (quant>15)
         quant=15;
      if (quant<0)
         quant=0;
      speex_bits_pack(bits, quant, 4);
      ol_pitch_coef=MULT16_16_P15(QCONST16(0.066667,15),SHL16(quant,GAIN_SHIFT));
   }
   
   
   /*Quantize and transmit open-loop excitation gain*/
#ifdef FIXED_POINT
   {
      int qe = scal_quant32(ol_gain, ol_gain_table, 32);
      /*ol_gain = exp(qe/3.5)*SIG_SCALING;*/
      ol_gain = MULT16_32_Q15(28406,ol_gain_table[qe]);
      speex_bits_pack(bits, qe, 5);
   }
#else
   {
      int qe = (int)(floor(.5+3.5*log(ol_gain*1.0/SIG_SCALING)));
      if (qe<0)
         qe=0;
      if (qe>31)
         qe=31;
      ol_gain = exp(qe/3.5)*SIG_SCALING;
      speex_bits_pack(bits, qe, 5);
   }
#endif



   /* Special case for first frame */
   if (st->first)
   {
      for (i=0;i<st->lpcSize;i++)
         st->old_qlsp[i] = qlsp[i];
   }

   /* Target signal */
   ALLOC(target, st->subframeSize, spx_word16_t);
   ALLOC(innov, st->subframeSize, spx_sig_t);
   ALLOC(exc32, st->subframeSize, spx_word32_t);
   ALLOC(ringing, st->subframeSize, spx_word16_t);
   ALLOC(syn_resp, st->subframeSize, spx_word16_t);
   ALLOC(real_exc, st->subframeSize, spx_word16_t);
   ALLOC(mem, st->lpcSize, spx_mem_t);

   /* Loop on sub-frames */
   for (sub=0;sub<st->nbSubframes;sub++)
   {
      int   offset;
      spx_word16_t *sw;
      spx_word16_t *exc;
      int pitch;
      int response_bound = st->subframeSize;

      /* Offset relative to start of frame */
      offset = st->subframeSize*sub;
      /* Excitation */
      exc=st->exc+offset;
      /* Weighted signal */
      sw=st->sw+offset;
      
      /* LSP interpolation (quantized and unquantized) */
      lsp_interpolate(st->old_lsp, lsp, interp_lsp, st->lpcSize, sub, st->nbSubframes);
      lsp_interpolate(st->old_qlsp, qlsp, interp_qlsp, st->lpcSize, sub, st->nbSubframes);

      /* Make sure the filters are stable */
      lsp_enforce_margin(interp_lsp, st->lpcSize, LSP_MARGIN);
      lsp_enforce_margin(interp_qlsp, st->lpcSize, LSP_MARGIN);

      /* Compute interpolated LPCs (quantized and unquantized) */
      lsp_to_lpc(interp_lsp, interp_lpc, st->lpcSize,stack);

      lsp_to_lpc(interp_qlsp, interp_qlpc, st->lpcSize, stack);

      /* Compute analysis filter gain at w=pi (for use in SB-CELP) */
      {
         spx_word32_t pi_g=LPC_SCALING;
         for (i=0;i<st->lpcSize;i+=2)
         {
            /*pi_g += -st->interp_qlpc[i] +  st->interp_qlpc[i+1];*/
            pi_g = ADD32(pi_g, SUB32(EXTEND32(interp_qlpc[i+1]),EXTEND32(interp_qlpc[i])));
         }
         st->pi_gain[sub] = pi_g;
      }

#ifdef VORBIS_PSYCHO
      {
         float curr_curve[128];
         float fact = ((float)sub+1.0f)/st->nbSubframes;
         for (i=0;i<128;i++)
            curr_curve[i] = (1.0f-fact)*st->old_curve[i] + fact*st->curve[i];
         curve_to_lpc(st->psy, curr_curve, bw_lpc1, bw_lpc2, 10);
      }
#else
      /* Compute bandwidth-expanded (unquantized) LPCs for perceptual weighting */
      bw_lpc(st->gamma1, interp_lpc, bw_lpc1, st->lpcSize);
      if (st->gamma2>=0)
         bw_lpc(st->gamma2, interp_lpc, bw_lpc2, st->lpcSize);
      else
      {
         for (i=0;i<st->lpcSize;i++)
            bw_lpc2[i]=0;
      }
      /*print_vec(st->bw_lpc1, 10, "bw_lpc");*/
#endif

      /*FIXME: This will break if we change the window size */
      speex_assert(st->windowSize-st->frameSize == st->subframeSize);
      if (sub==0)
      {
         for (i=0;i<st->subframeSize;i++)
            real_exc[i] = sw[i] = st->winBuf[i];
      } else {
         for (i=0;i<st->subframeSize;i++)
            real_exc[i] = sw[i] = in[i+((sub-1)*st->subframeSize)];
      }
      fir_mem16(real_exc, interp_qlpc, real_exc, st->subframeSize, st->lpcSize, st->mem_exc2, stack);
      
      if (st->complexity==0)
         response_bound >>= 1;
      compute_impulse_response(interp_qlpc, bw_lpc1, bw_lpc2, syn_resp, response_bound, st->lpcSize, stack);
      for (i=response_bound;i<st->subframeSize;i++)
         syn_resp[i]=VERY_SMALL;
      
      /* Compute zero response of A(z/g1) / ( A(z/g2) * A(z) ) */
      for (i=0;i<st->lpcSize;i++)
         mem[i]=SHL32(st->mem_sp[i],1);
      for (i=0;i<st->subframeSize;i++)
         ringing[i] = VERY_SMALL;
#ifdef SHORTCUTS2
      iir_mem16(ringing, interp_qlpc, ringing, response_bound, st->lpcSize, mem, stack);
      for (i=0;i<st->lpcSize;i++)
         mem[i]=SHL32(st->mem_sw[i],1);
      filter_mem16(ringing, st->bw_lpc1, st->bw_lpc2, ringing, response_bound, st->lpcSize, mem, stack);
      SPEEX_MEMSET(&ringing[response_bound], 0, st->subframeSize-response_bound);
#else
      iir_mem16(ringing, interp_qlpc, ringing, st->subframeSize, st->lpcSize, mem, stack);
      for (i=0;i<st->lpcSize;i++)
         mem[i]=SHL32(st->mem_sw[i],1);
      filter_mem16(ringing, bw_lpc1, bw_lpc2, ringing, st->subframeSize, st->lpcSize, mem, stack);
#endif
      
      /* Compute weighted signal */
      for (i=0;i<st->lpcSize;i++)
         mem[i]=st->mem_sw[i];
      filter_mem16(sw, bw_lpc1, bw_lpc2, sw, st->subframeSize, st->lpcSize, mem, stack);
      
      if (st->complexity==0)
         for (i=0;i<st->lpcSize;i++)
            st->mem_sw[i]=mem[i];
      
      /* Compute target signal (saturation prevents overflows on clipped input speech) */
      for (i=0;i<st->subframeSize;i++)
         target[i]=EXTRACT16(SATURATE(SUB32(sw[i],PSHR32(ringing[i],1)),32767));

      /* Reset excitation */
      SPEEX_MEMSET(exc, 0, st->subframeSize);

      /* If we have a long-term predictor (otherwise, something's wrong) */
      speex_assert (SUBMODE(ltp_quant));
      {
         int pit_min, pit_max;
         /* Long-term prediction */
         if (SUBMODE(lbr_pitch) != -1)
         {
            /* Low bit-rate pitch handling */
            int margin;
            margin = SUBMODE(lbr_pitch);
            if (margin)
            {
               if (ol_pitch < st->min_pitch+margin-1)
                  ol_pitch=st->min_pitch+margin-1;
               if (ol_pitch > st->max_pitch-margin)
                  ol_pitch=st->max_pitch-margin;
               pit_min = ol_pitch-margin+1;
               pit_max = ol_pitch+margin;
            } else {
               pit_min=pit_max=ol_pitch;
            }
         } else {
            pit_min = st->min_pitch;
            pit_max = st->max_pitch;
         }
         
         /* Force pitch to use only the current frame if needed */
         if (st->bounded_pitch && pit_max>offset)
            pit_max=offset;

         /* Perform pitch search */
         pitch = SUBMODE(ltp_quant)(target, sw, interp_qlpc, bw_lpc1, bw_lpc2,
                                    exc32, SUBMODE(ltp_params), pit_min, pit_max, ol_pitch_coef,
                                    st->lpcSize, st->subframeSize, bits, stack, 
                                    exc, syn_resp, st->complexity, 0, st->plc_tuning, &st->cumul_gain);

         st->pitch[sub]=pitch;
      }
      /* Quantization of innovation */
      SPEEX_MEMSET(innov, 0, st->subframeSize);
      
      /* FIXME: Make sure this is save from overflows (so far so good) */
      for (i=0;i<st->subframeSize;i++)
         real_exc[i] = EXTRACT16(SUB32(EXTEND32(real_exc[i]), PSHR32(exc32[i],SIG_SHIFT-1)));
      
      ener = SHL32(EXTEND32(compute_rms16(real_exc, st->subframeSize)),SIG_SHIFT);
      
      /*FIXME: Should use DIV32_16 and make sure result fits in 16 bits */
#ifdef FIXED_POINT
      {
         spx_word32_t f = PDIV32(ener,PSHR32(ol_gain,SIG_SHIFT));
         if (f<=32767)
            fine_gain = f;
         else
            fine_gain = 32767;
      }
#else
      fine_gain = PDIV32_16(ener,PSHR32(ol_gain,SIG_SHIFT));
#endif
      /* Calculate gain correction for the sub-frame (if any) */
      if (SUBMODE(have_subframe_gain)) 
      {
         int qe;
         if (SUBMODE(have_subframe_gain)==3)
         {
            qe = scal_quant(fine_gain, exc_gain_quant_scal3_bound, 8);
            speex_bits_pack(bits, qe, 3);
            ener=MULT16_32_Q14(exc_gain_quant_scal3[qe],ol_gain);
         } else {
            qe = scal_quant(fine_gain, exc_gain_quant_scal1_bound, 2);
            speex_bits_pack(bits, qe, 1);
            ener=MULT16_32_Q14(exc_gain_quant_scal1[qe],ol_gain);               
         }
      } else {
         ener=ol_gain;
      }
      
      /*printf ("%f %f\n", ener, ol_gain);*/
      
      /* Normalize innovation */
      signal_div(target, target, ener, st->subframeSize);
      
      /* Quantize innovation */
      speex_assert (SUBMODE(innovation_quant));
      {
         /* Codebook search */
         SUBMODE(innovation_quant)(target, interp_qlpc, bw_lpc1, bw_lpc2, 
                  SUBMODE(innovation_params), st->lpcSize, st->subframeSize, 
                  innov, syn_resp, bits, stack, st->complexity, SUBMODE(double_codebook));
         
         /* De-normalize innovation and update excitation */
         signal_mul(innov, innov, ener, st->subframeSize);
         
         for (i=0;i<st->subframeSize;i++)
            exc[i] = EXTRACT16(SATURATE32(PSHR32(ADD32(SHL32(exc32[i],1),innov[i]),SIG_SHIFT),32767));

         /* In some (rare) modes, we do a second search (more bits) to reduce noise even more */
         if (SUBMODE(double_codebook)) {
            char *tmp_stack=stack;
            VARDECL(spx_sig_t *innov2);
            ALLOC(innov2, st->subframeSize, spx_sig_t);
            SPEEX_MEMSET(innov2, 0, st->subframeSize);
            for (i=0;i<st->subframeSize;i++)
               target[i]=MULT16_16_P13(QCONST16(2.2f,13), target[i]);
            SUBMODE(innovation_quant)(target, interp_qlpc, bw_lpc1, bw_lpc2, 
                                      SUBMODE(innovation_params), st->lpcSize, st->subframeSize, 
                                      innov2, syn_resp, bits, stack, st->complexity, 0);
            signal_mul(innov2, innov2, MULT16_32_Q15(QCONST16(0.454545f,15),ener), st->subframeSize);
            for (i=0;i<st->subframeSize;i++)
               innov[i] = ADD32(innov[i],innov2[i]);
            stack = tmp_stack;
         }
         for (i=0;i<st->subframeSize;i++)
            exc[i] = EXTRACT16(SATURATE32(PSHR32(ADD32(SHL32(exc32[i],1),innov[i]),SIG_SHIFT),32767));
         if (st->innov_rms_save)
         {
            st->innov_rms_save[sub] = compute_rms(innov, st->subframeSize);
         }
      }

      /* Final signal synthesis from excitation */
      iir_mem16(exc, interp_qlpc, sw, st->subframeSize, st->lpcSize, st->mem_sp, stack);

      /* Compute weighted signal again, from synthesized speech (not sure it's the right thing) */
      if (st->complexity!=0)
         filter_mem16(sw, bw_lpc1, bw_lpc2, sw, st->subframeSize, st->lpcSize, st->mem_sw, stack);
      
   }

   /* Store the LSPs for interpolation in the next frame */
   if (st->submodeID>=1)
   {
      for (i=0;i<st->lpcSize;i++)
         st->old_lsp[i] = lsp[i];
      for (i=0;i<st->lpcSize;i++)
         st->old_qlsp[i] = qlsp[i];
   }

#ifdef VORBIS_PSYCHO
   if (st->submodeID>=1)
      SPEEX_COPY(st->old_curve, st->curve, 128);
#endif

   if (st->submodeID==1)
   {
#ifndef DISABLE_VBR
      if (st->dtx_count)
         speex_bits_pack(bits, 15, 4);
      else
#endif
         speex_bits_pack(bits, 0, 4);
   }

   /* The next frame will not be the first (Duh!) */
   st->first = 0;
   SPEEX_COPY(st->winBuf, in+2*st->frameSize-st->windowSize, st->windowSize-st->frameSize);

   if (SUBMODE(innovation_quant) == noise_codebook_quant || st->submodeID==0)
      st->bounded_pitch = 1;
   else
      st->bounded_pitch = 0;

   return 1;
}

void *nb_decoder_init(const SpeexMode *m)
{
   DecState *st;
   const SpeexNBMode *mode;
   int i;

   mode=(const SpeexNBMode*)m->mode;
   st = (DecState *)speex_alloc(sizeof(DecState));
   if (!st)
      return NULL;
#if defined(VAR_ARRAYS) || defined (USE_ALLOCA)
   st->stack = NULL;
#else
   st->stack = (char*)speex_alloc_scratch(NB_DEC_STACK);
#endif

   st->mode=m;


   st->encode_submode = 1;

   st->first=1;
   /* Codec parameters, should eventually have several "modes"*/
   st->frameSize = mode->frameSize;
   st->nbSubframes=mode->frameSize/mode->subframeSize;
   st->subframeSize=mode->subframeSize;
   st->lpcSize = mode->lpcSize;
   st->min_pitch=mode->pitchStart;
   st->max_pitch=mode->pitchEnd;

   st->submodes=mode->submodes;
   st->submodeID=mode->defaultSubmode;

   st->lpc_enh_enabled=1;

   st->excBuf = (spx_word16_t*)speex_alloc((st->frameSize + 2*st->max_pitch + st->subframeSize + 12)*sizeof(spx_word16_t));
   st->exc = st->excBuf + 2*st->max_pitch + st->subframeSize + 6;
   SPEEX_MEMSET(st->excBuf, 0, st->frameSize + st->max_pitch);

   st->interp_qlpc = (spx_coef_t*)speex_alloc(st->lpcSize*sizeof(spx_coef_t));
   st->old_qlsp = (spx_lsp_t*)speex_alloc(st->lpcSize*sizeof(spx_lsp_t));
   st->mem_sp = (spx_mem_t*)speex_alloc(st->lpcSize*sizeof(spx_mem_t));
   st->pi_gain = (spx_word32_t*)speex_alloc((st->nbSubframes)*sizeof(spx_word32_t));
   st->last_pitch = 40;
   st->count_lost=0;
   st->pitch_gain_buf[0] = st->pitch_gain_buf[1] = st->pitch_gain_buf[2] = 0;
   st->pitch_gain_buf_idx = 0;
   st->seed = 1000;
   
   st->sampling_rate=8000;
   st->last_ol_gain = 0;

   st->user_callback.func = &speex_default_user_handler;
   st->user_callback.data = NULL;
   for (i=0;i<16;i++)
      st->speex_callbacks[i].func = NULL;

   st->voc_m1=st->voc_m2=st->voc_mean=0;
   st->voc_offset=0;
   st->dtx_enabled=0;
   st->isWideband = 0;
   st->highpass_enabled = 1;

#ifdef ENABLE_VALGRIND
   VALGRIND_MAKE_READABLE(st, NB_DEC_STACK);
#endif
   return st;
}

void nb_decoder_destroy(void *state)
{
   DecState *st;
   st=(DecState*)state;
   
#if !(defined(VAR_ARRAYS) || defined (USE_ALLOCA))
   speex_free_scratch(st->stack);
#endif

   speex_free (st->excBuf);
   speex_free (st->interp_qlpc);
   speex_free (st->old_qlsp);
   speex_free (st->mem_sp);
   speex_free (st->pi_gain);

   speex_free(state);
}

#define median3(a, b, c)	((a) < (b) ? ((b) < (c) ? (b) : ((a) < (c) ? (c) : (a))) : ((c) < (b) ? (b) : ((c) < (a) ? (c) : (a))))

#ifdef FIXED_POINT
const spx_word16_t attenuation[10] = {32767, 31483, 27923, 22861, 17278, 12055, 7764, 4616, 2533, 1283};
#else
const spx_word16_t attenuation[10] = {1., 0.961, 0.852, 0.698, 0.527, 0.368, 0.237, 0.141, 0.077, 0.039};

#endif

static void nb_decode_lost(DecState *st, spx_word16_t *out, char *stack)
{
   int i;
   int pitch_val;
   spx_word16_t pitch_gain;
   spx_word16_t fact;
   spx_word16_t gain_med;
   spx_word16_t innov_gain;
   spx_word16_t noise_gain;
   
   if (st->count_lost<10)
      fact = attenuation[st->count_lost];
   else
      fact = 0;

   gain_med = median3(st->pitch_gain_buf[0], st->pitch_gain_buf[1], st->pitch_gain_buf[2]);
   if (gain_med < st->last_pitch_gain)
      st->last_pitch_gain = gain_med;
   
#ifdef FIXED_POINT
   pitch_gain = st->last_pitch_gain;
   if (pitch_gain>54)
      pitch_gain = 54;
   pitch_gain = SHL16(pitch_gain, 9);
#else   
   pitch_gain = GAIN_SCALING_1*st->last_pitch_gain;
   if (pitch_gain>.85)
      pitch_gain=.85;
#endif
   pitch_gain = MULT16_16_Q15(fact,pitch_gain) + VERY_SMALL;
   /* FIXME: This was rms of innovation (not exc) */
   innov_gain = compute_rms16(st->exc, st->frameSize);
   noise_gain = MULT16_16_Q15(innov_gain, MULT16_16_Q15(fact, SUB16(Q15ONE,MULT16_16_Q15(pitch_gain,pitch_gain))));
   /* Shift all buffers by one frame */
   SPEEX_MOVE(st->excBuf, st->excBuf+st->frameSize, 2*st->max_pitch + st->subframeSize + 12);
   

   pitch_val = st->last_pitch + SHR32((spx_int32_t)speex_rand(1+st->count_lost, &st->seed),SIG_SHIFT);
   if (pitch_val > st->max_pitch)
      pitch_val = st->max_pitch;
   if (pitch_val < st->min_pitch)
      pitch_val = st->min_pitch;
   for (i=0;i<st->frameSize;i++)
   {
      st->exc[i]= MULT16_16_Q15(pitch_gain, (st->exc[i-pitch_val]+VERY_SMALL)) + 
            speex_rand(noise_gain, &st->seed);
   }

   bw_lpc(QCONST16(.98,15), st->interp_qlpc, st->interp_qlpc, st->lpcSize);
   iir_mem16(&st->exc[-st->subframeSize], st->interp_qlpc, out, st->frameSize,
             st->lpcSize, st->mem_sp, stack);
   highpass(out, out, st->frameSize, HIGHPASS_NARROWBAND|HIGHPASS_OUTPUT, st->mem_hp);
   
   st->first = 0;
   st->count_lost++;
   st->pitch_gain_buf[st->pitch_gain_buf_idx++] = PSHR16(pitch_gain,9);
   if (st->pitch_gain_buf_idx > 2) /* rollover */
      st->pitch_gain_buf_idx = 0;
}

/* Just so we don't need to carry the complete wideband mode information */
static const int wb_skip_table[8] = {0, 36, 112, 192, 352, 0, 0, 0};
   
int nb_decode(void *state, SpeexBits *bits, void *vout)
{
   DecState *st;
   int i, sub;
   int pitch;
   spx_word16_t pitch_gain[3];
   spx_word32_t ol_gain=0;
   int ol_pitch=0;
   spx_word16_t ol_pitch_coef=0;
   int best_pitch=40;
   spx_word16_t best_pitch_gain=0;
   int wideband;
   int m;
   char *stack;
   VARDECL(spx_sig_t *innov);
   VARDECL(spx_word32_t *exc32);
   VARDECL(spx_coef_t *ak);
   VARDECL(spx_lsp_t *qlsp);
   spx_word16_t pitch_average=0;
   
   spx_word16_t *out = (spx_word16_t*)vout;
   VARDECL(spx_lsp_t *interp_qlsp);

   st=(DecState*)state;
   stack=st->stack;

   /* Check if we're in DTX mode*/
   if (!bits && st->dtx_enabled)
   {
      st->submodeID=0;
   } else 
   {
      /* If bits is NULL, consider the packet to be lost (what could we do anyway) */
      if (!bits)
      {
         nb_decode_lost(st, out, stack);
         return 0;
      }

      if (st->encode_submode)
      {

      /* Search for next narrowband block (handle requests, skip wideband blocks) */
      do {
         if (speex_bits_remaining(bits)<5)
            return -1;
         wideband = speex_bits_unpack_unsigned(bits, 1);
         if (wideband) /* Skip wideband block (for compatibility) */
         {
            int submode;
            int advance;
            advance = submode = speex_bits_unpack_unsigned(bits, SB_SUBMODE_BITS);
            /*speex_mode_query(&speex_wb_mode, SPEEX_SUBMODE_BITS_PER_FRAME, &advance);*/
            advance = wb_skip_table[submode];
            if (advance < 0)
            {
               speex_notify("Invalid mode encountered. The stream is corrupted.");
               return -2;
            } 
            advance -= (SB_SUBMODE_BITS+1);
            speex_bits_advance(bits, advance);
            
            if (speex_bits_remaining(bits)<5)
               return -1;
            wideband = speex_bits_unpack_unsigned(bits, 1);
            if (wideband)
            {
               advance = submode = speex_bits_unpack_unsigned(bits, SB_SUBMODE_BITS);
               /*speex_mode_query(&speex_wb_mode, SPEEX_SUBMODE_BITS_PER_FRAME, &advance);*/
               advance = wb_skip_table[submode];
               if (advance < 0)
               {
                  speex_notify("Invalid mode encountered. The stream is corrupted.");
                  return -2;
               } 
               advance -= (SB_SUBMODE_BITS+1);
               speex_bits_advance(bits, advance);
               wideband = speex_bits_unpack_unsigned(bits, 1);
               if (wideband)
               {
                  speex_notify("More than two wideband layers found. The stream is corrupted.");
                  return -2;
               }

            }
         }
         if (speex_bits_remaining(bits)<4)
            return -1;
         /* FIXME: Check for overflow */
         m = speex_bits_unpack_unsigned(bits, 4);
         if (m==15) /* We found a terminator */
         {
            return -1;
         } else if (m==14) /* Speex in-band request */
         {
            int ret = speex_inband_handler(bits, st->speex_callbacks, state);
            if (ret)
               return ret;
         } else if (m==13) /* User in-band request */
         {
            int ret = st->user_callback.func(bits, state, st->user_callback.data);
            if (ret)
               return ret;
         } else if (m>8) /* Invalid mode */
         {
            speex_notify("Invalid mode encountered. The stream is corrupted.");
            return -2;
         }
      
      } while (m>8);

      /* Get the sub-mode that was used */
      st->submodeID = m;
      }

   }

   /* Shift all buffers by one frame */
   SPEEX_MOVE(st->excBuf, st->excBuf+st->frameSize, 2*st->max_pitch + st->subframeSize + 12);

   /* If null mode (no transmission), just set a couple things to zero*/
   if (st->submodes[st->submodeID] == NULL)
   {
      VARDECL(spx_coef_t *lpc);
      ALLOC(lpc, st->lpcSize, spx_coef_t);
      bw_lpc(QCONST16(0.93f,15), st->interp_qlpc, lpc, st->lpcSize);
      {
         spx_word16_t innov_gain=0;
         /* FIXME: This was innov, not exc */
         innov_gain = compute_rms16(st->exc, st->frameSize);
         for (i=0;i<st->frameSize;i++)
            st->exc[i]=speex_rand(innov_gain, &st->seed);
      }


      st->first=1;

      /* Final signal synthesis from excitation */
      iir_mem16(st->exc, lpc, out, st->frameSize, st->lpcSize, st->mem_sp, stack);

      st->count_lost=0;
      return 0;
   }

   ALLOC(qlsp, st->lpcSize, spx_lsp_t);

   /* Unquantize LSPs */
   SUBMODE(lsp_unquant)(qlsp, st->lpcSize, bits);

   /*Damp memory if a frame was lost and the LSP changed too much*/
   if (st->count_lost)
   {
      spx_word16_t fact;
      spx_word32_t lsp_dist=0;
      for (i=0;i<st->lpcSize;i++)
         lsp_dist = ADD32(lsp_dist, EXTEND32(ABS(st->old_qlsp[i] - qlsp[i])));
#ifdef FIXED_POINT
      fact = SHR16(19661,SHR32(lsp_dist,LSP_SHIFT+2));      
#else
      fact = .6*exp(-.2*lsp_dist);
#endif
      for (i=0;i<st->lpcSize;i++)
         st->mem_sp[i] = MULT16_32_Q15(fact,st->mem_sp[i]);
   }


   /* Handle first frame and lost-packet case */
   if (st->first || st->count_lost)
   {
      for (i=0;i<st->lpcSize;i++)
         st->old_qlsp[i] = qlsp[i];
   }

   /* Get open-loop pitch estimation for low bit-rate pitch coding */
   if (SUBMODE(lbr_pitch)!=-1)
   {
      ol_pitch = st->min_pitch+speex_bits_unpack_unsigned(bits, 7);
   } 
   
   if (SUBMODE(forced_pitch_gain))
   {
      int quant;
      quant = speex_bits_unpack_unsigned(bits, 4);
      ol_pitch_coef=MULT16_16_P15(QCONST16(0.066667,15),SHL16(quant,GAIN_SHIFT));
   }
   
   /* Get global excitation gain */
   {
      int qe;
      qe = speex_bits_unpack_unsigned(bits, 5);
#ifdef FIXED_POINT
      /* FIXME: Perhaps we could slightly lower the gain here when the output is going to saturate? */
      ol_gain = MULT16_32_Q15(28406,ol_gain_table[qe]);
#else
      ol_gain = SIG_SCALING*exp(qe/3.5);
#endif
   }

   ALLOC(ak, st->lpcSize, spx_coef_t);
   ALLOC(innov, st->subframeSize, spx_sig_t);
   ALLOC(exc32, st->subframeSize, spx_word32_t);

   if (st->submodeID==1)
   {
      int extra;
      extra = speex_bits_unpack_unsigned(bits, 4);

      if (extra==15)
         st->dtx_enabled=1;
      else
         st->dtx_enabled=0;
   }
   if (st->submodeID>1)
      st->dtx_enabled=0;

   /*Loop on subframes */
   for (sub=0;sub<st->nbSubframes;sub++)
   {
      int offset;
      spx_word16_t *exc;
      spx_word16_t *sp;
      spx_word16_t *innov_save = NULL;
      spx_word16_t tmp;

      /* Offset relative to start of frame */
      offset = st->subframeSize*sub;
      /* Excitation */
      exc=st->exc+offset;
      /* Original signal */
      sp=out+offset;
      if (st->innov_save)
         innov_save = st->innov_save+offset;


      /* Reset excitation */
      SPEEX_MEMSET(exc, 0, st->subframeSize);

      /*Adaptive codebook contribution*/
      speex_assert (SUBMODE(ltp_unquant));
      {
         int pit_min, pit_max;
         /* Handle pitch constraints if any */
         if (SUBMODE(lbr_pitch) != -1)
         {
            int margin;
            margin = SUBMODE(lbr_pitch);
            if (margin)
            {
/* GT - need optimization?
               if (ol_pitch < st->min_pitch+margin-1)
                  ol_pitch=st->min_pitch+margin-1;
               if (ol_pitch > st->max_pitch-margin)
                  ol_pitch=st->max_pitch-margin;
               pit_min = ol_pitch-margin+1;
               pit_max = ol_pitch+margin;
*/
               pit_min = ol_pitch-margin+1;
               if (pit_min < st->min_pitch)
		  pit_min = st->min_pitch;
               pit_max = ol_pitch+margin;
               if (pit_max > st->max_pitch)
		  pit_max = st->max_pitch;
            } else {
               pit_min = pit_max = ol_pitch;
            }
         } else {
            pit_min = st->min_pitch;
            pit_max = st->max_pitch;
         }



         SUBMODE(ltp_unquant)(exc, exc32, pit_min, pit_max, ol_pitch_coef, SUBMODE(ltp_params), 
                 st->subframeSize, &pitch, &pitch_gain[0], bits, stack, 
                 st->count_lost, offset, st->last_pitch_gain, 0);

         /* Ensuring that things aren't blowing up as would happen if e.g. an encoder is 
         crafting packets to make us produce NaNs and slow down the decoder (vague DoS threat).
         We can probably be even more aggressive and limit to 15000 or so. */
         sanitize_values32(exc32, NEG32(QCONST32(32000,SIG_SHIFT-1)), QCONST32(32000,SIG_SHIFT-1), st->subframeSize);
         
         tmp = gain_3tap_to_1tap(pitch_gain);

         pitch_average += tmp;
         if ((tmp>best_pitch_gain&&ABS(2*best_pitch-pitch)>=3&&ABS(3*best_pitch-pitch)>=4&&ABS(4*best_pitch-pitch)>=5) 
              || (tmp>MULT16_16_Q15(QCONST16(.6,15),best_pitch_gain)&&(ABS(best_pitch-2*pitch)<3||ABS(best_pitch-3*pitch)<4||ABS(best_pitch-4*pitch)<5)) 
              || (MULT16_16_Q15(QCONST16(.67,15),tmp)>best_pitch_gain&&(ABS(2*best_pitch-pitch)<3||ABS(3*best_pitch-pitch)<4||ABS(4*best_pitch-pitch)<5)) )
         {
            best_pitch = pitch;
            if (tmp > best_pitch_gain)
               best_pitch_gain = tmp;
         }
      }
      
      /* Unquantize the innovation */
      {
         int q_energy;
         spx_word32_t ener;
         
         SPEEX_MEMSET(innov, 0, st->subframeSize);

         /* Decode sub-frame gain correction */
         if (SUBMODE(have_subframe_gain)==3)
         {
            q_energy = speex_bits_unpack_unsigned(bits, 3);
            ener = MULT16_32_Q14(exc_gain_quant_scal3[q_energy],ol_gain);
         } else if (SUBMODE(have_subframe_gain)==1)
         {
            q_energy = speex_bits_unpack_unsigned(bits, 1);
            ener = MULT16_32_Q14(exc_gain_quant_scal1[q_energy],ol_gain);
         } else {
            ener = ol_gain;
         }
                  
         speex_assert (SUBMODE(innovation_unquant));
         {
            /*Fixed codebook contribution*/
            SUBMODE(innovation_unquant)(innov, SUBMODE(innovation_params), st->subframeSize, bits, stack, &st->seed);
            /* De-normalize innovation and update excitation */

            signal_mul(innov, innov, ener, st->subframeSize);

            /* Decode second codebook (only for some modes) */
            if (SUBMODE(double_codebook))
            {
               char *tmp_stack=stack;
               VARDECL(spx_sig_t *innov2);
               ALLOC(innov2, st->subframeSize, spx_sig_t);
               SPEEX_MEMSET(innov2, 0, st->subframeSize);
               SUBMODE(innovation_unquant)(innov2, SUBMODE(innovation_params), st->subframeSize, bits, stack, &st->seed);
               signal_mul(innov2, innov2, MULT16_32_Q15(QCONST16(0.454545f,15),ener), st->subframeSize);
               for (i=0;i<st->subframeSize;i++)
                  innov[i] = ADD32(innov[i], innov2[i]);
               stack = tmp_stack;
            }
            for (i=0;i<st->subframeSize;i++)
               exc[i]=EXTRACT16(SATURATE32(PSHR32(ADD32(SHL32(exc32[i],1),innov[i]),SIG_SHIFT),32767));
            /*print_vec(exc, 40, "innov");*/
            if (innov_save)
            {
               for (i=0;i<st->subframeSize;i++)
                  innov_save[i] = EXTRACT16(PSHR32(innov[i], SIG_SHIFT));
            }
         }

         /*Vocoder mode*/
         if (st->submodeID==1) 
         {
            spx_word16_t g=ol_pitch_coef;
            g=MULT16_16_P14(QCONST16(1.5f,14),(g-QCONST16(.2f,6)));
            if (g<0)
               g=0;
            if (g>GAIN_SCALING)
               g=GAIN_SCALING;
            
            SPEEX_MEMSET(exc, 0, st->subframeSize);
            while (st->voc_offset<st->subframeSize)
            {
               /* exc[st->voc_offset]= g*sqrt(2*ol_pitch)*ol_gain;
                  Not quite sure why we need the factor of two in the sqrt */
               if (st->voc_offset>=0)
                  exc[st->voc_offset]=MULT16_16(spx_sqrt(MULT16_16_16(2,ol_pitch)),EXTRACT16(PSHR32(MULT16_16(g,PSHR32(ol_gain,SIG_SHIFT)),6)));
               st->voc_offset+=ol_pitch;
            }
            st->voc_offset -= st->subframeSize;
            
            for (i=0;i<st->subframeSize;i++)
            {
               spx_word16_t exci=exc[i];
               exc[i]= ADD16(ADD16(MULT16_16_Q15(QCONST16(.7f,15),exc[i]) , MULT16_16_Q15(QCONST16(.3f,15),st->voc_m1)),
                             SUB16(MULT16_16_Q15(Q15_ONE-MULT16_16_16(QCONST16(.85f,9),g),EXTRACT16(PSHR32(innov[i],SIG_SHIFT))),
                                   MULT16_16_Q15(MULT16_16_16(QCONST16(.15f,9),g),EXTRACT16(PSHR32(st->voc_m2,SIG_SHIFT)))
                                  ));
               st->voc_m1 = exci;
               st->voc_m2=innov[i];
               st->voc_mean = EXTRACT16(PSHR32(ADD32(MULT16_16(QCONST16(.8f,15),st->voc_mean), MULT16_16(QCONST16(.2f,15),exc[i])), 15));
               exc[i]-=st->voc_mean;
            }
         }

      }
   }
   
   ALLOC(interp_qlsp, st->lpcSize, spx_lsp_t);

   if (st->lpc_enh_enabled && SUBMODE(comb_gain)>0 && !st->count_lost)
   {
      multicomb(st->exc-st->subframeSize, out, st->interp_qlpc, st->lpcSize, 2*st->subframeSize, best_pitch, 40, SUBMODE(comb_gain), stack);
      multicomb(st->exc+st->subframeSize, out+2*st->subframeSize, st->interp_qlpc, st->lpcSize, 2*st->subframeSize, best_pitch, 40, SUBMODE(comb_gain), stack);
   } else {
      SPEEX_COPY(out, &st->exc[-st->subframeSize], st->frameSize);
   }
   
   /* If the last packet was lost, re-scale the excitation to obtain the same energy as encoded in ol_gain */
   if (st->count_lost) 
   {
      spx_word16_t exc_ener;
      spx_word32_t gain32;
      spx_word16_t gain;
      exc_ener = compute_rms16 (st->exc, st->frameSize);
      gain32 = PDIV32(ol_gain, ADD16(exc_ener,1));
#ifdef FIXED_POINT
      if (gain32 > 32767)
         gain32 = 32767;
      gain = EXTRACT16(gain32);
#else
      if (gain32 > 2)
         gain32=2;
      gain = gain32;
#endif
      for (i=0;i<st->frameSize;i++)
      {
         st->exc[i] = MULT16_16_Q14(gain, st->exc[i]);
         out[i]=st->exc[i-st->subframeSize];
      }
   }

   /*Loop on subframes */
   for (sub=0;sub<st->nbSubframes;sub++)
   {
      int offset;
      spx_word16_t *sp;
      spx_word16_t *exc;
      /* Offset relative to start of frame */
      offset = st->subframeSize*sub;
      /* Original signal */
      sp=out+offset;
      /* Excitation */
      exc=st->exc+offset;

      /* LSP interpolation (quantized and unquantized) */
      lsp_interpolate(st->old_qlsp, qlsp, interp_qlsp, st->lpcSize, sub, st->nbSubframes);

      /* Make sure the LSP's are stable */
      lsp_enforce_margin(interp_qlsp, st->lpcSize, LSP_MARGIN);

      /* Compute interpolated LPCs (unquantized) */
      lsp_to_lpc(interp_qlsp, ak, st->lpcSize, stack);

      /* Compute analysis filter at w=pi */
      {
         spx_word32_t pi_g=LPC_SCALING;
         for (i=0;i<st->lpcSize;i+=2)
         {
            /*pi_g += -st->interp_qlpc[i] +  st->interp_qlpc[i+1];*/
            pi_g = ADD32(pi_g, SUB32(EXTEND32(ak[i+1]),EXTEND32(ak[i])));
         }
         st->pi_gain[sub] = pi_g;
      }
      
      iir_mem16(sp, st->interp_qlpc, sp, st->subframeSize, st->lpcSize, 
                st->mem_sp, stack);
      
      for (i=0;i<st->lpcSize;i++)
         st->interp_qlpc[i] = ak[i];

   }

   if (st->highpass_enabled)
      highpass(out, out, st->frameSize, (st->isWideband?HIGHPASS_WIDEBAND:HIGHPASS_NARROWBAND)|HIGHPASS_OUTPUT, st->mem_hp);
   /*for (i=0;i<st->frameSize;i++)
     printf ("%d\n", (int)st->frame[i]);*/

   /* Tracking output level */
   st->level = 1+PSHR32(ol_gain,SIG_SHIFT);
   st->max_level = MAX16(MULT16_16_Q15(QCONST16(.99f,15), st->max_level), st->level);
   st->min_level = MIN16(ADD16(1,MULT16_16_Q14(QCONST16(1.01f,14), st->min_level)), st->level);
   if (st->max_level < st->min_level+1)
      st->max_level = st->min_level+1;
   /*printf ("%f %f %f %d\n", og, st->min_level, st->max_level, update);*/
   
   /* Store the LSPs for interpolation in the next frame */
   for (i=0;i<st->lpcSize;i++)
      st->old_qlsp[i] = qlsp[i];

   /* The next frame will not be the first (Duh!) */
   st->first = 0;
   st->count_lost=0;
   st->last_pitch = best_pitch;
#ifdef FIXED_POINT
   st->last_pitch_gain = PSHR16(pitch_average,2);
#else
   st->last_pitch_gain = .25*pitch_average;   
#endif
   st->pitch_gain_buf[st->pitch_gain_buf_idx++] = st->last_pitch_gain;
   if (st->pitch_gain_buf_idx > 2) /* rollover */
      st->pitch_gain_buf_idx = 0;

   st->last_ol_gain = ol_gain;

   return 0;
}

int nb_encoder_ctl(void *state, int request, void *ptr)
{
   EncState *st;
   st=(EncState*)state;     
   switch(request)
   {
   case SPEEX_GET_FRAME_SIZE:
      (*(spx_int32_t*)ptr) = st->frameSize;
      break;
   case SPEEX_SET_LOW_MODE:
   case SPEEX_SET_MODE:
      st->submodeSelect = st->submodeID = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_GET_LOW_MODE:
   case SPEEX_GET_MODE:
      (*(spx_int32_t*)ptr) = st->submodeID;
      break;
#ifndef DISABLE_VBR
      case SPEEX_SET_VBR:
      st->vbr_enabled = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_GET_VBR:
      (*(spx_int32_t*)ptr) = st->vbr_enabled;
      break;
   case SPEEX_SET_VAD:
      st->vad_enabled = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_GET_VAD:
      (*(spx_int32_t*)ptr) = st->vad_enabled;
      break;
   case SPEEX_SET_DTX:
      st->dtx_enabled = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_GET_DTX:
      (*(spx_int32_t*)ptr) = st->dtx_enabled;
      break;
   case SPEEX_SET_ABR:
      st->abr_enabled = (*(spx_int32_t*)ptr);
      st->vbr_enabled = st->abr_enabled!=0;
      if (st->vbr_enabled) 
      {
         spx_int32_t i=10;
         spx_int32_t rate, target;
         float vbr_qual;
         target = (*(spx_int32_t*)ptr);
         while (i>=0)
         {
            speex_encoder_ctl(st, SPEEX_SET_QUALITY, &i);
            speex_encoder_ctl(st, SPEEX_GET_BITRATE, &rate);
            if (rate <= target)
               break;
            i--;
         }
         vbr_qual=i;
         if (vbr_qual<0)
            vbr_qual=0;
         speex_encoder_ctl(st, SPEEX_SET_VBR_QUALITY, &vbr_qual);
         st->abr_count=0;
         st->abr_drift=0;
         st->abr_drift2=0;
      }
      
      break;
   case SPEEX_GET_ABR:
      (*(spx_int32_t*)ptr) = st->abr_enabled;
      break;
#endif /* #ifndef DISABLE_VBR */
#if !defined(DISABLE_VBR) && !defined(DISABLE_FLOAT_API)
   case SPEEX_SET_VBR_QUALITY:
      st->vbr_quality = (*(float*)ptr);
      break;
   case SPEEX_GET_VBR_QUALITY:
      (*(float*)ptr) = st->vbr_quality;
      break;
#endif /* !defined(DISABLE_VBR) && !defined(DISABLE_FLOAT_API) */
   case SPEEX_SET_QUALITY:
      {
         int quality = (*(spx_int32_t*)ptr);
         if (quality < 0)
            quality = 0;
         if (quality > 10)
            quality = 10;
         st->submodeSelect = st->submodeID = ((const SpeexNBMode*)(st->mode->mode))->quality_map[quality];
      }
      break;
   case SPEEX_SET_COMPLEXITY:
      st->complexity = (*(spx_int32_t*)ptr);
      if (st->complexity<0)
         st->complexity=0;
      break;
   case SPEEX_GET_COMPLEXITY:
      (*(spx_int32_t*)ptr) = st->complexity;
      break;
   case SPEEX_SET_BITRATE:
      {
         spx_int32_t i=10;
         spx_int32_t rate, target;
         target = (*(spx_int32_t*)ptr);
         while (i>=0)
         {
            speex_encoder_ctl(st, SPEEX_SET_QUALITY, &i);
            speex_encoder_ctl(st, SPEEX_GET_BITRATE, &rate);
            if (rate <= target)
               break;
            i--;
         }
      }
      break;
   case SPEEX_GET_BITRATE:
      if (st->submodes[st->submodeID])
         (*(spx_int32_t*)ptr) = st->sampling_rate*SUBMODE(bits_per_frame)/st->frameSize;
      else
         (*(spx_int32_t*)ptr) = st->sampling_rate*(NB_SUBMODE_BITS+1)/st->frameSize;
      break;
   case SPEEX_SET_SAMPLING_RATE:
      st->sampling_rate = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_GET_SAMPLING_RATE:
      (*(spx_int32_t*)ptr)=st->sampling_rate;
      break;
   case SPEEX_RESET_STATE:
      {
         int i;
         st->bounded_pitch = 1;
         st->first = 1;
         for (i=0;i<st->lpcSize;i++)
            st->old_lsp[i]= DIV32(MULT16_16(QCONST16(3.1415927f, LSP_SHIFT), i+1), st->lpcSize+1);
         for (i=0;i<st->lpcSize;i++)
            st->mem_sw[i]=st->mem_sw_whole[i]=st->mem_sp[i]=st->mem_exc[i]=0;
         for (i=0;i<st->frameSize+st->max_pitch+1;i++)
            st->excBuf[i]=st->swBuf[i]=0;
         for (i=0;i<st->windowSize-st->frameSize;i++)
            st->winBuf[i]=0;
      }
      break;
   case SPEEX_SET_SUBMODE_ENCODING:
      st->encode_submode = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_GET_SUBMODE_ENCODING:
      (*(spx_int32_t*)ptr) = st->encode_submode;
      break;
   case SPEEX_GET_LOOKAHEAD:
      (*(spx_int32_t*)ptr)=(st->windowSize-st->frameSize);
      break;
   case SPEEX_SET_PLC_TUNING:
      st->plc_tuning = (*(spx_int32_t*)ptr);
      if (st->plc_tuning>100)
         st->plc_tuning=100;
      break;
   case SPEEX_GET_PLC_TUNING:
      (*(spx_int32_t*)ptr)=(st->plc_tuning);
      break;
#ifndef DISABLE_VBR
   case SPEEX_SET_VBR_MAX_BITRATE:
      st->vbr_max = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_GET_VBR_MAX_BITRATE:
      (*(spx_int32_t*)ptr) = st->vbr_max;
      break;
#endif /* #ifndef DISABLE_VBR */
   case SPEEX_SET_HIGHPASS:
      st->highpass_enabled = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_GET_HIGHPASS:
      (*(spx_int32_t*)ptr) = st->highpass_enabled;
      break;

   /* This is all internal stuff past this point */
   case SPEEX_GET_PI_GAIN:
      {
         int i;
         spx_word32_t *g = (spx_word32_t*)ptr;
         for (i=0;i<st->nbSubframes;i++)
            g[i]=st->pi_gain[i];
      }
      break;
   case SPEEX_GET_EXC:
      {
         int i;
         for (i=0;i<st->nbSubframes;i++)
            ((spx_word16_t*)ptr)[i] = compute_rms16(st->exc+i*st->subframeSize, st->subframeSize);
      }
      break;
#ifndef DISABLE_VBR
   case SPEEX_GET_RELATIVE_QUALITY:
      (*(float*)ptr)=st->relative_quality;
      break;
#endif /* #ifndef DISABLE_VBR */
   case SPEEX_SET_INNOVATION_SAVE:
      st->innov_rms_save = (spx_word16_t*)ptr;
      break;
   case SPEEX_SET_WIDEBAND:
      st->isWideband = *((spx_int32_t*)ptr);
      break;
   case SPEEX_GET_STACK:
      *((char**)ptr) = st->stack;
      break;
   default:
      speex_warning_int("Unknown nb_ctl request: ", request);
      return -1;
   }
   return 0;
}

int nb_decoder_ctl(void *state, int request, void *ptr)
{
   DecState *st;
   st=(DecState*)state;
   switch(request)
   {
   case SPEEX_SET_LOW_MODE:
   case SPEEX_SET_MODE:
      st->submodeID = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_GET_LOW_MODE:
   case SPEEX_GET_MODE:
      (*(spx_int32_t*)ptr) = st->submodeID;
      break;
   case SPEEX_SET_ENH:
      st->lpc_enh_enabled = *((spx_int32_t*)ptr);
      break;
   case SPEEX_GET_ENH:
      *((spx_int32_t*)ptr) = st->lpc_enh_enabled;
      break;
   case SPEEX_GET_FRAME_SIZE:
      (*(spx_int32_t*)ptr) = st->frameSize;
      break;
   case SPEEX_GET_BITRATE:
      if (st->submodes[st->submodeID])
         (*(spx_int32_t*)ptr) = st->sampling_rate*SUBMODE(bits_per_frame)/st->frameSize;
      else
         (*(spx_int32_t*)ptr) = st->sampling_rate*(NB_SUBMODE_BITS+1)/st->frameSize;
      break;
   case SPEEX_SET_SAMPLING_RATE:
      st->sampling_rate = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_GET_SAMPLING_RATE:
      (*(spx_int32_t*)ptr)=st->sampling_rate;
      break;
   case SPEEX_SET_HANDLER:
      {
         SpeexCallback *c = (SpeexCallback*)ptr;
         st->speex_callbacks[c->callback_id].func=c->func;
         st->speex_callbacks[c->callback_id].data=c->data;
         st->speex_callbacks[c->callback_id].callback_id=c->callback_id;
      }
      break;
   case SPEEX_SET_USER_HANDLER:
      {
         SpeexCallback *c = (SpeexCallback*)ptr;
         st->user_callback.func=c->func;
         st->user_callback.data=c->data;
         st->user_callback.callback_id=c->callback_id;
      }
      break;
   case SPEEX_RESET_STATE:
      {
         int i;
         for (i=0;i<st->lpcSize;i++)
            st->mem_sp[i]=0;
         for (i=0;i<st->frameSize + st->max_pitch + 1;i++)
            st->excBuf[i]=0;
      }
      break;
   case SPEEX_SET_SUBMODE_ENCODING:
      st->encode_submode = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_GET_SUBMODE_ENCODING:
      (*(spx_int32_t*)ptr) = st->encode_submode;
      break;
   case SPEEX_GET_LOOKAHEAD:
      (*(spx_int32_t*)ptr)=st->subframeSize;
      break;
   case SPEEX_SET_HIGHPASS:
      st->highpass_enabled = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_GET_HIGHPASS:
      (*(spx_int32_t*)ptr) = st->highpass_enabled;
      break;
      /* FIXME: Convert to fixed-point and re-enable even when float API is disabled */
#ifndef DISABLE_FLOAT_API
   case SPEEX_GET_ACTIVITY:
   {
      float ret;
      ret = log(st->level/st->min_level)/log(st->max_level/st->min_level);
      if (ret>1)
         ret = 1;
      /* Done in a strange way to catch NaNs as well */
      if (!(ret > 0))
         ret = 0;
      /*printf ("%f %f %f %f\n", st->level, st->min_level, st->max_level, ret);*/
      (*(spx_int32_t*)ptr) = (int)(100*ret);
   }
   break;
#endif
   case SPEEX_GET_PI_GAIN:
      {
         int i;
         spx_word32_t *g = (spx_word32_t*)ptr;
         for (i=0;i<st->nbSubframes;i++)
            g[i]=st->pi_gain[i];
      }
      break;
   case SPEEX_GET_EXC:
      {
         int i;
         for (i=0;i<st->nbSubframes;i++)
            ((spx_word16_t*)ptr)[i] = compute_rms16(st->exc+i*st->subframeSize, st->subframeSize);
      }
      break;
   case SPEEX_GET_DTX_STATUS:
      *((spx_int32_t*)ptr) = st->dtx_enabled;
      break;
   case SPEEX_SET_INNOVATION_SAVE:
      st->innov_save = (spx_word16_t*)ptr;
      break;
   case SPEEX_SET_WIDEBAND:
      st->isWideband = *((spx_int32_t*)ptr);
      break;
   case SPEEX_GET_STACK:
      *((char**)ptr) = st->stack;
      break;
   default:
      speex_warning_int("Unknown nb_ctl request: ", request);
      return -1;
   }
   return 0;
}
