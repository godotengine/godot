/* Copyright (C) 2002-2006 Jean-Marc Valin 
   File: sb_celp.c

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
#include "sb_celp.h"
#include "filters.h"
#include "lpc.h"
#include "lsp.h"
#include "stack_alloc.h"
#include "cb_search.h"
#include "quant_lsp.h"
#include "vq.h"
#include "ltp.h"
#include "arch.h"
#include "math_approx.h"
#include "os_support.h"

#ifndef NULL
#define NULL 0
#endif

/* Default size for the encoder and decoder stack (can be changed at compile time).
   This does not apply when using variable-size arrays or alloca. */
#ifndef SB_ENC_STACK
#define SB_ENC_STACK (10000*sizeof(spx_sig_t))
#endif

#ifndef SB_DEC_STACK
#define SB_DEC_STACK (6000*sizeof(spx_sig_t))
#endif


#ifdef DISABLE_WIDEBAND
void *sb_encoder_init(const SpeexMode *m)
{
   speex_fatal("Wideband and Ultra-wideband are disabled");
   return NULL;
}
void sb_encoder_destroy(void *state)
{
   speex_fatal("Wideband and Ultra-wideband are disabled");
}
int sb_encode(void *state, void *vin, SpeexBits *bits)
{
   speex_fatal("Wideband and Ultra-wideband are disabled");
   return -2;
}
void *sb_decoder_init(const SpeexMode *m)
{
   speex_fatal("Wideband and Ultra-wideband are disabled");
   return NULL;
}
void sb_decoder_destroy(void *state)
{
   speex_fatal("Wideband and Ultra-wideband are disabled");
}
int sb_decode(void *state, SpeexBits *bits, void *vout)
{
   speex_fatal("Wideband and Ultra-wideband are disabled");
   return -2;
}
int sb_encoder_ctl(void *state, int request, void *ptr)
{
   speex_fatal("Wideband and Ultra-wideband are disabled");
   return -2;
}
int sb_decoder_ctl(void *state, int request, void *ptr)
{
   speex_fatal("Wideband and Ultra-wideband are disabled");
   return -2;
}
#else


#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif

#define sqr(x) ((x)*(x))

#define SUBMODE(x) st->submodes[st->submodeID]->x

#ifdef FIXED_POINT
static const spx_word16_t gc_quant_bound[16] = {125, 164, 215, 282, 370, 484, 635, 832, 1090, 1428, 1871, 2452, 3213, 4210, 5516, 7228};
static const spx_word16_t fold_quant_bound[32] = {
   39, 44, 50, 57, 64, 73, 83, 94,
   106, 120, 136, 154, 175, 198, 225, 255,
   288, 327, 370, 420, 476, 539, 611, 692,
   784, 889, 1007, 1141, 1293, 1465, 1660, 1881};
#define LSP_MARGIN 410
#define LSP_DELTA1 6553
#define LSP_DELTA2 1638

#else

static const spx_word16_t gc_quant_bound[16] = {
      0.97979, 1.28384, 1.68223, 2.20426, 2.88829, 3.78458, 4.95900, 6.49787, 
      8.51428, 11.15642, 14.61846, 19.15484, 25.09895, 32.88761, 43.09325, 56.46588};
static const spx_word16_t fold_quant_bound[32] = {
   0.30498, 0.34559, 0.39161, 0.44375, 0.50283, 0.56979, 0.64565, 0.73162,
   0.82903, 0.93942, 1.06450, 1.20624, 1.36685, 1.54884, 1.75506, 1.98875,
   2.25355, 2.55360, 2.89361, 3.27889, 3.71547, 4.21018, 4.77076, 5.40598,
   6.12577, 6.94141, 7.86565, 8.91295, 10.09969, 11.44445, 12.96826, 14.69497};

#define LSP_MARGIN .05
#define LSP_DELTA1 .2
#define LSP_DELTA2 .05

#endif

#define QMF_ORDER 64

#ifdef FIXED_POINT
static const spx_word16_t h0[64] = {2, -7, -7, 18, 15, -39, -25, 75, 35, -130, -41, 212, 38, -327, -17, 483, -32, -689, 124, 956, -283, -1307, 543, 1780, -973, -2467, 1733, 3633, -3339, -6409, 9059, 30153, 30153, 9059, -6409, -3339, 3633, 1733, -2467, -973, 1780, 543, -1307, -283, 956, 124, -689, -32, 483, -17, -327, 38, 212, -41, -130, 35, 75, -25, -39, 15, 18, -7, -7, 2};

#else
static const float h0[64] = {
   3.596189e-05f, -0.0001123515f,
   -0.0001104587f, 0.0002790277f,
   0.0002298438f, -0.0005953563f,
   -0.0003823631f, 0.00113826f,
   0.0005308539f, -0.001986177f,
   -0.0006243724f, 0.003235877f,
   0.0005743159f, -0.004989147f,
   -0.0002584767f, 0.007367171f,
   -0.0004857935f, -0.01050689f,
   0.001894714f, 0.01459396f,
   -0.004313674f, -0.01994365f,
   0.00828756f, 0.02716055f,
   -0.01485397f, -0.03764973f,
   0.026447f, 0.05543245f,
   -0.05095487f, -0.09779096f,
   0.1382363f, 0.4600981f,
   0.4600981f, 0.1382363f,
   -0.09779096f, -0.05095487f,
   0.05543245f, 0.026447f,
   -0.03764973f, -0.01485397f,
   0.02716055f, 0.00828756f,
   -0.01994365f, -0.004313674f,
   0.01459396f, 0.001894714f,
   -0.01050689f, -0.0004857935f,
   0.007367171f, -0.0002584767f,
   -0.004989147f, 0.0005743159f,
   0.003235877f, -0.0006243724f,
   -0.001986177f, 0.0005308539f,
   0.00113826f, -0.0003823631f,
   -0.0005953563f, 0.0002298438f,
   0.0002790277f, -0.0001104587f,
   -0.0001123515f, 3.596189e-05f
};

#endif

extern const spx_word16_t lag_window[];
extern const spx_word16_t lpc_window[];


void *sb_encoder_init(const SpeexMode *m)
{
   int i;
   spx_int32_t tmp;
   SBEncState *st;
   const SpeexSBMode *mode;

   st = (SBEncState*)speex_alloc(sizeof(SBEncState));
   if (!st)
      return NULL;
   st->mode = m;
   mode = (const SpeexSBMode*)m->mode;


   st->st_low = speex_encoder_init(mode->nb_mode);
#if defined(VAR_ARRAYS) || defined (USE_ALLOCA)
   st->stack = NULL;
#else
   /*st->stack = (char*)speex_alloc_scratch(SB_ENC_STACK);*/
   speex_encoder_ctl(st->st_low, SPEEX_GET_STACK, &st->stack);
#endif

   st->full_frame_size = 2*mode->frameSize;
   st->frame_size = mode->frameSize;
   st->subframeSize = mode->subframeSize;
   st->nbSubframes = mode->frameSize/mode->subframeSize;
   st->windowSize = st->frame_size+st->subframeSize;
   st->lpcSize=mode->lpcSize;

   st->encode_submode = 1;
   st->submodes=mode->submodes;
   st->submodeSelect = st->submodeID=mode->defaultSubmode;
   
   tmp=9;
   speex_encoder_ctl(st->st_low, SPEEX_SET_QUALITY, &tmp);
   tmp=1;
   speex_encoder_ctl(st->st_low, SPEEX_SET_WIDEBAND, &tmp);

   st->lpc_floor = mode->lpc_floor;
   st->gamma1=mode->gamma1;
   st->gamma2=mode->gamma2;
   st->first=1;

   st->high=(spx_word16_t*)speex_alloc((st->windowSize-st->frame_size)*sizeof(spx_word16_t));

   st->h0_mem=(spx_word16_t*)speex_alloc((QMF_ORDER)*sizeof(spx_word16_t));
   st->h1_mem=(spx_word16_t*)speex_alloc((QMF_ORDER)*sizeof(spx_word16_t));

   st->window= lpc_window;

   st->lagWindow = lag_window;

   st->old_lsp = (spx_lsp_t*)speex_alloc(st->lpcSize*sizeof(spx_lsp_t));
   st->old_qlsp = (spx_lsp_t*)speex_alloc(st->lpcSize*sizeof(spx_lsp_t));
   st->interp_qlpc = (spx_coef_t*)speex_alloc(st->lpcSize*sizeof(spx_coef_t));
   st->pi_gain = (spx_word32_t*)speex_alloc((st->nbSubframes)*sizeof(spx_word32_t));
   st->exc_rms = (spx_word16_t*)speex_alloc((st->nbSubframes)*sizeof(spx_word16_t));
   st->innov_rms_save = NULL;
   
   st->mem_sp = (spx_mem_t*)speex_alloc((st->lpcSize)*sizeof(spx_mem_t));
   st->mem_sp2 = (spx_mem_t*)speex_alloc((st->lpcSize)*sizeof(spx_mem_t));
   st->mem_sw = (spx_mem_t*)speex_alloc((st->lpcSize)*sizeof(spx_mem_t));

   for (i=0;i<st->lpcSize;i++)
      st->old_lsp[i]= DIV32(MULT16_16(QCONST16(3.1415927f, LSP_SHIFT), i+1), st->lpcSize+1);

#ifndef DISABLE_VBR
   st->vbr_quality = 8;
   st->vbr_enabled = 0;
   st->vbr_max = 0;
   st->vbr_max_high = 20000;  /* We just need a big value here */
   st->vad_enabled = 0;
   st->abr_enabled = 0;
   st->relative_quality=0;
#endif /* #ifndef DISABLE_VBR */

   st->complexity=2;
   speex_encoder_ctl(st->st_low, SPEEX_GET_SAMPLING_RATE, &st->sampling_rate);
   st->sampling_rate*=2;
#ifdef ENABLE_VALGRIND
   VALGRIND_MAKE_READABLE(st, (st->stack-(char*)st));
#endif
   return st;
}

void sb_encoder_destroy(void *state)
{
   SBEncState *st=(SBEncState*)state;

   speex_encoder_destroy(st->st_low);
#if !(defined(VAR_ARRAYS) || defined (USE_ALLOCA))
   /*speex_free_scratch(st->stack);*/
#endif

   speex_free(st->high);

   speex_free(st->h0_mem);
   speex_free(st->h1_mem);

   speex_free(st->old_lsp);
   speex_free(st->old_qlsp);
   speex_free(st->interp_qlpc);
   speex_free(st->pi_gain);
   speex_free(st->exc_rms);

   speex_free(st->mem_sp);
   speex_free(st->mem_sp2);
   speex_free(st->mem_sw);

   
   speex_free(st);
}


int sb_encode(void *state, void *vin, SpeexBits *bits)
{
   SBEncState *st;
   int i, roots, sub;
   char *stack;
   VARDECL(spx_mem_t *mem);
   VARDECL(spx_sig_t *innov);
   VARDECL(spx_word16_t *target);
   VARDECL(spx_word16_t *syn_resp);
   VARDECL(spx_word32_t *low_pi_gain);
   spx_word16_t *low;
   spx_word16_t *high;
   VARDECL(spx_word16_t *low_exc_rms);
   VARDECL(spx_word16_t *low_innov_rms);
   const SpeexSBMode *mode;
   spx_int32_t dtx;
   spx_word16_t *in = (spx_word16_t*)vin;
   spx_word16_t e_low=0, e_high=0;
   VARDECL(spx_coef_t *lpc);
   VARDECL(spx_coef_t *interp_lpc);
   VARDECL(spx_coef_t *bw_lpc1);
   VARDECL(spx_coef_t *bw_lpc2);
   VARDECL(spx_lsp_t *lsp);
   VARDECL(spx_lsp_t *qlsp);
   VARDECL(spx_lsp_t *interp_lsp);
   VARDECL(spx_lsp_t *interp_qlsp);
      
   st = (SBEncState*)state;
   stack=st->stack;
   mode = (const SpeexSBMode*)(st->mode->mode);
   low = in;
   high = in+st->frame_size;
   
   /* High-band buffering / sync with low band */
   /* Compute the two sub-bands by filtering with QMF h0*/
   qmf_decomp(in, h0, low, high, st->full_frame_size, QMF_ORDER, st->h0_mem, stack);
   
#ifndef DISABLE_VBR
   if (st->vbr_enabled || st->vad_enabled)
   {
      /* Need to compute things here before the signal is trashed by the encoder */
      /*FIXME: Are the two signals (low, high) in sync? */
      e_low = compute_rms16(low, st->frame_size);
      e_high = compute_rms16(high, st->frame_size);
   }
#endif /* #ifndef DISABLE_VBR */

   ALLOC(low_innov_rms, st->nbSubframes, spx_word16_t);
   speex_encoder_ctl(st->st_low, SPEEX_SET_INNOVATION_SAVE, low_innov_rms);
   /* Encode the narrowband part*/
   speex_encode_native(st->st_low, low, bits);

   high = high - (st->windowSize-st->frame_size);
   SPEEX_COPY(high, st->high, st->windowSize-st->frame_size);
   SPEEX_COPY(st->high, &high[st->frame_size], st->windowSize-st->frame_size);
   

   ALLOC(low_pi_gain, st->nbSubframes, spx_word32_t);
   ALLOC(low_exc_rms, st->nbSubframes, spx_word16_t);
   speex_encoder_ctl(st->st_low, SPEEX_GET_PI_GAIN, low_pi_gain);
   speex_encoder_ctl(st->st_low, SPEEX_GET_EXC, low_exc_rms);
   
   speex_encoder_ctl(st->st_low, SPEEX_GET_LOW_MODE, &dtx);

   if (dtx==0)
      dtx=1;
   else
      dtx=0;

   ALLOC(lpc, st->lpcSize, spx_coef_t);
   ALLOC(interp_lpc, st->lpcSize, spx_coef_t);
   ALLOC(bw_lpc1, st->lpcSize, spx_coef_t);
   ALLOC(bw_lpc2, st->lpcSize, spx_coef_t);
   
   ALLOC(lsp, st->lpcSize, spx_lsp_t);
   ALLOC(qlsp, st->lpcSize, spx_lsp_t);
   ALLOC(interp_lsp, st->lpcSize, spx_lsp_t);
   ALLOC(interp_qlsp, st->lpcSize, spx_lsp_t);
   
   {
      VARDECL(spx_word16_t *autocorr);
      VARDECL(spx_word16_t *w_sig);
      ALLOC(autocorr, st->lpcSize+1, spx_word16_t);
      ALLOC(w_sig, st->windowSize, spx_word16_t);
      /* Window for analysis */
      /* FIXME: This is a kludge */
      if (st->subframeSize==80)
      {
         for (i=0;i<st->windowSize;i++)
            w_sig[i] = EXTRACT16(SHR32(MULT16_16(high[i],st->window[i>>1]),SIG_SHIFT));
      } else {
         for (i=0;i<st->windowSize;i++)
            w_sig[i] = EXTRACT16(SHR32(MULT16_16(high[i],st->window[i]),SIG_SHIFT));
      }
      /* Compute auto-correlation */
      _spx_autocorr(w_sig, autocorr, st->lpcSize+1, st->windowSize);
      autocorr[0] = ADD16(autocorr[0],MULT16_16_Q15(autocorr[0],st->lpc_floor)); /* Noise floor in auto-correlation domain */

      /* Lag windowing: equivalent to filtering in the power-spectrum domain */
      for (i=0;i<st->lpcSize+1;i++)
         autocorr[i] = MULT16_16_Q14(autocorr[i],st->lagWindow[i]);

      /* Levinson-Durbin */
      _spx_lpc(lpc, autocorr, st->lpcSize);
   }

   /* LPC to LSPs (x-domain) transform */
   roots=lpc_to_lsp (lpc, st->lpcSize, lsp, 10, LSP_DELTA1, stack);
   if (roots!=st->lpcSize)
   {
      roots = lpc_to_lsp (lpc, st->lpcSize, lsp, 10, LSP_DELTA2, stack);
      if (roots!=st->lpcSize) {
         /*If we can't find all LSP's, do some damage control and use a flat filter*/
         for (i=0;i<st->lpcSize;i++)
         {
            lsp[i]=st->old_lsp[i];
         }
      }
   }

#ifndef DISABLE_VBR
   /* VBR code */
   if ((st->vbr_enabled || st->vad_enabled) && !dtx)
   {
      float ratio;
      if (st->abr_enabled)
      {
         float qual_change=0;
         if (st->abr_drift2 * st->abr_drift > 0)
         {
            /* Only adapt if long-term and short-term drift are the same sign */
            qual_change = -.00001*st->abr_drift/(1+st->abr_count);
            if (qual_change>.1)
               qual_change=.1;
            if (qual_change<-.1)
               qual_change=-.1;
         }
         st->vbr_quality += qual_change;
         if (st->vbr_quality>10)
            st->vbr_quality=10;
         if (st->vbr_quality<0)
            st->vbr_quality=0;
      }


      ratio = 2*log((1.f+e_high)/(1.f+e_low));
      
      speex_encoder_ctl(st->st_low, SPEEX_GET_RELATIVE_QUALITY, &st->relative_quality);
      if (ratio<-4)
         ratio=-4;
      if (ratio>2)
         ratio=2;
      /*if (ratio>-2)*/
      if (st->vbr_enabled) 
      {
         spx_int32_t modeid;
         modeid = mode->nb_modes-1;
         st->relative_quality+=1.0*(ratio+2);
	 if (st->relative_quality<-1)
            st->relative_quality=-1;
         while (modeid)
         {
            int v1;
            float thresh;
            v1=(int)floor(st->vbr_quality);
            if (v1==10)
               thresh = mode->vbr_thresh[modeid][v1];
            else
               thresh = (st->vbr_quality-v1)   * mode->vbr_thresh[modeid][v1+1] + 
                        (1+v1-st->vbr_quality) * mode->vbr_thresh[modeid][v1];
            if (st->relative_quality >= thresh && st->sampling_rate*st->submodes[modeid]->bits_per_frame/st->full_frame_size <= st->vbr_max_high)
               break;
            modeid--;
         }
         speex_encoder_ctl(state, SPEEX_SET_HIGH_MODE, &modeid);
         if (st->abr_enabled)
         {
            spx_int32_t bitrate;
            speex_encoder_ctl(state, SPEEX_GET_BITRATE, &bitrate);
            st->abr_drift+=(bitrate-st->abr_enabled);
            st->abr_drift2 = .95*st->abr_drift2 + .05*(bitrate-st->abr_enabled);
            st->abr_count += 1.0;
         }

      } else {
         /* VAD only */
         int modeid;
         if (st->relative_quality<2.0)
            modeid=1;
         else
            modeid=st->submodeSelect;
         /*speex_encoder_ctl(state, SPEEX_SET_MODE, &mode);*/
         st->submodeID=modeid;

      }
      /*fprintf (stderr, "%f %f\n", ratio, low_qual);*/
   }
#endif /* #ifndef DISABLE_VBR */

   if (st->encode_submode)
   {
      speex_bits_pack(bits, 1, 1);
      if (dtx)
         speex_bits_pack(bits, 0, SB_SUBMODE_BITS);
      else
         speex_bits_pack(bits, st->submodeID, SB_SUBMODE_BITS);
   }

   /* If null mode (no transmission), just set a couple things to zero*/
   if (dtx || st->submodes[st->submodeID] == NULL)
   {
      for (i=0;i<st->frame_size;i++)
         high[i]=VERY_SMALL;

      for (i=0;i<st->lpcSize;i++)
         st->mem_sw[i]=0;
      st->first=1;

      /* Final signal synthesis from excitation */
      iir_mem16(high, st->interp_qlpc, high, st->frame_size, st->lpcSize, st->mem_sp, stack);

      if (dtx)
         return 0;
      else
         return 1;
   }


   /* LSP quantization */
   SUBMODE(lsp_quant)(lsp, qlsp, st->lpcSize, bits);   

   if (st->first)
   {
      for (i=0;i<st->lpcSize;i++)
         st->old_lsp[i] = lsp[i];
      for (i=0;i<st->lpcSize;i++)
         st->old_qlsp[i] = qlsp[i];
   }
   
   ALLOC(mem, st->lpcSize, spx_mem_t);
   ALLOC(syn_resp, st->subframeSize, spx_word16_t);
   ALLOC(innov, st->subframeSize, spx_sig_t);
   ALLOC(target, st->subframeSize, spx_word16_t);

   for (sub=0;sub<st->nbSubframes;sub++)
   {
      VARDECL(spx_word16_t *exc);
      VARDECL(spx_word16_t *res);
      VARDECL(spx_word16_t *sw);
      spx_word16_t *sp;
      spx_word16_t filter_ratio;     /*Q7*/
      int offset;
      spx_word32_t rl, rh;           /*Q13*/
      spx_word16_t eh=0;

      offset = st->subframeSize*sub;
      sp=high+offset;
      ALLOC(exc, st->subframeSize, spx_word16_t);
      ALLOC(res, st->subframeSize, spx_word16_t);
      ALLOC(sw, st->subframeSize, spx_word16_t);
      
      /* LSP interpolation (quantized and unquantized) */
      lsp_interpolate(st->old_lsp, lsp, interp_lsp, st->lpcSize, sub, st->nbSubframes);
      lsp_interpolate(st->old_qlsp, qlsp, interp_qlsp, st->lpcSize, sub, st->nbSubframes);

      lsp_enforce_margin(interp_lsp, st->lpcSize, LSP_MARGIN);
      lsp_enforce_margin(interp_qlsp, st->lpcSize, LSP_MARGIN);

      lsp_to_lpc(interp_lsp, interp_lpc, st->lpcSize,stack);
      lsp_to_lpc(interp_qlsp, st->interp_qlpc, st->lpcSize, stack);

      bw_lpc(st->gamma1, interp_lpc, bw_lpc1, st->lpcSize);
      bw_lpc(st->gamma2, interp_lpc, bw_lpc2, st->lpcSize);

      /* Compute mid-band (4000 Hz for wideband) response of low-band and high-band
         filters */
      st->pi_gain[sub]=LPC_SCALING;
      rh = LPC_SCALING;
      for (i=0;i<st->lpcSize;i+=2)
      {
         rh += st->interp_qlpc[i+1] - st->interp_qlpc[i];
         st->pi_gain[sub] += st->interp_qlpc[i] + st->interp_qlpc[i+1];
      }
      
      rl = low_pi_gain[sub];
#ifdef FIXED_POINT
      filter_ratio=EXTRACT16(SATURATE(PDIV32(SHL32(ADD32(rl,82),7),ADD32(82,rh)),32767));
#else
      filter_ratio=(rl+.01)/(rh+.01);
#endif
      
      /* Compute "real excitation" */
      fir_mem16(sp, st->interp_qlpc, exc, st->subframeSize, st->lpcSize, st->mem_sp2, stack);
      /* Compute energy of low-band and high-band excitation */

      eh = compute_rms16(exc, st->subframeSize);

      if (!SUBMODE(innovation_quant)) {/* 1 for spectral folding excitation, 0 for stochastic */
         spx_word32_t g;   /*Q7*/
         spx_word16_t el;  /*Q0*/
         el = low_innov_rms[sub];

         /* Gain to use if we want to use the low-band excitation for high-band */
         g=PDIV32(MULT16_16(filter_ratio,eh),EXTEND32(ADD16(1,el)));
         
#if 0
         {
            char *tmp_stack=stack;
            float *tmp_sig;
            float g2;
            ALLOC(tmp_sig, st->subframeSize, spx_sig_t);
            for (i=0;i<st->lpcSize;i++)
               mem[i]=st->mem_sp[i];
            iir_mem2(st->low_innov+offset, st->interp_qlpc, tmp_sig, st->subframeSize, st->lpcSize, mem);
            g2 = compute_rms(sp, st->subframeSize)/(.01+compute_rms(tmp_sig, st->subframeSize));
            /*fprintf (stderr, "gains: %f %f\n", g, g2);*/
            g = g2;
            stack = tmp_stack;
         }
#endif

         /*print_vec(&g, 1, "gain factor");*/
         /* Gain quantization */
         {
            int quant = scal_quant(g, fold_quant_bound, 32);
            /*speex_warning_int("tata", quant);*/
            if (quant<0)
               quant=0;
            if (quant>31)
               quant=31;
            speex_bits_pack(bits, quant, 5);
         }
         if (st->innov_rms_save)
         {
            st->innov_rms_save[sub] = eh;
         }
         st->exc_rms[sub] = eh;
      } else {
         spx_word16_t gc;       /*Q7*/
         spx_word32_t scale;    /*Q14*/
         spx_word16_t el;       /*Q0*/
         el = low_exc_rms[sub]; /*Q0*/

         gc = PDIV32_16(MULT16_16(filter_ratio,1+eh),1+el);

         /* This is a kludge that cleans up a historical bug */
         if (st->subframeSize==80)
            gc = MULT16_16_P15(QCONST16(0.70711f,15),gc);
         /*printf ("%f %f %f %f\n", el, eh, filter_ratio, gc);*/
         {
            int qgc = scal_quant(gc, gc_quant_bound, 16);
            speex_bits_pack(bits, qgc, 4);
            gc = MULT16_16_Q15(QCONST16(0.87360,15),gc_quant_bound[qgc]);
         }
         if (st->subframeSize==80)
            gc = MULT16_16_P14(QCONST16(1.4142f,14), gc);

         scale = SHL32(MULT16_16(PDIV32_16(SHL32(EXTEND32(gc),SIG_SHIFT-6),filter_ratio),(1+el)),6);

         compute_impulse_response(st->interp_qlpc, bw_lpc1, bw_lpc2, syn_resp, st->subframeSize, st->lpcSize, stack);

         
         /* Reset excitation */
         for (i=0;i<st->subframeSize;i++)
            res[i]=VERY_SMALL;
         
         /* Compute zero response (ringing) of A(z/g1) / ( A(z/g2) * Aq(z) ) */
         for (i=0;i<st->lpcSize;i++)
            mem[i]=st->mem_sp[i];
         iir_mem16(res, st->interp_qlpc, res, st->subframeSize, st->lpcSize, mem, stack);

         for (i=0;i<st->lpcSize;i++)
            mem[i]=st->mem_sw[i];
         filter_mem16(res, bw_lpc1, bw_lpc2, res, st->subframeSize, st->lpcSize, mem, stack);

         /* Compute weighted signal */
         for (i=0;i<st->lpcSize;i++)
            mem[i]=st->mem_sw[i];
         filter_mem16(sp, bw_lpc1, bw_lpc2, sw, st->subframeSize, st->lpcSize, mem, stack);

         /* Compute target signal */
         for (i=0;i<st->subframeSize;i++)
            target[i]=SUB16(sw[i],res[i]);

         signal_div(target, target, scale, st->subframeSize);

         /* Reset excitation */
         SPEEX_MEMSET(innov, 0, st->subframeSize);

         /*print_vec(target, st->subframeSize, "\ntarget");*/
         SUBMODE(innovation_quant)(target, st->interp_qlpc, bw_lpc1, bw_lpc2, 
                                   SUBMODE(innovation_params), st->lpcSize, st->subframeSize, 
                                   innov, syn_resp, bits, stack, st->complexity, SUBMODE(double_codebook));
         /*print_vec(target, st->subframeSize, "after");*/

         signal_mul(innov, innov, scale, st->subframeSize);

         if (SUBMODE(double_codebook)) {
            char *tmp_stack=stack;
            VARDECL(spx_sig_t *innov2);
            ALLOC(innov2, st->subframeSize, spx_sig_t);
            SPEEX_MEMSET(innov2, 0, st->subframeSize);
            for (i=0;i<st->subframeSize;i++)
               target[i]=MULT16_16_P13(QCONST16(2.5f,13), target[i]);

            SUBMODE(innovation_quant)(target, st->interp_qlpc, bw_lpc1, bw_lpc2, 
                                      SUBMODE(innovation_params), st->lpcSize, st->subframeSize, 
                                      innov2, syn_resp, bits, stack, st->complexity, 0);
            signal_mul(innov2, innov2, MULT16_32_P15(QCONST16(0.4f,15),scale), st->subframeSize);

            for (i=0;i<st->subframeSize;i++)
               innov[i] = ADD32(innov[i],innov2[i]);
            stack = tmp_stack;
         }
         for (i=0;i<st->subframeSize;i++)
            exc[i] = PSHR32(innov[i],SIG_SHIFT);

         if (st->innov_rms_save)
         {
            st->innov_rms_save[sub] = MULT16_16_Q15(QCONST16(.70711f, 15), compute_rms(innov, st->subframeSize));
         }
         st->exc_rms[sub] = compute_rms16(exc, st->subframeSize);
         

      }

      
      /*Keep the previous memory*/
      for (i=0;i<st->lpcSize;i++)
         mem[i]=st->mem_sp[i];
      /* Final signal synthesis from excitation */
      iir_mem16(exc, st->interp_qlpc, sp, st->subframeSize, st->lpcSize, st->mem_sp, stack);
      
      /* Compute weighted signal again, from synthesized speech (not sure it's the right thing) */
      filter_mem16(sp, bw_lpc1, bw_lpc2, sw, st->subframeSize, st->lpcSize, st->mem_sw, stack);
   }

   for (i=0;i<st->lpcSize;i++)
      st->old_lsp[i] = lsp[i];
   for (i=0;i<st->lpcSize;i++)
      st->old_qlsp[i] = qlsp[i];

   st->first=0;

   return 1;
}





void *sb_decoder_init(const SpeexMode *m)
{
   spx_int32_t tmp;
   SBDecState *st;
   const SpeexSBMode *mode;
   st = (SBDecState*)speex_alloc(sizeof(SBDecState));
   if (!st)
      return NULL;
   st->mode = m;
   mode=(const SpeexSBMode*)m->mode;
   st->encode_submode = 1;

   st->st_low = speex_decoder_init(mode->nb_mode);
#if defined(VAR_ARRAYS) || defined (USE_ALLOCA)
   st->stack = NULL;
#else
   /*st->stack = (char*)speex_alloc_scratch(SB_DEC_STACK);*/
   speex_decoder_ctl(st->st_low, SPEEX_GET_STACK, &st->stack);
#endif

   st->full_frame_size = 2*mode->frameSize;
   st->frame_size = mode->frameSize;
   st->subframeSize = mode->subframeSize;
   st->nbSubframes = mode->frameSize/mode->subframeSize;
   st->lpcSize=mode->lpcSize;
   speex_decoder_ctl(st->st_low, SPEEX_GET_SAMPLING_RATE, &st->sampling_rate);
   st->sampling_rate*=2;
   tmp=1;
   speex_decoder_ctl(st->st_low, SPEEX_SET_WIDEBAND, &tmp);

   st->submodes=mode->submodes;
   st->submodeID=mode->defaultSubmode;

   st->first=1;

   st->g0_mem = (spx_word16_t*)speex_alloc((QMF_ORDER)*sizeof(spx_word16_t));
   st->g1_mem = (spx_word16_t*)speex_alloc((QMF_ORDER)*sizeof(spx_word16_t));

   st->excBuf = (spx_word16_t*)speex_alloc((st->subframeSize)*sizeof(spx_word16_t));

   st->old_qlsp = (spx_lsp_t*)speex_alloc((st->lpcSize)*sizeof(spx_lsp_t));
   st->interp_qlpc = (spx_coef_t*)speex_alloc(st->lpcSize*sizeof(spx_coef_t));

   st->pi_gain = (spx_word32_t*)speex_alloc((st->nbSubframes)*sizeof(spx_word32_t));
   st->exc_rms = (spx_word16_t*)speex_alloc((st->nbSubframes)*sizeof(spx_word16_t));
   st->mem_sp = (spx_mem_t*)speex_alloc((2*st->lpcSize)*sizeof(spx_mem_t));
   
   st->innov_save = NULL;


   st->lpc_enh_enabled=0;
   st->seed = 1000;

#ifdef ENABLE_VALGRIND
   VALGRIND_MAKE_READABLE(st, (st->stack-(char*)st));
#endif
   return st;
}

void sb_decoder_destroy(void *state)
{
   SBDecState *st;
   st = (SBDecState*)state;
   speex_decoder_destroy(st->st_low);
#if !(defined(VAR_ARRAYS) || defined (USE_ALLOCA))
   /*speex_free_scratch(st->stack);*/
#endif

   speex_free(st->g0_mem);
   speex_free(st->g1_mem);
   speex_free(st->excBuf);
   speex_free(st->old_qlsp);
   speex_free(st->interp_qlpc);
   speex_free(st->pi_gain);
   speex_free(st->exc_rms);
   speex_free(st->mem_sp);

   speex_free(state);
}

static void sb_decode_lost(SBDecState *st, spx_word16_t *out, int dtx, char *stack)
{
   int i;
   int saved_modeid=0;

   if (dtx)
   {
      saved_modeid=st->submodeID;
      st->submodeID=1;
   } else {
      bw_lpc(QCONST16(0.99f,15), st->interp_qlpc, st->interp_qlpc, st->lpcSize);
   }

   st->first=1;
   
   
   /* Final signal synthesis from excitation */
   if (!dtx)
   {
      st->last_ener =  MULT16_16_Q15(QCONST16(.9f,15),st->last_ener);
   }
   for (i=0;i<st->frame_size;i++)
      out[i+st->frame_size] = speex_rand(st->last_ener, &st->seed);

   iir_mem16(out+st->frame_size, st->interp_qlpc, out+st->frame_size, st->frame_size, st->lpcSize, 
            st->mem_sp, stack);
   
   
   /* Reconstruct the original */
   qmf_synth(out, out+st->frame_size, h0, out, st->full_frame_size, QMF_ORDER, st->g0_mem, st->g1_mem, stack);
   if (dtx)
   {
      st->submodeID=saved_modeid;
   }

   return;
}

int sb_decode(void *state, SpeexBits *bits, void *vout)
{
   int i, sub;
   SBDecState *st;
   int wideband;
   int ret;
   char *stack;
   VARDECL(spx_word32_t *low_pi_gain);
   VARDECL(spx_word16_t *low_exc_rms);
   VARDECL(spx_coef_t *ak);
   VARDECL(spx_lsp_t *qlsp);
   VARDECL(spx_lsp_t *interp_qlsp);
   spx_int32_t dtx;
   const SpeexSBMode *mode;
   spx_word16_t *out = (spx_word16_t*)vout;
   spx_word16_t *low_innov_alias;
   spx_word32_t exc_ener_sum = 0;
   
   st = (SBDecState*)state;
   stack=st->stack;
   mode = (const SpeexSBMode*)(st->mode->mode);

   low_innov_alias = out+st->frame_size;
   speex_decoder_ctl(st->st_low, SPEEX_SET_INNOVATION_SAVE, low_innov_alias);
   /* Decode the low-band */
   ret = speex_decode_native(st->st_low, bits, out);

   speex_decoder_ctl(st->st_low, SPEEX_GET_DTX_STATUS, &dtx);

   /* If error decoding the narrowband part, propagate error */
   if (ret!=0)
   {
      return ret;
   }

   if (!bits)
   {
      sb_decode_lost(st, out, dtx, stack);
      return 0;
   }

   if (st->encode_submode)
   {

      /*Check "wideband bit"*/
      if (speex_bits_remaining(bits)>0)
         wideband = speex_bits_peek(bits);
      else
         wideband = 0;
      if (wideband)
      {
         /*Regular wideband frame, read the submode*/
         wideband = speex_bits_unpack_unsigned(bits, 1);
         st->submodeID = speex_bits_unpack_unsigned(bits, SB_SUBMODE_BITS);
      } else
      {
         /*Was a narrowband frame, set "null submode"*/
         st->submodeID = 0;
      }
      if (st->submodeID != 0 && st->submodes[st->submodeID] == NULL)
      {
         speex_notify("Invalid mode encountered. The stream is corrupted.");
         return -2;
      }
   }

   /* If null mode (no transmission), just set a couple things to zero*/
   if (st->submodes[st->submodeID] == NULL)
   {
      if (dtx)
      {
         sb_decode_lost(st, out, 1, stack);
         return 0;
      }

      for (i=0;i<st->frame_size;i++)
         out[st->frame_size+i]=VERY_SMALL;

      st->first=1;

      /* Final signal synthesis from excitation */
      iir_mem16(out+st->frame_size, st->interp_qlpc, out+st->frame_size, st->frame_size, st->lpcSize, st->mem_sp, stack);

      qmf_synth(out, out+st->frame_size, h0, out, st->full_frame_size, QMF_ORDER, st->g0_mem, st->g1_mem, stack);

      return 0;

   }

   ALLOC(low_pi_gain, st->nbSubframes, spx_word32_t);
   ALLOC(low_exc_rms, st->nbSubframes, spx_word16_t);
   speex_decoder_ctl(st->st_low, SPEEX_GET_PI_GAIN, low_pi_gain);
   speex_decoder_ctl(st->st_low, SPEEX_GET_EXC, low_exc_rms);

   ALLOC(qlsp, st->lpcSize, spx_lsp_t);
   ALLOC(interp_qlsp, st->lpcSize, spx_lsp_t);
   SUBMODE(lsp_unquant)(qlsp, st->lpcSize, bits);
   
   if (st->first)
   {
      for (i=0;i<st->lpcSize;i++)
         st->old_qlsp[i] = qlsp[i];
   }
   
   ALLOC(ak, st->lpcSize, spx_coef_t);

   for (sub=0;sub<st->nbSubframes;sub++)
   {
      VARDECL(spx_word32_t *exc);
      spx_word16_t *innov_save=NULL;
      spx_word16_t *sp;
      spx_word16_t filter_ratio;
      spx_word16_t el=0;
      int offset;
      spx_word32_t rl=0,rh=0;
      
      offset = st->subframeSize*sub;
      sp=out+st->frame_size+offset;
      ALLOC(exc, st->subframeSize, spx_word32_t);
      /* Pointer for saving innovation */
      if (st->innov_save)
      {
         innov_save = st->innov_save+2*offset;
         SPEEX_MEMSET(innov_save, 0, 2*st->subframeSize);
      }
      
      /* LSP interpolation */
      lsp_interpolate(st->old_qlsp, qlsp, interp_qlsp, st->lpcSize, sub, st->nbSubframes);

      lsp_enforce_margin(interp_qlsp, st->lpcSize, LSP_MARGIN);

      /* LSP to LPC */
      lsp_to_lpc(interp_qlsp, ak, st->lpcSize, stack);

      /* Calculate reponse ratio between the low and high filter in the middle
         of the band (4000 Hz) */
      
         st->pi_gain[sub]=LPC_SCALING;
         rh = LPC_SCALING;
         for (i=0;i<st->lpcSize;i+=2)
         {
            rh += ak[i+1] - ak[i];
            st->pi_gain[sub] += ak[i] + ak[i+1];
         }

         rl = low_pi_gain[sub];
#ifdef FIXED_POINT
         filter_ratio=EXTRACT16(SATURATE(PDIV32(SHL32(ADD32(rl,82),7),ADD32(82,rh)),32767));
#else
         filter_ratio=(rl+.01)/(rh+.01);
#endif
      
      SPEEX_MEMSET(exc, 0, st->subframeSize);
      if (!SUBMODE(innovation_unquant))
      {
         spx_word32_t g;
         int quant;

         quant = speex_bits_unpack_unsigned(bits, 5);
         g= spx_exp(MULT16_16(QCONST16(.125f,11),(quant-10)));
         
         g = PDIV32(g, filter_ratio);
         
         for (i=0;i<st->subframeSize;i+=2)
         {
            exc[i]=SHL32(MULT16_32_P15(MULT16_16_Q15(mode->folding_gain,low_innov_alias[offset+i]),SHL32(g,6)),SIG_SHIFT);
            exc[i+1]=NEG32(SHL32(MULT16_32_P15(MULT16_16_Q15(mode->folding_gain,low_innov_alias[offset+i+1]),SHL32(g,6)),SIG_SHIFT));
         }
         
      } else {
         spx_word16_t gc;
         spx_word32_t scale;
         int qgc = speex_bits_unpack_unsigned(bits, 4);
         
         el = low_exc_rms[sub];
         gc = MULT16_16_Q15(QCONST16(0.87360,15),gc_quant_bound[qgc]);

         if (st->subframeSize==80)
            gc = MULT16_16_P14(QCONST16(1.4142f,14),gc);

         scale = SHL32(PDIV32(SHL32(MULT16_16(gc, el),3), filter_ratio),SIG_SHIFT-3);
         SUBMODE(innovation_unquant)(exc, SUBMODE(innovation_params), st->subframeSize, 
                                     bits, stack, &st->seed);

         signal_mul(exc,exc,scale,st->subframeSize);

         if (SUBMODE(double_codebook)) {
            char *tmp_stack=stack;
            VARDECL(spx_sig_t *innov2);
            ALLOC(innov2, st->subframeSize, spx_sig_t);
            SPEEX_MEMSET(innov2, 0, st->subframeSize);
            SUBMODE(innovation_unquant)(innov2, SUBMODE(innovation_params), st->subframeSize, 
                                        bits, stack, &st->seed);
            signal_mul(innov2, innov2, MULT16_32_P15(QCONST16(0.4f,15),scale), st->subframeSize);
            for (i=0;i<st->subframeSize;i++)
               exc[i] = ADD32(exc[i],innov2[i]);
            stack = tmp_stack;
         }

      }
      
      if (st->innov_save)
      {
         for (i=0;i<st->subframeSize;i++)
            innov_save[2*i]=EXTRACT16(PSHR32(exc[i],SIG_SHIFT));
      }
      
      iir_mem16(st->excBuf, st->interp_qlpc, sp, st->subframeSize, st->lpcSize, 
               st->mem_sp, stack);
      for (i=0;i<st->subframeSize;i++)
         st->excBuf[i]=EXTRACT16(PSHR32(exc[i],SIG_SHIFT));
      for (i=0;i<st->lpcSize;i++)
         st->interp_qlpc[i] = ak[i];
      st->exc_rms[sub] = compute_rms16(st->excBuf, st->subframeSize);
      exc_ener_sum = ADD32(exc_ener_sum, DIV32(MULT16_16(st->exc_rms[sub],st->exc_rms[sub]), st->nbSubframes));
   }
   st->last_ener = spx_sqrt(exc_ener_sum);
   
   qmf_synth(out, out+st->frame_size, h0, out, st->full_frame_size, QMF_ORDER, st->g0_mem, st->g1_mem, stack);
   for (i=0;i<st->lpcSize;i++)
      st->old_qlsp[i] = qlsp[i];

   st->first=0;

   return 0;
}


int sb_encoder_ctl(void *state, int request, void *ptr)
{
   SBEncState *st;
   st=(SBEncState*)state;
   switch(request)
   {
   case SPEEX_GET_FRAME_SIZE:
      (*(spx_int32_t*)ptr) = st->full_frame_size;
      break;
   case SPEEX_SET_HIGH_MODE:
      st->submodeSelect = st->submodeID = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_SET_LOW_MODE:
      speex_encoder_ctl(st->st_low, SPEEX_SET_LOW_MODE, ptr);
      break;
   case SPEEX_SET_DTX:
      speex_encoder_ctl(st->st_low, SPEEX_SET_DTX, ptr);
      break;
   case SPEEX_GET_DTX:
      speex_encoder_ctl(st->st_low, SPEEX_GET_DTX, ptr);
      break;
   case SPEEX_GET_LOW_MODE:
      speex_encoder_ctl(st->st_low, SPEEX_GET_LOW_MODE, ptr);
      break;
   case SPEEX_SET_MODE:
      speex_encoder_ctl(st, SPEEX_SET_QUALITY, ptr);
      break;
#ifndef DISABLE_VBR
   case SPEEX_SET_VBR:
      st->vbr_enabled = (*(spx_int32_t*)ptr);
      speex_encoder_ctl(st->st_low, SPEEX_SET_VBR, ptr);
      break;
   case SPEEX_GET_VBR:
      (*(spx_int32_t*)ptr) = st->vbr_enabled;
      break;
   case SPEEX_SET_VAD:
      st->vad_enabled = (*(spx_int32_t*)ptr);
      speex_encoder_ctl(st->st_low, SPEEX_SET_VAD, ptr);
      break;
   case SPEEX_GET_VAD:
      (*(spx_int32_t*)ptr) = st->vad_enabled;
      break;
#endif /* #ifndef DISABLE_VBR */
#if !defined(DISABLE_VBR) && !defined(DISABLE_FLOAT_API)
   case SPEEX_SET_VBR_QUALITY:
      {
         spx_int32_t q;
         float qual = (*(float*)ptr)+.6;
         st->vbr_quality = (*(float*)ptr);
         if (qual>10)
            qual=10;
         q=(int)floor(.5+*(float*)ptr);
         if (q>10)
            q=10;
         speex_encoder_ctl(st->st_low, SPEEX_SET_VBR_QUALITY, &qual);
         speex_encoder_ctl(state, SPEEX_SET_QUALITY, &q);
         break;
      }
   case SPEEX_GET_VBR_QUALITY:
      (*(float*)ptr) = st->vbr_quality;
      break;
#endif /* #if !defined(DISABLE_VBR) && !defined(DISABLE_FLOAT_API) */
#ifndef DISABLE_VBR
   case SPEEX_SET_ABR:
      st->abr_enabled = (*(spx_int32_t*)ptr);
      st->vbr_enabled = st->abr_enabled!=0;
      speex_encoder_ctl(st->st_low, SPEEX_SET_VBR, &st->vbr_enabled);
      if (st->vbr_enabled) 
      {
         spx_int32_t i=10, rate, target;
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

   case SPEEX_SET_QUALITY:
      {
         spx_int32_t nb_qual;
         int quality = (*(spx_int32_t*)ptr);
         if (quality < 0)
            quality = 0;
         if (quality > 10)
            quality = 10;
         st->submodeSelect = st->submodeID = ((const SpeexSBMode*)(st->mode->mode))->quality_map[quality];
         nb_qual = ((const SpeexSBMode*)(st->mode->mode))->low_quality_map[quality];
         speex_encoder_ctl(st->st_low, SPEEX_SET_MODE, &nb_qual);
      }
      break;
   case SPEEX_SET_COMPLEXITY:
      speex_encoder_ctl(st->st_low, SPEEX_SET_COMPLEXITY, ptr);
      st->complexity = (*(spx_int32_t*)ptr);
      if (st->complexity<1)
         st->complexity=1;
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
      speex_encoder_ctl(st->st_low, request, ptr);
      /*fprintf (stderr, "before: %d\n", (*(int*)ptr));*/
      if (st->submodes[st->submodeID])
         (*(spx_int32_t*)ptr) += st->sampling_rate*SUBMODE(bits_per_frame)/st->full_frame_size;
      else
         (*(spx_int32_t*)ptr) += st->sampling_rate*(SB_SUBMODE_BITS+1)/st->full_frame_size;
      /*fprintf (stderr, "after: %d\n", (*(int*)ptr));*/
      break;
   case SPEEX_SET_SAMPLING_RATE:
      {
         spx_int32_t tmp=(*(spx_int32_t*)ptr);
         st->sampling_rate = tmp;
         tmp>>=1;
         speex_encoder_ctl(st->st_low, SPEEX_SET_SAMPLING_RATE, &tmp);
      }
      break;
   case SPEEX_GET_SAMPLING_RATE:
      (*(spx_int32_t*)ptr)=st->sampling_rate;
      break;
   case SPEEX_RESET_STATE:
      {
         int i;
         st->first = 1;
         for (i=0;i<st->lpcSize;i++)
            st->old_lsp[i]= DIV32(MULT16_16(QCONST16(3.1415927f, LSP_SHIFT), i+1), st->lpcSize+1);
         for (i=0;i<st->lpcSize;i++)
            st->mem_sw[i]=st->mem_sp[i]=st->mem_sp2[i]=0;
         for (i=0;i<QMF_ORDER;i++)
            st->h0_mem[i]=st->h1_mem[i]=0;
      }
      break;
   case SPEEX_SET_SUBMODE_ENCODING:
      st->encode_submode = (*(spx_int32_t*)ptr);
      speex_encoder_ctl(st->st_low, SPEEX_SET_SUBMODE_ENCODING, ptr);
      break;
   case SPEEX_GET_SUBMODE_ENCODING:
      (*(spx_int32_t*)ptr) = st->encode_submode;
      break;
   case SPEEX_GET_LOOKAHEAD:
      speex_encoder_ctl(st->st_low, SPEEX_GET_LOOKAHEAD, ptr);
      (*(spx_int32_t*)ptr) = 2*(*(spx_int32_t*)ptr) + QMF_ORDER - 1;
      break;
   case SPEEX_SET_PLC_TUNING:
      speex_encoder_ctl(st->st_low, SPEEX_SET_PLC_TUNING, ptr);
      break;
   case SPEEX_GET_PLC_TUNING:
      speex_encoder_ctl(st->st_low, SPEEX_GET_PLC_TUNING, ptr);
      break;
#ifndef DISABLE_VBR
   case SPEEX_SET_VBR_MAX_BITRATE:
      {
         st->vbr_max = (*(spx_int32_t*)ptr);
         if (SPEEX_SET_VBR_MAX_BITRATE<1)
         {
            speex_encoder_ctl(st->st_low, SPEEX_SET_VBR_MAX_BITRATE, &st->vbr_max);
            st->vbr_max_high = 17600;
         } else {
            spx_int32_t low_rate;
            if (st->vbr_max >= 42200)
            {
               st->vbr_max_high = 17600;
            } else if (st->vbr_max >= 27800)
            {
               st->vbr_max_high = 9600;
            } else if (st->vbr_max > 20600)
            {
               st->vbr_max_high = 5600;
            } else {
               st->vbr_max_high = 1800;
            }
            if (st->subframeSize==80)
               st->vbr_max_high = 1800;
            low_rate = st->vbr_max - st->vbr_max_high;
            speex_encoder_ctl(st->st_low, SPEEX_SET_VBR_MAX_BITRATE, &low_rate);
         }
      }
      break;
   case SPEEX_GET_VBR_MAX_BITRATE:
      (*(spx_int32_t*)ptr) = st->vbr_max;
      break;
#endif /* #ifndef DISABLE_VBR */
   case SPEEX_SET_HIGHPASS:
      speex_encoder_ctl(st->st_low, SPEEX_SET_HIGHPASS, ptr);
      break;
   case SPEEX_GET_HIGHPASS:
      speex_encoder_ctl(st->st_low, SPEEX_GET_HIGHPASS, ptr);
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
            ((spx_word16_t*)ptr)[i] = st->exc_rms[i];
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
      speex_encoder_ctl(st->st_low, SPEEX_SET_WIDEBAND, ptr);
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

int sb_decoder_ctl(void *state, int request, void *ptr)
{
   SBDecState *st;
   st=(SBDecState*)state;
   switch(request)
   {
   case SPEEX_SET_HIGH_MODE:
      st->submodeID = (*(spx_int32_t*)ptr);
      break;
   case SPEEX_SET_LOW_MODE:
      speex_decoder_ctl(st->st_low, SPEEX_SET_LOW_MODE, ptr);
      break;
   case SPEEX_GET_LOW_MODE:
      speex_decoder_ctl(st->st_low, SPEEX_GET_LOW_MODE, ptr);
      break;
   case SPEEX_GET_FRAME_SIZE:
      (*(spx_int32_t*)ptr) = st->full_frame_size;
      break;
   case SPEEX_SET_ENH:
      speex_decoder_ctl(st->st_low, request, ptr);
      st->lpc_enh_enabled = *((spx_int32_t*)ptr);
      break;
   case SPEEX_GET_ENH:
      *((spx_int32_t*)ptr) = st->lpc_enh_enabled;
      break;
   case SPEEX_SET_MODE:
   case SPEEX_SET_QUALITY:
      {
         spx_int32_t nb_qual;
         int quality = (*(spx_int32_t*)ptr);
         if (quality < 0)
            quality = 0;
         if (quality > 10)
            quality = 10;
         st->submodeID = ((const SpeexSBMode*)(st->mode->mode))->quality_map[quality];
         nb_qual = ((const SpeexSBMode*)(st->mode->mode))->low_quality_map[quality];
         speex_decoder_ctl(st->st_low, SPEEX_SET_MODE, &nb_qual);
      }
      break;
   case SPEEX_GET_BITRATE:
      speex_decoder_ctl(st->st_low, request, ptr);
      if (st->submodes[st->submodeID])
         (*(spx_int32_t*)ptr) += st->sampling_rate*SUBMODE(bits_per_frame)/st->full_frame_size;
      else
         (*(spx_int32_t*)ptr) += st->sampling_rate*(SB_SUBMODE_BITS+1)/st->full_frame_size;
      break;
   case SPEEX_SET_SAMPLING_RATE:
      {
         spx_int32_t tmp=(*(spx_int32_t*)ptr);
         st->sampling_rate = tmp;
         tmp>>=1;
         speex_decoder_ctl(st->st_low, SPEEX_SET_SAMPLING_RATE, &tmp);
      }
      break;
   case SPEEX_GET_SAMPLING_RATE:
      (*(spx_int32_t*)ptr)=st->sampling_rate;
      break;
   case SPEEX_SET_HANDLER:
      speex_decoder_ctl(st->st_low, SPEEX_SET_HANDLER, ptr);
      break;
   case SPEEX_SET_USER_HANDLER:
      speex_decoder_ctl(st->st_low, SPEEX_SET_USER_HANDLER, ptr);
      break;
   case SPEEX_RESET_STATE:
      {
         int i;
         for (i=0;i<2*st->lpcSize;i++)
            st->mem_sp[i]=0;
         for (i=0;i<QMF_ORDER;i++)
            st->g0_mem[i]=st->g1_mem[i]=0;
         st->last_ener=0;
      }
      break;
   case SPEEX_SET_SUBMODE_ENCODING:
      st->encode_submode = (*(spx_int32_t*)ptr);
      speex_decoder_ctl(st->st_low, SPEEX_SET_SUBMODE_ENCODING, ptr);
      break;
   case SPEEX_GET_SUBMODE_ENCODING:
      (*(spx_int32_t*)ptr) = st->encode_submode;
      break;
   case SPEEX_GET_LOOKAHEAD:
      speex_decoder_ctl(st->st_low, SPEEX_GET_LOOKAHEAD, ptr);
      (*(spx_int32_t*)ptr) = 2*(*(spx_int32_t*)ptr);
      break;
   case SPEEX_SET_HIGHPASS:
      speex_decoder_ctl(st->st_low, SPEEX_SET_HIGHPASS, ptr);
      break;
   case SPEEX_GET_HIGHPASS:
      speex_decoder_ctl(st->st_low, SPEEX_GET_HIGHPASS, ptr);
      break;
   case SPEEX_GET_ACTIVITY:
      speex_decoder_ctl(st->st_low, SPEEX_GET_ACTIVITY, ptr);
      break;
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
            ((spx_word16_t*)ptr)[i] = st->exc_rms[i];
      }
      break;
   case SPEEX_GET_DTX_STATUS:
      speex_decoder_ctl(st->st_low, SPEEX_GET_DTX_STATUS, ptr);
      break;
   case SPEEX_SET_INNOVATION_SAVE:
      st->innov_save = (spx_word16_t*)ptr;
      break;
   case SPEEX_SET_WIDEBAND:
      speex_decoder_ctl(st->st_low, SPEEX_SET_WIDEBAND, ptr);
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

#endif

