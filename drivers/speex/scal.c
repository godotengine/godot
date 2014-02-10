/* Copyright (C) 2006-2008 CSIRO, Jean-Marc Valin, Xiph.Org Foundation

   File: scal.c
   Shaped comb-allpass filter for channel decorrelation

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

/*
The algorithm implemented here is described in:

* J.-M. Valin, Perceptually-Motivated Nonlinear Channel Decorrelation For 
  Stereo Acoustic Echo Cancellation, Accepted for Joint Workshop on 
  Hands­free Speech Communication and Microphone Arrays (HSCMA), 2008.
  http://people.xiph.org/~jm/papers/valin_hscma2008.pdf

*/


#include "config.h"


#include "speex/speex_echo.h"
#include "vorbis_psy.h"
#include "arch.h"
#include "os_support.h"
#include "smallft.h"
#include <math.h>
#include <stdlib.h>

#define ALLPASS_ORDER 20

struct SpeexDecorrState_ {
   int rate;
   int channels;
   int frame_size;
#ifdef VORBIS_PSYCHO
   VorbisPsy *psy;
   struct drft_lookup lookup;
   float *wola_mem;
   float *curve;
#endif
   float *vorbis_win;
   int    seed;
   float *y;
   
   /* Per-channel stuff */
   float *buff;
   float (*ring)[ALLPASS_ORDER];
   int *ringID;
   int *order;
   float *alpha;
};



EXPORT SpeexDecorrState *speex_decorrelate_new(int rate, int channels, int frame_size)
{
   int i, ch;
   SpeexDecorrState *st = speex_alloc(sizeof(SpeexDecorrState));
   st->rate = rate;
   st->channels = channels;
   st->frame_size = frame_size;
#ifdef VORBIS_PSYCHO
   st->psy = vorbis_psy_init(rate, 2*frame_size);
   spx_drft_init(&st->lookup, 2*frame_size);
   st->wola_mem = speex_alloc(frame_size*sizeof(float));
   st->curve = speex_alloc(frame_size*sizeof(float));
#endif
   st->y = speex_alloc(frame_size*sizeof(float));

   st->buff = speex_alloc(channels*2*frame_size*sizeof(float));
   st->ringID = speex_alloc(channels*sizeof(int));
   st->order = speex_alloc(channels*sizeof(int));
   st->alpha = speex_alloc(channels*sizeof(float));
   st->ring = speex_alloc(channels*ALLPASS_ORDER*sizeof(float));
   
   /*FIXME: The +20 is there only as a kludge for ALL_PASS_OLA*/
   st->vorbis_win = speex_alloc((2*frame_size+20)*sizeof(float));
   for (i=0;i<2*frame_size;i++)
      st->vorbis_win[i] = sin(.5*SPEEX_PI* sin(SPEEX_PI*i/(2*frame_size))*sin(SPEEX_PI*i/(2*frame_size)) );
   st->seed = rand();
   
   for (ch=0;ch<channels;ch++)
   {
      for (i=0;i<ALLPASS_ORDER;i++)
         st->ring[ch][i] = 0;
      st->ringID[ch] = 0;
      st->alpha[ch] = 0;
      st->order[ch] = 10;
   }
   return st;
}

static float uni_rand(int *seed)
{
   const unsigned int jflone = 0x3f800000;
   const unsigned int jflmsk = 0x007fffff;
   union {int i; float f;} ran;
   *seed = 1664525 * *seed + 1013904223;
   ran.i = jflone | (jflmsk & *seed);
   ran.f -= 1.5;
   return 2*ran.f;
}

static unsigned int irand(int *seed)
{
   *seed = 1664525 * *seed + 1013904223;
   return ((unsigned int)*seed)>>16;
}


EXPORT void speex_decorrelate(SpeexDecorrState *st, const spx_int16_t *in, spx_int16_t *out, int strength)
{
   int ch;
   float amount;
   
   if (strength<0)
      strength = 0;
   if (strength>100)
      strength = 100;
   
   amount = .01*strength;
   for (ch=0;ch<st->channels;ch++)
   {
      int i;
      int N=2*st->frame_size;
      float beta, beta2;
      float *x;
      float max_alpha = 0;
      
      float *buff;
      float *ring;
      int ringID;
      int order;
      float alpha;

      buff = st->buff+ch*2*st->frame_size;
      ring = st->ring[ch];
      ringID = st->ringID[ch];
      order = st->order[ch];
      alpha = st->alpha[ch];
      
      for (i=0;i<st->frame_size;i++)
         buff[i] = buff[i+st->frame_size];
      for (i=0;i<st->frame_size;i++)
         buff[i+st->frame_size] = in[i*st->channels+ch];

      x = buff+st->frame_size;
      beta = 1.-.3*amount*amount;
      if (amount>1)
         beta = 1-sqrt(.4*amount);
      else
         beta = 1-0.63246*amount;
      if (beta<0)
         beta = 0;
   
      beta2 = beta;
      for (i=0;i<st->frame_size;i++)
      {
         st->y[i] = alpha*(x[i-ALLPASS_ORDER+order]-beta*x[i-ALLPASS_ORDER+order-1])*st->vorbis_win[st->frame_size+i+order] 
               + x[i-ALLPASS_ORDER]*st->vorbis_win[st->frame_size+i] 
               - alpha*(ring[ringID]
               - beta*ring[ringID+1>=order?0:ringID+1]);
         ring[ringID++]=st->y[i];
         st->y[i] *= st->vorbis_win[st->frame_size+i];
         if (ringID>=order)
            ringID=0;
      }
      order = order+(irand(&st->seed)%3)-1;
      if (order < 5)
         order = 5;
      if (order > 10)
         order = 10;
      /*order = 5+(irand(&st->seed)%6);*/
      max_alpha = pow(.96+.04*(amount-1),order);
      if (max_alpha > .98/(1.+beta2))
         max_alpha = .98/(1.+beta2);
   
      alpha = alpha + .4*uni_rand(&st->seed);
      if (alpha > max_alpha)
         alpha = max_alpha;
      if (alpha < -max_alpha)
         alpha = -max_alpha;
      for (i=0;i<ALLPASS_ORDER;i++)
         ring[i] = 0;
      ringID = 0;
      for (i=0;i<st->frame_size;i++)
      {
         float tmp =  alpha*(x[i-ALLPASS_ORDER+order]-beta*x[i-ALLPASS_ORDER+order-1])*st->vorbis_win[i+order] 
               + x[i-ALLPASS_ORDER]*st->vorbis_win[i] 
               - alpha*(ring[ringID]
               - beta*ring[ringID+1>=order?0:ringID+1]);
         ring[ringID++]=tmp;
         tmp *= st->vorbis_win[i];
         if (ringID>=order)
            ringID=0;
         st->y[i] += tmp;
      }
   
#ifdef VORBIS_PSYCHO
      float frame[N];
      float scale = 1./N;
      for (i=0;i<2*st->frame_size;i++)
         frame[i] = buff[i];
   //float coef = .5*0.78130;
      float coef = SPEEX_PI*0.075063 * 0.93763 * amount * .8 * 0.707;
      compute_curve(st->psy, buff, st->curve);
      for (i=1;i<st->frame_size;i++)
      {
         float x1,x2;
         float gain;
         do {
            x1 = uni_rand(&st->seed);
            x2 = uni_rand(&st->seed);
         } while (x1*x1+x2*x2 > 1.);
         gain = coef*sqrt(.1+st->curve[i]);
         frame[2*i-1] = gain*x1;
         frame[2*i] = gain*x2;
      }
      frame[0] = coef*uni_rand(&st->seed)*sqrt(.1+st->curve[0]);
      frame[2*st->frame_size-1] = coef*uni_rand(&st->seed)*sqrt(.1+st->curve[st->frame_size-1]);
      spx_drft_backward(&st->lookup,frame);
      for (i=0;i<2*st->frame_size;i++)
         frame[i] *= st->vorbis_win[i];
#endif
   
      for (i=0;i<st->frame_size;i++)
      {
#ifdef VORBIS_PSYCHO
         float tmp = st->y[i] + frame[i] + st->wola_mem[i];
         st->wola_mem[i] = frame[i+st->frame_size];
#else
         float tmp = st->y[i];
#endif
         if (tmp>32767)
            tmp = 32767;
         if (tmp < -32767)
            tmp = -32767;
         out[i*st->channels+ch] = tmp;
      }
      
      st->ringID[ch] = ringID;
      st->order[ch] = order;
      st->alpha[ch] = alpha;

   }
}

EXPORT void speex_decorrelate_destroy(SpeexDecorrState *st)
{
#ifdef VORBIS_PSYCHO
   vorbis_psy_destroy(st->psy);
   speex_free(st->wola_mem);
   speex_free(st->curve);
#endif
   speex_free(st->buff);
   speex_free(st->ring);
   speex_free(st->ringID);
   speex_free(st->alpha);
   speex_free(st->vorbis_win);
   speex_free(st->order);
   speex_free(st->y);
   speex_free(st);
}
