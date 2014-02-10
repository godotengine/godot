/* Copyright (C) 2002-2006 Jean-Marc Valin 
   File: ltp.c
   Long-Term Prediction functions

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
#include "ltp.h"
#include "stack_alloc.h"
#include "filters.h"
#include <speex/speex_bits.h>
#include "math_approx.h"
#include "os_support.h"

#ifndef NULL
#define NULL 0
#endif


#ifdef _USE_SSE
#include "ltp_sse.h"
#elif defined (ARM4_ASM) || defined(ARM5E_ASM)
#include "ltp_arm4.h"
#elif defined (BFIN_ASM)
#include "ltp_bfin.h"
#endif

#ifndef OVERRIDE_INNER_PROD
spx_word32_t inner_prod(const spx_word16_t *x, const spx_word16_t *y, int len)
{
   spx_word32_t sum=0;
   len >>= 2;
   while(len--)
   {
      spx_word32_t part=0;
      part = MAC16_16(part,*x++,*y++);
      part = MAC16_16(part,*x++,*y++);
      part = MAC16_16(part,*x++,*y++);
      part = MAC16_16(part,*x++,*y++);
      /* HINT: If you had a 40-bit accumulator, you could shift only at the end */
      sum = ADD32(sum,SHR32(part,6));
   }
   return sum;
}
#endif

#ifndef OVERRIDE_PITCH_XCORR
#if 0 /* HINT: Enable this for machines with enough registers (i.e. not x86) */
void pitch_xcorr(const spx_word16_t *_x, const spx_word16_t *_y, spx_word32_t *corr, int len, int nb_pitch, char *stack)
{
   int i,j;
   for (i=0;i<nb_pitch;i+=4)
   {
      /* Compute correlation*/
      /*corr[nb_pitch-1-i]=inner_prod(x, _y+i, len);*/
      spx_word32_t sum1=0;
      spx_word32_t sum2=0;
      spx_word32_t sum3=0;
      spx_word32_t sum4=0;
      const spx_word16_t *y = _y+i;
      const spx_word16_t *x = _x;
      spx_word16_t y0, y1, y2, y3;
      /*y0=y[0];y1=y[1];y2=y[2];y3=y[3];*/
      y0=*y++;
      y1=*y++;
      y2=*y++;
      y3=*y++;
      for (j=0;j<len;j+=4)
      {
         spx_word32_t part1;
         spx_word32_t part2;
         spx_word32_t part3;
         spx_word32_t part4;
         part1 = MULT16_16(*x,y0);
         part2 = MULT16_16(*x,y1);
         part3 = MULT16_16(*x,y2);
         part4 = MULT16_16(*x,y3);
         x++;
         y0=*y++;
         part1 = MAC16_16(part1,*x,y1);
         part2 = MAC16_16(part2,*x,y2);
         part3 = MAC16_16(part3,*x,y3);
         part4 = MAC16_16(part4,*x,y0);
         x++;
         y1=*y++;
         part1 = MAC16_16(part1,*x,y2);
         part2 = MAC16_16(part2,*x,y3);
         part3 = MAC16_16(part3,*x,y0);
         part4 = MAC16_16(part4,*x,y1);
         x++;
         y2=*y++;
         part1 = MAC16_16(part1,*x,y3);
         part2 = MAC16_16(part2,*x,y0);
         part3 = MAC16_16(part3,*x,y1);
         part4 = MAC16_16(part4,*x,y2);
         x++;
         y3=*y++;
         
         sum1 = ADD32(sum1,SHR32(part1,6));
         sum2 = ADD32(sum2,SHR32(part2,6));
         sum3 = ADD32(sum3,SHR32(part3,6));
         sum4 = ADD32(sum4,SHR32(part4,6));
      }
      corr[nb_pitch-1-i]=sum1;
      corr[nb_pitch-2-i]=sum2;
      corr[nb_pitch-3-i]=sum3;
      corr[nb_pitch-4-i]=sum4;
   }

}
#else
void pitch_xcorr(const spx_word16_t *_x, const spx_word16_t *_y, spx_word32_t *corr, int len, int nb_pitch, char *stack)
{
   int i;
   for (i=0;i<nb_pitch;i++)
   {
      /* Compute correlation*/
      corr[nb_pitch-1-i]=inner_prod(_x, _y+i, len);
   }

}
#endif
#endif

#ifndef OVERRIDE_COMPUTE_PITCH_ERROR
static SPEEX_INLINE spx_word32_t compute_pitch_error(spx_word16_t *C, spx_word16_t *g, spx_word16_t pitch_control)
{
   spx_word32_t sum = 0;
   sum = ADD32(sum,MULT16_16(MULT16_16_16(g[0],pitch_control),C[0]));
   sum = ADD32(sum,MULT16_16(MULT16_16_16(g[1],pitch_control),C[1]));
   sum = ADD32(sum,MULT16_16(MULT16_16_16(g[2],pitch_control),C[2]));
   sum = SUB32(sum,MULT16_16(MULT16_16_16(g[0],g[1]),C[3]));
   sum = SUB32(sum,MULT16_16(MULT16_16_16(g[2],g[1]),C[4]));
   sum = SUB32(sum,MULT16_16(MULT16_16_16(g[2],g[0]),C[5]));
   sum = SUB32(sum,MULT16_16(MULT16_16_16(g[0],g[0]),C[6]));
   sum = SUB32(sum,MULT16_16(MULT16_16_16(g[1],g[1]),C[7]));
   sum = SUB32(sum,MULT16_16(MULT16_16_16(g[2],g[2]),C[8]));
   return sum;
}
#endif

#ifndef OVERRIDE_OPEN_LOOP_NBEST_PITCH
void open_loop_nbest_pitch(spx_word16_t *sw, int start, int end, int len, int *pitch, spx_word16_t *gain, int N, char *stack)
{
   int i,j,k;
   VARDECL(spx_word32_t *best_score);
   VARDECL(spx_word32_t *best_ener);
   spx_word32_t e0;
   VARDECL(spx_word32_t *corr);
#ifdef FIXED_POINT
   /* In fixed-point, we need only one (temporary) array of 32-bit values and two (corr16, ener16) 
      arrays for (normalized) 16-bit values */
   VARDECL(spx_word16_t *corr16);
   VARDECL(spx_word16_t *ener16);
   spx_word32_t *energy;
   int cshift=0, eshift=0;
   int scaledown = 0;
   ALLOC(corr16, end-start+1, spx_word16_t);
   ALLOC(ener16, end-start+1, spx_word16_t);
   ALLOC(corr, end-start+1, spx_word32_t);
   energy = corr;
#else
   /* In floating-point, we need to float arrays and no normalized copies */
   VARDECL(spx_word32_t *energy);
   spx_word16_t *corr16;
   spx_word16_t *ener16;
   ALLOC(energy, end-start+2, spx_word32_t);
   ALLOC(corr, end-start+1, spx_word32_t);
   corr16 = corr;
   ener16 = energy;
#endif
   
   ALLOC(best_score, N, spx_word32_t);
   ALLOC(best_ener, N, spx_word32_t);
   for (i=0;i<N;i++)
   {
        best_score[i]=-1;
        best_ener[i]=0;
        pitch[i]=start;
   }
   
#ifdef FIXED_POINT
   for (i=-end;i<len;i++)
   {
      if (ABS16(sw[i])>16383)
      {
         scaledown=1;
         break;
      }
   }
   /* If the weighted input is close to saturation, then we scale it down */
   if (scaledown)
   {
      for (i=-end;i<len;i++)
      {
         sw[i]=SHR16(sw[i],1);
      }
   }      
#endif
   energy[0]=inner_prod(sw-start, sw-start, len);
   e0=inner_prod(sw, sw, len);
   for (i=start;i<end;i++)
   {
      /* Update energy for next pitch*/
      energy[i-start+1] = SUB32(ADD32(energy[i-start],SHR32(MULT16_16(sw[-i-1],sw[-i-1]),6)), SHR32(MULT16_16(sw[-i+len-1],sw[-i+len-1]),6));
      if (energy[i-start+1] < 0)
         energy[i-start+1] = 0;
   }
   
#ifdef FIXED_POINT
   eshift = normalize16(energy, ener16, 32766, end-start+1);
#endif
   
   /* In fixed-point, this actually overrites the energy array (aliased to corr) */
   pitch_xcorr(sw, sw-end, corr, len, end-start+1, stack);
   
#ifdef FIXED_POINT
   /* Normalize to 180 so we can square it and it still fits in 16 bits */
   cshift = normalize16(corr, corr16, 180, end-start+1);
   /* If we scaled weighted input down, we need to scale it up again (OK, so we've just lost the LSB, who cares?) */
   if (scaledown)
   {
      for (i=-end;i<len;i++)
      {
         sw[i]=SHL16(sw[i],1);
      }
   }      
#endif

   /* Search for the best pitch prediction gain */
   for (i=start;i<=end;i++)
   {
      spx_word16_t tmp = MULT16_16_16(corr16[i-start],corr16[i-start]);
      /* Instead of dividing the tmp by the energy, we multiply on the other side */
      if (MULT16_16(tmp,best_ener[N-1])>MULT16_16(best_score[N-1],ADD16(1,ener16[i-start])))
      {
         /* We can safely put it last and then check */
         best_score[N-1]=tmp;
         best_ener[N-1]=ener16[i-start]+1;
         pitch[N-1]=i;
         /* Check if it comes in front of others */
         for (j=0;j<N-1;j++)
         {
            if (MULT16_16(tmp,best_ener[j])>MULT16_16(best_score[j],ADD16(1,ener16[i-start])))
            {
               for (k=N-1;k>j;k--)
               {
                  best_score[k]=best_score[k-1];
                  best_ener[k]=best_ener[k-1];
                  pitch[k]=pitch[k-1];
               }
               best_score[j]=tmp;
               best_ener[j]=ener16[i-start]+1;
               pitch[j]=i;
               break;
            }
         }
      }
   }
   
   /* Compute open-loop gain if necessary */
   if (gain)
   {
      for (j=0;j<N;j++)
      {
         spx_word16_t g;
         i=pitch[j];
         g = DIV32(SHL32(EXTEND32(corr16[i-start]),cshift), 10+SHR32(MULT16_16(spx_sqrt(e0),spx_sqrt(SHL32(EXTEND32(ener16[i-start]),eshift))),6));
         /* FIXME: g = max(g,corr/energy) */
         if (g<0)
            g = 0;
         gain[j]=g;
      }
   }


}
#endif

#ifndef OVERRIDE_PITCH_GAIN_SEARCH_3TAP_VQ
static int pitch_gain_search_3tap_vq(
  const signed char *gain_cdbk,
  int                gain_cdbk_size,
  spx_word16_t      *C16,
  spx_word16_t       max_gain
)
{
  const signed char *ptr=gain_cdbk;
  int                best_cdbk=0;
  spx_word32_t       best_sum=-VERY_LARGE32;
  spx_word32_t       sum=0;
  spx_word16_t       g[3];
  spx_word16_t       pitch_control=64;
  spx_word16_t       gain_sum;
  int                i;

  for (i=0;i<gain_cdbk_size;i++) {
         
    ptr = gain_cdbk+4*i;
    g[0]=ADD16((spx_word16_t)ptr[0],32);
    g[1]=ADD16((spx_word16_t)ptr[1],32);
    g[2]=ADD16((spx_word16_t)ptr[2],32);
    gain_sum = (spx_word16_t)ptr[3];
         
    sum = compute_pitch_error(C16, g, pitch_control);
         
    if (sum>best_sum && gain_sum<=max_gain) {
      best_sum=sum;
      best_cdbk=i;
    }
  }

  return best_cdbk;
}
#endif

/** Finds the best quantized 3-tap pitch predictor by analysis by synthesis */
static spx_word32_t pitch_gain_search_3tap(
const spx_word16_t target[],       /* Target vector */
const spx_coef_t ak[],          /* LPCs for this subframe */
const spx_coef_t awk1[],        /* Weighted LPCs #1 for this subframe */
const spx_coef_t awk2[],        /* Weighted LPCs #2 for this subframe */
spx_sig_t exc[],                /* Excitation */
const signed char *gain_cdbk,
int gain_cdbk_size,
int   pitch,                    /* Pitch value */
int   p,                        /* Number of LPC coeffs */
int   nsf,                      /* Number of samples in subframe */
SpeexBits *bits,
char *stack,
const spx_word16_t *exc2,
const spx_word16_t *r,
spx_word16_t *new_target,
int  *cdbk_index,
int plc_tuning,
spx_word32_t cumul_gain,
int scaledown
)
{
   int i,j;
   VARDECL(spx_word16_t *tmp1);
   VARDECL(spx_word16_t *e);
   spx_word16_t *x[3];
   spx_word32_t corr[3];
   spx_word32_t A[3][3];
   spx_word16_t gain[3];
   spx_word32_t err;
   spx_word16_t max_gain=128;
   int          best_cdbk=0;

   ALLOC(tmp1, 3*nsf, spx_word16_t);
   ALLOC(e, nsf, spx_word16_t);

   if (cumul_gain > 262144)
      max_gain = 31;
   
   x[0]=tmp1;
   x[1]=tmp1+nsf;
   x[2]=tmp1+2*nsf;
   
   for (j=0;j<nsf;j++)
      new_target[j] = target[j];

   {
      VARDECL(spx_mem_t *mm);
      int pp=pitch-1;
      ALLOC(mm, p, spx_mem_t);
      for (j=0;j<nsf;j++)
      {
         if (j-pp<0)
            e[j]=exc2[j-pp];
         else if (j-pp-pitch<0)
            e[j]=exc2[j-pp-pitch];
         else
            e[j]=0;
      }
#ifdef FIXED_POINT
      /* Scale target and excitation down if needed (avoiding overflow) */
      if (scaledown)
      {
         for (j=0;j<nsf;j++)
            e[j] = SHR16(e[j],1);
         for (j=0;j<nsf;j++)
            new_target[j] = SHR16(new_target[j],1);
      }
#endif
      for (j=0;j<p;j++)
         mm[j] = 0;
      iir_mem16(e, ak, e, nsf, p, mm, stack);
      for (j=0;j<p;j++)
         mm[j] = 0;
      filter_mem16(e, awk1, awk2, e, nsf, p, mm, stack);
      for (j=0;j<nsf;j++)
         x[2][j] = e[j];
   }
   for (i=1;i>=0;i--)
   {
      spx_word16_t e0=exc2[-pitch-1+i];
#ifdef FIXED_POINT
      /* Scale excitation down if needed (avoiding overflow) */
      if (scaledown)
         e0 = SHR16(e0,1);
#endif
      x[i][0]=MULT16_16_Q14(r[0], e0);
      for (j=0;j<nsf-1;j++)
         x[i][j+1]=ADD32(x[i+1][j],MULT16_16_P14(r[j+1], e0));
   }

   for (i=0;i<3;i++)
      corr[i]=inner_prod(x[i],new_target,nsf);
   for (i=0;i<3;i++)
      for (j=0;j<=i;j++)
         A[i][j]=A[j][i]=inner_prod(x[i],x[j],nsf);

   {
      spx_word32_t C[9];
#ifdef FIXED_POINT
      spx_word16_t C16[9];
#else
      spx_word16_t *C16=C;
#endif      
      C[0]=corr[2];
      C[1]=corr[1];
      C[2]=corr[0];
      C[3]=A[1][2];
      C[4]=A[0][1];
      C[5]=A[0][2];      
      C[6]=A[2][2];
      C[7]=A[1][1];
      C[8]=A[0][0];
      
      /*plc_tuning *= 2;*/
      if (plc_tuning<2)
         plc_tuning=2;
      if (plc_tuning>30)
         plc_tuning=30;
#ifdef FIXED_POINT
      C[0] = SHL32(C[0],1);
      C[1] = SHL32(C[1],1);
      C[2] = SHL32(C[2],1);
      C[3] = SHL32(C[3],1);
      C[4] = SHL32(C[4],1);
      C[5] = SHL32(C[5],1);
      C[6] = MAC16_32_Q15(C[6],MULT16_16_16(plc_tuning,655),C[6]);
      C[7] = MAC16_32_Q15(C[7],MULT16_16_16(plc_tuning,655),C[7]);
      C[8] = MAC16_32_Q15(C[8],MULT16_16_16(plc_tuning,655),C[8]);
      normalize16(C, C16, 32767, 9);
#else
      C[6]*=.5*(1+.02*plc_tuning);
      C[7]*=.5*(1+.02*plc_tuning);
      C[8]*=.5*(1+.02*plc_tuning);
#endif

      best_cdbk = pitch_gain_search_3tap_vq(gain_cdbk, gain_cdbk_size, C16, max_gain);

#ifdef FIXED_POINT
      gain[0] = ADD16(32,(spx_word16_t)gain_cdbk[best_cdbk*4]);
      gain[1] = ADD16(32,(spx_word16_t)gain_cdbk[best_cdbk*4+1]);
      gain[2] = ADD16(32,(spx_word16_t)gain_cdbk[best_cdbk*4+2]);
      /*printf ("%d %d %d %d\n",gain[0],gain[1],gain[2], best_cdbk);*/
#else
      gain[0] = 0.015625*gain_cdbk[best_cdbk*4]  + .5;
      gain[1] = 0.015625*gain_cdbk[best_cdbk*4+1]+ .5;
      gain[2] = 0.015625*gain_cdbk[best_cdbk*4+2]+ .5;
#endif
      *cdbk_index=best_cdbk;
   }

   SPEEX_MEMSET(exc, 0, nsf);
   for (i=0;i<3;i++)
   {
      int j;
      int tmp1, tmp3;
      int pp=pitch+1-i;
      tmp1=nsf;
      if (tmp1>pp)
         tmp1=pp;
      for (j=0;j<tmp1;j++)
         exc[j]=MAC16_16(exc[j],SHL16(gain[2-i],7),exc2[j-pp]);
      tmp3=nsf;
      if (tmp3>pp+pitch)
         tmp3=pp+pitch;
      for (j=tmp1;j<tmp3;j++)
         exc[j]=MAC16_16(exc[j],SHL16(gain[2-i],7),exc2[j-pp-pitch]);
   }
   for (i=0;i<nsf;i++)
   {
      spx_word32_t tmp = ADD32(ADD32(MULT16_16(gain[0],x[2][i]),MULT16_16(gain[1],x[1][i])),
                            MULT16_16(gain[2],x[0][i]));
      new_target[i] = SUB16(new_target[i], EXTRACT16(PSHR32(tmp,6)));
   }
   err = inner_prod(new_target, new_target, nsf);

   return err;
}

/** Finds the best quantized 3-tap pitch predictor by analysis by synthesis */
int pitch_search_3tap(
spx_word16_t target[],                 /* Target vector */
spx_word16_t *sw,
spx_coef_t ak[],                     /* LPCs for this subframe */
spx_coef_t awk1[],                   /* Weighted LPCs #1 for this subframe */
spx_coef_t awk2[],                   /* Weighted LPCs #2 for this subframe */
spx_sig_t exc[],                    /* Excitation */
const void *par,
int   start,                    /* Smallest pitch value allowed */
int   end,                      /* Largest pitch value allowed */
spx_word16_t pitch_coef,               /* Voicing (pitch) coefficient */
int   p,                        /* Number of LPC coeffs */
int   nsf,                      /* Number of samples in subframe */
SpeexBits *bits,
char *stack,
spx_word16_t *exc2,
spx_word16_t *r,
int complexity,
int cdbk_offset,
int plc_tuning,
spx_word32_t *cumul_gain
)
{
   int i;
   int cdbk_index, pitch=0, best_gain_index=0;
   VARDECL(spx_sig_t *best_exc);
   VARDECL(spx_word16_t *new_target);
   VARDECL(spx_word16_t *best_target);
   int best_pitch=0;
   spx_word32_t err, best_err=-1;
   int N;
   const ltp_params *params;
   const signed char *gain_cdbk;
   int   gain_cdbk_size;
   int scaledown=0;
         
   VARDECL(int *nbest);
   
   params = (const ltp_params*) par;
   gain_cdbk_size = 1<<params->gain_bits;
   gain_cdbk = params->gain_cdbk + 4*gain_cdbk_size*cdbk_offset;
   
   N=complexity;
   if (N>10)
      N=10;
   if (N<1)
      N=1;

   ALLOC(nbest, N, int);
   params = (const ltp_params*) par;

   if (end<start)
   {
      speex_bits_pack(bits, 0, params->pitch_bits);
      speex_bits_pack(bits, 0, params->gain_bits);
      SPEEX_MEMSET(exc, 0, nsf);
      return start;
   }
   
#ifdef FIXED_POINT
   /* Check if we need to scale everything down in the pitch search to avoid overflows */
   for (i=0;i<nsf;i++)
   {
      if (ABS16(target[i])>16383)
      {
         scaledown=1;
         break;
      }
   }
   for (i=-end;i<nsf;i++)
   {
      if (ABS16(exc2[i])>16383)
      {
         scaledown=1;
         break;
      }
   }
#endif
   if (N>end-start+1)
      N=end-start+1;
   if (end != start)
      open_loop_nbest_pitch(sw, start, end, nsf, nbest, NULL, N, stack);
   else
      nbest[0] = start;
   
   ALLOC(best_exc, nsf, spx_sig_t);
   ALLOC(new_target, nsf, spx_word16_t);
   ALLOC(best_target, nsf, spx_word16_t);
   
   for (i=0;i<N;i++)
   {
      pitch=nbest[i];
      SPEEX_MEMSET(exc, 0, nsf);
      err=pitch_gain_search_3tap(target, ak, awk1, awk2, exc, gain_cdbk, gain_cdbk_size, pitch, p, nsf,
                                 bits, stack, exc2, r, new_target, &cdbk_index, plc_tuning, *cumul_gain, scaledown);
      if (err<best_err || best_err<0)
      {
         SPEEX_COPY(best_exc, exc, nsf);
         SPEEX_COPY(best_target, new_target, nsf);
         best_err=err;
         best_pitch=pitch;
         best_gain_index=cdbk_index;
      }
   }
   /*printf ("pitch: %d %d\n", best_pitch, best_gain_index);*/
   speex_bits_pack(bits, best_pitch-start, params->pitch_bits);
   speex_bits_pack(bits, best_gain_index, params->gain_bits);
#ifdef FIXED_POINT
   *cumul_gain = MULT16_32_Q13(SHL16(params->gain_cdbk[4*best_gain_index+3],8), MAX32(1024,*cumul_gain));
#else
   *cumul_gain = 0.03125*MAX32(1024,*cumul_gain)*params->gain_cdbk[4*best_gain_index+3];
#endif
   /*printf ("%f\n", cumul_gain);*/
   /*printf ("encode pitch: %d %d\n", best_pitch, best_gain_index);*/
   SPEEX_COPY(exc, best_exc, nsf);
   SPEEX_COPY(target, best_target, nsf);
#ifdef FIXED_POINT
   /* Scale target back up if needed */
   if (scaledown)
   {
      for (i=0;i<nsf;i++)
         target[i]=SHL16(target[i],1);
   }
#endif
   return pitch;
}

void pitch_unquant_3tap(
spx_word16_t exc[],             /* Input excitation */
spx_word32_t exc_out[],         /* Output excitation */
int   start,                    /* Smallest pitch value allowed */
int   end,                      /* Largest pitch value allowed */
spx_word16_t pitch_coef,        /* Voicing (pitch) coefficient */
const void *par,
int   nsf,                      /* Number of samples in subframe */
int *pitch_val,
spx_word16_t *gain_val,
SpeexBits *bits,
char *stack,
int count_lost,
int subframe_offset,
spx_word16_t last_pitch_gain,
int cdbk_offset
)
{
   int i;
   int pitch;
   int gain_index;
   spx_word16_t gain[3];
   const signed char *gain_cdbk;
   int gain_cdbk_size;
   const ltp_params *params;

   params = (const ltp_params*) par;
   gain_cdbk_size = 1<<params->gain_bits;
   gain_cdbk = params->gain_cdbk + 4*gain_cdbk_size*cdbk_offset;

   pitch = speex_bits_unpack_unsigned(bits, params->pitch_bits);
   pitch += start;
   gain_index = speex_bits_unpack_unsigned(bits, params->gain_bits);
   /*printf ("decode pitch: %d %d\n", pitch, gain_index);*/
#ifdef FIXED_POINT
   gain[0] = ADD16(32,(spx_word16_t)gain_cdbk[gain_index*4]);
   gain[1] = ADD16(32,(spx_word16_t)gain_cdbk[gain_index*4+1]);
   gain[2] = ADD16(32,(spx_word16_t)gain_cdbk[gain_index*4+2]);
#else
   gain[0] = 0.015625*gain_cdbk[gain_index*4]+.5;
   gain[1] = 0.015625*gain_cdbk[gain_index*4+1]+.5;
   gain[2] = 0.015625*gain_cdbk[gain_index*4+2]+.5;
#endif

   if (count_lost && pitch > subframe_offset)
   {
      spx_word16_t gain_sum;
      if (1) {
#ifdef FIXED_POINT
         spx_word16_t tmp = count_lost < 4 ? last_pitch_gain : SHR16(last_pitch_gain,1);
         if (tmp>62)
            tmp=62;
#else
         spx_word16_t tmp = count_lost < 4 ? last_pitch_gain : 0.5 * last_pitch_gain;
         if (tmp>.95)
            tmp=.95;
#endif
         gain_sum = gain_3tap_to_1tap(gain);

         if (gain_sum > tmp)
         {
            spx_word16_t fact = DIV32_16(SHL32(EXTEND32(tmp),14),gain_sum);
            for (i=0;i<3;i++)
               gain[i]=MULT16_16_Q14(fact,gain[i]);
         }

      }

   }

   *pitch_val = pitch;
   gain_val[0]=gain[0];
   gain_val[1]=gain[1];
   gain_val[2]=gain[2];
   gain[0] = SHL16(gain[0],7);
   gain[1] = SHL16(gain[1],7);
   gain[2] = SHL16(gain[2],7);
   SPEEX_MEMSET(exc_out, 0, nsf);
   for (i=0;i<3;i++)
   {
      int j;
      int tmp1, tmp3;
      int pp=pitch+1-i;
      tmp1=nsf;
      if (tmp1>pp)
         tmp1=pp;
      for (j=0;j<tmp1;j++)
         exc_out[j]=MAC16_16(exc_out[j],gain[2-i],exc[j-pp]);
      tmp3=nsf;
      if (tmp3>pp+pitch)
         tmp3=pp+pitch;
      for (j=tmp1;j<tmp3;j++)
         exc_out[j]=MAC16_16(exc_out[j],gain[2-i],exc[j-pp-pitch]);
   }
   /*for (i=0;i<nsf;i++)
   exc[i]=PSHR32(exc32[i],13);*/
}


/** Forced pitch delay and gain */
int forced_pitch_quant(
spx_word16_t target[],                 /* Target vector */
spx_word16_t *sw,
spx_coef_t ak[],                     /* LPCs for this subframe */
spx_coef_t awk1[],                   /* Weighted LPCs #1 for this subframe */
spx_coef_t awk2[],                   /* Weighted LPCs #2 for this subframe */
spx_sig_t exc[],                    /* Excitation */
const void *par,
int   start,                    /* Smallest pitch value allowed */
int   end,                      /* Largest pitch value allowed */
spx_word16_t pitch_coef,               /* Voicing (pitch) coefficient */
int   p,                        /* Number of LPC coeffs */
int   nsf,                      /* Number of samples in subframe */
SpeexBits *bits,
char *stack,
spx_word16_t *exc2,
spx_word16_t *r,
int complexity,
int cdbk_offset,
int plc_tuning,
spx_word32_t *cumul_gain
)
{
   int i;
   VARDECL(spx_word16_t *res);
   ALLOC(res, nsf, spx_word16_t);
#ifdef FIXED_POINT
   if (pitch_coef>63)
      pitch_coef=63;
#else
   if (pitch_coef>.99)
      pitch_coef=.99;
#endif
   for (i=0;i<nsf&&i<start;i++)
   {
      exc[i]=MULT16_16(SHL16(pitch_coef, 7),exc2[i-start]);
   }
   for (;i<nsf;i++)
   {
      exc[i]=MULT16_32_Q15(SHL16(pitch_coef, 9),exc[i-start]);
   }
   for (i=0;i<nsf;i++)
      res[i] = EXTRACT16(PSHR32(exc[i], SIG_SHIFT-1));
   syn_percep_zero16(res, ak, awk1, awk2, res, nsf, p, stack);
   for (i=0;i<nsf;i++)
      target[i]=EXTRACT16(SATURATE(SUB32(EXTEND32(target[i]),EXTEND32(res[i])),32700));
   return start;
}

/** Unquantize forced pitch delay and gain */
void forced_pitch_unquant(
spx_word16_t exc[],             /* Input excitation */
spx_word32_t exc_out[],         /* Output excitation */
int   start,                    /* Smallest pitch value allowed */
int   end,                      /* Largest pitch value allowed */
spx_word16_t pitch_coef,        /* Voicing (pitch) coefficient */
const void *par,
int   nsf,                      /* Number of samples in subframe */
int *pitch_val,
spx_word16_t *gain_val,
SpeexBits *bits,
char *stack,
int count_lost,
int subframe_offset,
spx_word16_t last_pitch_gain,
int cdbk_offset
)
{
   int i;
#ifdef FIXED_POINT
   if (pitch_coef>63)
      pitch_coef=63;
#else
   if (pitch_coef>.99)
      pitch_coef=.99;
#endif
   for (i=0;i<nsf;i++)
   {
      exc_out[i]=MULT16_16(exc[i-start],SHL16(pitch_coef,7));
      exc[i] = EXTRACT16(PSHR32(exc_out[i],13));
   }
   *pitch_val = start;
   gain_val[0]=gain_val[2]=0;
   gain_val[1] = pitch_coef;
}
