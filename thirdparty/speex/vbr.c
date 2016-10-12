/* Copyright (C) 2002 Jean-Marc Valin 
   File: vbr.c

   VBR-related routines

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


#include "vbr.h"
#include <math.h>


#define sqr(x) ((x)*(x))

#define MIN_ENERGY 6000
#define NOISE_POW .3

#ifndef DISABLE_VBR

const float vbr_nb_thresh[9][11]={
   {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f}, /*   CNG   */
   { 4.0f,  2.5f,  2.0f,  1.2f,  0.5f,  0.0f, -0.5f, -0.7f, -0.8f, -0.9f, -1.0f}, /*  2 kbps */
   {10.0f,  6.5f,  5.2f,  4.5f,  3.9f,  3.5f,  3.0f,  2.5f,  2.3f,  1.8f,  1.0f}, /*  6 kbps */
   {11.0f,  8.8f,  7.5f,  6.5f,  5.0f,  3.9f,  3.9f,  3.9f,  3.5f,  3.0f,  1.0f}, /*  8 kbps */
   {11.0f, 11.0f,  9.9f,  8.5f,  7.0f,  6.0f,  4.5f,  4.0f,  4.0f,  4.0f,  2.0f}, /* 11 kbps */
   {11.0f, 11.0f, 11.0f, 11.0f,  9.5f,  8.5f,  8.0f,  7.0f,  6.0f,  5.0f,  3.0f}, /* 15 kbps */
   {11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f,  9.5f,  8.5f,  7.0f,  6.0f,  5.0f}, /* 18 kbps */
   {11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f,  9.8f,  9.5f,  7.5f}, /* 24 kbps */ 
   { 7.0f,  4.5f,  3.7f,  3.0f,  2.5f,  2.0f,  1.8f,  1.5f,  1.0f,  0.0f,  0.0f}  /*  4 kbps */
};


const float vbr_hb_thresh[5][11]={
   {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f}, /* silence */
   {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f}, /*  2 kbps */
   {11.0f, 11.0f,  9.5f,  8.5f,  7.5f,  6.0f,  5.0f,  3.9f,  3.0f,  2.0f,  1.0f}, /*  6 kbps */
   {11.0f, 11.0f, 11.0f, 11.0f, 11.0f,  9.5f,  8.7f,  7.8f,  7.0f,  6.5f,  4.0f}, /* 10 kbps */
   {11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f,  9.8f,  7.5f,  5.5f}  /* 18 kbps */ 
};

const float vbr_uhb_thresh[2][11]={
   {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f}, /* silence */
   { 3.9f,  2.5f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f, -1.0f}  /*  2 kbps */
};

void vbr_init(VBRState *vbr)
{
   int i;

   vbr->average_energy=0;
   vbr->last_energy=1;
   vbr->accum_sum=0;
   vbr->energy_alpha=.1;
   vbr->soft_pitch=0;
   vbr->last_pitch_coef=0;
   vbr->last_quality=0;

   vbr->noise_accum = .05*pow(MIN_ENERGY, NOISE_POW);
   vbr->noise_accum_count=.05;
   vbr->noise_level=vbr->noise_accum/vbr->noise_accum_count;
   vbr->consec_noise=0;


   for (i=0;i<VBR_MEMORY_SIZE;i++)
      vbr->last_log_energy[i] = log(MIN_ENERGY);
}


/*
  This function should analyse the signal and decide how critical the
  coding error will be perceptually. The following factors should be
  taken into account:

  -Attacks (positive energy derivative) should be coded with more bits

  -Stationary voiced segments should receive more bits

  -Segments with (very) low absolute energy should receive less bits (maybe
  only shaped noise?)

  -DTX for near-zero energy?

  -Stationary fricative segments should have less bits

  -Temporal masking: when energy slope is decreasing, decrease the bit-rate

  -Decrease bit-rate for males (low pitch)?

  -(wideband only) less bits in the high-band when signal is very 
  non-stationary (harder to notice high-frequency noise)???

*/

float vbr_analysis(VBRState *vbr, spx_word16_t *sig, int len, int pitch, float pitch_coef)
{
   int i;
   float ener=0, ener1=0, ener2=0;
   float qual=7;
   int va;
   float log_energy;
   float non_st=0;
   float voicing;
   float pow_ener;

   for (i=0;i<len>>1;i++)
      ener1 += ((float)sig[i])*sig[i];

   for (i=len>>1;i<len;i++)
      ener2 += ((float)sig[i])*sig[i];
   ener=ener1+ener2;

   log_energy = log(ener+MIN_ENERGY);
   for (i=0;i<VBR_MEMORY_SIZE;i++)
      non_st += sqr(log_energy-vbr->last_log_energy[i]);
   non_st =  non_st/(30*VBR_MEMORY_SIZE);
   if (non_st>1)
      non_st=1;

   voicing = 3*(pitch_coef-.4)*fabs(pitch_coef-.4);
   vbr->average_energy = (1-vbr->energy_alpha)*vbr->average_energy + vbr->energy_alpha*ener;
   vbr->noise_level=vbr->noise_accum/vbr->noise_accum_count;
   pow_ener = pow(ener,NOISE_POW);
   if (vbr->noise_accum_count<.06 && ener>MIN_ENERGY)
      vbr->noise_accum = .05*pow_ener;

   if ((voicing<.3 && non_st < .2 && pow_ener < 1.2*vbr->noise_level)
       || (voicing<.3 && non_st < .05 && pow_ener < 1.5*vbr->noise_level)
       || (voicing<.4 && non_st < .05 && pow_ener < 1.2*vbr->noise_level)
       || (voicing<0 && non_st < .05))
   {
      float tmp;
      va = 0;
      vbr->consec_noise++;
      if (pow_ener > 3*vbr->noise_level)
         tmp = 3*vbr->noise_level;
      else 
         tmp = pow_ener;
      if (vbr->consec_noise>=4)
      {
         vbr->noise_accum = .95*vbr->noise_accum + .05*tmp;
         vbr->noise_accum_count = .95*vbr->noise_accum_count + .05;
      }
   } else {
      va = 1;
      vbr->consec_noise=0;
   }

   if (pow_ener < vbr->noise_level && ener>MIN_ENERGY)
   {
      vbr->noise_accum = .95*vbr->noise_accum + .05*pow_ener;
      vbr->noise_accum_count = .95*vbr->noise_accum_count + .05;      
   }

   /* Checking for very low absolute energy */
   if (ener < 30000)
   {
      qual -= .7;
      if (ener < 10000)
         qual-=.7;
      if (ener < 3000)
         qual-=.7;
   } else {
      float short_diff, long_diff;
      short_diff = log((ener+1)/(1+vbr->last_energy));
      long_diff = log((ener+1)/(1+vbr->average_energy));
      /*fprintf (stderr, "%f %f\n", short_diff, long_diff);*/

      if (long_diff<-5)
         long_diff=-5;
      if (long_diff>2)
         long_diff=2;

      if (long_diff>0)
         qual += .6*long_diff;
      if (long_diff<0)
         qual += .5*long_diff;
      if (short_diff>0)
      {
         if (short_diff>5)
            short_diff=5;
         qual += .5*short_diff;
      }
      /* Checking for energy increases */
      if (ener2 > 1.6*ener1)
         qual += .5;
   }
   vbr->last_energy = ener;
   vbr->soft_pitch = .6*vbr->soft_pitch + .4*pitch_coef;
   qual += 2.2*((pitch_coef-.4) + (vbr->soft_pitch-.4));

   if (qual < vbr->last_quality)
      qual = .5*qual + .5*vbr->last_quality;
   if (qual<4)
      qual=4;
   if (qual>10)
      qual=10;
   
   /*
   if (vbr->consec_noise>=2)
      qual-=1.3;
   if (vbr->consec_noise>=5)
      qual-=1.3;
   if (vbr->consec_noise>=12)
      qual-=1.3;
   */
   if (vbr->consec_noise>=3)
      qual=4;

   if (vbr->consec_noise)
      qual -= 1.0 * (log(3.0 + vbr->consec_noise)-log(3));
   if (qual<0)
      qual=0;
   
   if (ener<60000)
   {
      if (vbr->consec_noise>2)
         qual-=0.5*(log(3.0 + vbr->consec_noise)-log(3));
      if (ener<10000&&vbr->consec_noise>2)
         qual-=0.5*(log(3.0 + vbr->consec_noise)-log(3));
      if (qual<0)
         qual=0;
      qual += .3*log(.0001+ener/60000.0);
   }
   if (qual<-1)
      qual=-1;

   /*printf ("%f %f %f %f %d\n", qual, voicing, non_st, pow_ener/(.01+vbr->noise_level), va);*/

   vbr->last_pitch_coef = pitch_coef;
   vbr->last_quality = qual;

   for (i=VBR_MEMORY_SIZE-1;i>0;i--)
      vbr->last_log_energy[i] = vbr->last_log_energy[i-1];
   vbr->last_log_energy[0] = log_energy;

   /*printf ("VBR: %f %f %f %d %f\n", (float)(log_energy-log(vbr->average_energy+MIN_ENERGY)), non_st, voicing, va, vbr->noise_level);*/

   return qual;
}

void vbr_destroy(VBRState *vbr)
{
}

#endif /* #ifndef DISABLE_VBR */
