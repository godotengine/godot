/* Copyright (C) 2005 Jean-Marc Valin, CSIRO, Christopher Montgomery
   File: vorbis_psy.h

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

#ifndef VORBIS_PSY_H
#define VORBIS_PSY_H

#ifdef VORBIS_PSYCHO

#include "smallft.h"
#define P_BANDS 17      /* 62Hz to 16kHz */
#define NOISE_COMPAND_LEVELS 40


#define todB(x)   ((x)>1e-13?log((x)*(x))*4.34294480f:-30)
#define fromdB(x) (exp((x)*.11512925f))  

/* The bark scale equations are approximations, since the original
   table was somewhat hand rolled.  The below are chosen to have the
   best possible fit to the rolled tables, thus their somewhat odd
   appearance (these are more accurate and over a longer range than
   the oft-quoted bark equations found in the texts I have).  The
   approximations are valid from 0 - 30kHz (nyquist) or so.

   all f in Hz, z in Bark */

#define toBARK(n)   (13.1f*atan(.00074f*(n))+2.24f*atan((n)*(n)*1.85e-8f)+1e-4f*(n))
#define fromBARK(z) (102.f*(z)-2.f*pow(z,2.f)+.4f*pow(z,3.f)+pow(1.46f,z)-1.f)

/* Frequency to octave.  We arbitrarily declare 63.5 Hz to be octave
   0.0 */

#define toOC(n)     (log(n)*1.442695f-5.965784f)
#define fromOC(o)   (exp(((o)+5.965784f)*.693147f))


typedef struct {

  float noisewindowlo;
  float noisewindowhi;
  int   noisewindowlomin;
  int   noisewindowhimin;
  int   noisewindowfixed;
  float noiseoff[P_BANDS];
  float noisecompand[NOISE_COMPAND_LEVELS];

} VorbisPsyInfo;



typedef struct {
  int n;
  int rate;
  struct drft_lookup lookup;
  VorbisPsyInfo *vi;

  float *window;
  float *noiseoffset;
  long  *bark;

} VorbisPsy;


VorbisPsy *vorbis_psy_init(int rate, int size);
void vorbis_psy_destroy(VorbisPsy *psy);
void compute_curve(VorbisPsy *psy, float *audio, float *curve);
void curve_to_lpc(VorbisPsy *psy, float *curve, float *awk1, float *awk2, int ord);

#endif
#endif
