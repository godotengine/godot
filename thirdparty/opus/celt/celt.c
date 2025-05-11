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

#define CELT_C

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

#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION "unknown"
#endif

#if defined(MIPSr1_ASM)
#include "mips/celt_mipsr1.h"
#endif


int resampling_factor(opus_int32 rate)
{
   int ret;
   switch (rate)
   {
   case 48000:
      ret = 1;
      break;
   case 24000:
      ret = 2;
      break;
   case 16000:
      ret = 3;
      break;
   case 12000:
      ret = 4;
      break;
   case 8000:
      ret = 6;
      break;
   default:
#ifndef CUSTOM_MODES
      celt_assert(0);
#endif
      ret = 0;
      break;
   }
   return ret;
}

#if !defined(OVERRIDE_COMB_FILTER_CONST) || defined(NON_STATIC_COMB_FILTER_CONST_C)
/* This version should be faster on ARM */
#ifdef OPUS_ARM_ASM
#ifndef NON_STATIC_COMB_FILTER_CONST_C
static
#endif
void comb_filter_const_c(opus_val32 *y, opus_val32 *x, int T, int N,
      opus_val16 g10, opus_val16 g11, opus_val16 g12)
{
   opus_val32 x0, x1, x2, x3, x4;
   int i;
   x4 = SHL32(x[-T-2], 1);
   x3 = SHL32(x[-T-1], 1);
   x2 = SHL32(x[-T], 1);
   x1 = SHL32(x[-T+1], 1);
   for (i=0;i<N-4;i+=5)
   {
      opus_val32 t;
      x0=SHL32(x[i-T+2],1);
      t = MAC16_32_Q16(x[i], g10, x2);
      t = MAC16_32_Q16(t, g11, ADD32(x1,x3));
      t = MAC16_32_Q16(t, g12, ADD32(x0,x4));
      t = SATURATE(t, SIG_SAT);
      y[i] = t;
      x4=SHL32(x[i-T+3],1);
      t = MAC16_32_Q16(x[i+1], g10, x1);
      t = MAC16_32_Q16(t, g11, ADD32(x0,x2));
      t = MAC16_32_Q16(t, g12, ADD32(x4,x3));
      t = SATURATE(t, SIG_SAT);
      y[i+1] = t;
      x3=SHL32(x[i-T+4],1);
      t = MAC16_32_Q16(x[i+2], g10, x0);
      t = MAC16_32_Q16(t, g11, ADD32(x4,x1));
      t = MAC16_32_Q16(t, g12, ADD32(x3,x2));
      t = SATURATE(t, SIG_SAT);
      y[i+2] = t;
      x2=SHL32(x[i-T+5],1);
      t = MAC16_32_Q16(x[i+3], g10, x4);
      t = MAC16_32_Q16(t, g11, ADD32(x3,x0));
      t = MAC16_32_Q16(t, g12, ADD32(x2,x1));
      t = SATURATE(t, SIG_SAT);
      y[i+3] = t;
      x1=SHL32(x[i-T+6],1);
      t = MAC16_32_Q16(x[i+4], g10, x3);
      t = MAC16_32_Q16(t, g11, ADD32(x2,x4));
      t = MAC16_32_Q16(t, g12, ADD32(x1,x0));
      t = SATURATE(t, SIG_SAT);
      y[i+4] = t;
   }
#ifdef CUSTOM_MODES
   for (;i<N;i++)
   {
      opus_val32 t;
      x0=SHL32(x[i-T+2],1);
      t = MAC16_32_Q16(x[i], g10, x2);
      t = MAC16_32_Q16(t, g11, ADD32(x1,x3));
      t = MAC16_32_Q16(t, g12, ADD32(x0,x4));
      t = SATURATE(t, SIG_SAT);
      y[i] = t;
      x4=x3;
      x3=x2;
      x2=x1;
      x1=x0;
   }
#endif
}
#else
#ifndef NON_STATIC_COMB_FILTER_CONST_C
static
#endif
void comb_filter_const_c(opus_val32 *y, opus_val32 *x, int T, int N,
      opus_val16 g10, opus_val16 g11, opus_val16 g12)
{
   opus_val32 x0, x1, x2, x3, x4;
   int i;
   x4 = x[-T-2];
   x3 = x[-T-1];
   x2 = x[-T];
   x1 = x[-T+1];
   for (i=0;i<N;i++)
   {
      x0=x[i-T+2];
      y[i] = x[i]
               + MULT16_32_Q15(g10,x2)
               + MULT16_32_Q15(g11,ADD32(x1,x3))
               + MULT16_32_Q15(g12,ADD32(x0,x4));
      y[i] = SATURATE(y[i], SIG_SAT);
      x4=x3;
      x3=x2;
      x2=x1;
      x1=x0;
   }

}
#endif
#endif

#ifndef OVERRIDE_comb_filter
void comb_filter(opus_val32 *y, opus_val32 *x, int T0, int T1, int N,
      opus_val16 g0, opus_val16 g1, int tapset0, int tapset1,
      const opus_val16 *window, int overlap, int arch)
{
   int i;
   /* printf ("%d %d %f %f\n", T0, T1, g0, g1); */
   opus_val16 g00, g01, g02, g10, g11, g12;
   opus_val32 x0, x1, x2, x3, x4;
   static const opus_val16 gains[3][3] = {
         {QCONST16(0.3066406250f, 15), QCONST16(0.2170410156f, 15), QCONST16(0.1296386719f, 15)},
         {QCONST16(0.4638671875f, 15), QCONST16(0.2680664062f, 15), QCONST16(0.f, 15)},
         {QCONST16(0.7998046875f, 15), QCONST16(0.1000976562f, 15), QCONST16(0.f, 15)}};

   if (g0==0 && g1==0)
   {
      /* OPT: Happens to work without the OPUS_MOVE(), but only because the current encoder already copies x to y */
      if (x!=y)
         OPUS_MOVE(y, x, N);
      return;
   }
   /* When the gain is zero, T0 and/or T1 is set to zero. We need
      to have then be at least 2 to avoid processing garbage data. */
   T0 = IMAX(T0, COMBFILTER_MINPERIOD);
   T1 = IMAX(T1, COMBFILTER_MINPERIOD);
   g00 = MULT16_16_P15(g0, gains[tapset0][0]);
   g01 = MULT16_16_P15(g0, gains[tapset0][1]);
   g02 = MULT16_16_P15(g0, gains[tapset0][2]);
   g10 = MULT16_16_P15(g1, gains[tapset1][0]);
   g11 = MULT16_16_P15(g1, gains[tapset1][1]);
   g12 = MULT16_16_P15(g1, gains[tapset1][2]);
   x1 = x[-T1+1];
   x2 = x[-T1  ];
   x3 = x[-T1-1];
   x4 = x[-T1-2];
   /* If the filter didn't change, we don't need the overlap */
   if (g0==g1 && T0==T1 && tapset0==tapset1)
      overlap=0;
   for (i=0;i<overlap;i++)
   {
      opus_val16 f;
      x0=x[i-T1+2];
      f = MULT16_16_Q15(window[i],window[i]);
      y[i] = x[i]
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g00),x[i-T0])
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g01),ADD32(x[i-T0+1],x[i-T0-1]))
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g02),ADD32(x[i-T0+2],x[i-T0-2]))
               + MULT16_32_Q15(MULT16_16_Q15(f,g10),x2)
               + MULT16_32_Q15(MULT16_16_Q15(f,g11),ADD32(x1,x3))
               + MULT16_32_Q15(MULT16_16_Q15(f,g12),ADD32(x0,x4));
      y[i] = SATURATE(y[i], SIG_SAT);
      x4=x3;
      x3=x2;
      x2=x1;
      x1=x0;

   }
   if (g1==0)
   {
      /* OPT: Happens to work without the OPUS_MOVE(), but only because the current encoder already copies x to y */
      if (x!=y)
         OPUS_MOVE(y+overlap, x+overlap, N-overlap);
      return;
   }

   /* Compute the part with the constant filter. */
   comb_filter_const(y+i, x+i, T1, N-i, g10, g11, g12, arch);
}
#endif /* OVERRIDE_comb_filter */

/* TF change table. Positive values mean better frequency resolution (longer
   effective window), whereas negative values mean better time resolution
   (shorter effective window). The second index is computed as:
   4*isTransient + 2*tf_select + per_band_flag */
const signed char tf_select_table[4][8] = {
    /*isTransient=0     isTransient=1 */
      {0, -1, 0, -1,    0,-1, 0,-1}, /* 2.5 ms */
      {0, -1, 0, -2,    1, 0, 1,-1}, /* 5 ms */
      {0, -2, 0, -3,    2, 0, 1,-1}, /* 10 ms */
      {0, -2, 0, -3,    3, 0, 1,-1}, /* 20 ms */
};


void init_caps(const CELTMode *m,int *cap,int LM,int C)
{
   int i;
   for (i=0;i<m->nbEBands;i++)
   {
      int N;
      N=(m->eBands[i+1]-m->eBands[i])<<LM;
      cap[i] = (m->cache.caps[m->nbEBands*(2*LM+C-1)+i]+64)*C*N>>2;
   }
}



const char *opus_strerror(int error)
{
   static const char * const error_strings[8] = {
      "success",
      "invalid argument",
      "buffer too small",
      "internal error",
      "corrupted stream",
      "request not implemented",
      "invalid state",
      "memory allocation failed"
   };
   if (error > 0 || error < -7)
      return "unknown error";
   else
      return error_strings[-error];
}

const char *opus_get_version_string(void)
{
    return "libopus " PACKAGE_VERSION
    /* Applications may rely on the presence of this substring in the version
       string to determine if they have a fixed-point or floating-point build
       at runtime. */
#ifdef FIXED_POINT
          "-fixed"
#endif
#ifdef FUZZING
          "-fuzzing"
#endif
          ;
}
