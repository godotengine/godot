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

#if defined(FIXED_POINT) && defined(__mips)
#include "mips/celt_mipsr1.h"
#endif


int resampling_factor(opus_int32 rate)
{
   int ret;
   switch (rate)
   {
#ifdef ENABLE_QEXT
   case 96000:
#endif
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
      celt_coef g10, celt_coef g11, celt_coef g12)
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
      t = MAC_COEF_32_ARM(x[i], g10, x2);
      t = MAC_COEF_32_ARM(t, g11, ADD32(x1,x3));
      t = MAC_COEF_32_ARM(t, g12, ADD32(x0,x4));
      t = SATURATE(t, SIG_SAT);
      y[i] = t;
      x4=SHL32(x[i-T+3],1);
      t = MAC_COEF_32_ARM(x[i+1], g10, x1);
      t = MAC_COEF_32_ARM(t, g11, ADD32(x0,x2));
      t = MAC_COEF_32_ARM(t, g12, ADD32(x4,x3));
      t = SATURATE(t, SIG_SAT);
      y[i+1] = t;
      x3=SHL32(x[i-T+4],1);
      t = MAC_COEF_32_ARM(x[i+2], g10, x0);
      t = MAC_COEF_32_ARM(t, g11, ADD32(x4,x1));
      t = MAC_COEF_32_ARM(t, g12, ADD32(x3,x2));
      t = SATURATE(t, SIG_SAT);
      y[i+2] = t;
      x2=SHL32(x[i-T+5],1);
      t = MAC_COEF_32_ARM(x[i+3], g10, x4);
      t = MAC_COEF_32_ARM(t, g11, ADD32(x3,x0));
      t = MAC_COEF_32_ARM(t, g12, ADD32(x2,x1));
      t = SATURATE(t, SIG_SAT);
      y[i+3] = t;
      x1=SHL32(x[i-T+6],1);
      t = MAC_COEF_32_ARM(x[i+4], g10, x3);
      t = MAC_COEF_32_ARM(t, g11, ADD32(x2,x4));
      t = MAC_COEF_32_ARM(t, g12, ADD32(x1,x0));
      t = SATURATE(t, SIG_SAT);
      y[i+4] = t;
   }
#ifdef CUSTOM_MODES
   for (;i<N;i++)
   {
      opus_val32 t;
      x0=SHL32(x[i-T+2],1);
      t = MAC_COEF_32_ARM(x[i], g10, x2);
      t = MAC_COEF_32_ARM(t, g11, ADD32(x1,x3));
      t = MAC_COEF_32_ARM(t, g12, ADD32(x0,x4));
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
      celt_coef g10, celt_coef g11, celt_coef g12)
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
               + MULT_COEF_32(g10,x2)
               + MULT_COEF_32(g11,ADD32(x1,x3))
               + MULT_COEF_32(g12,ADD32(x0,x4));
#ifdef FIXED_POINT
      /* A bit of bias seems to help here. */
      y[i] = SUB32(y[i], 1);
#endif
      y[i] = SATURATE(y[i], SIG_SAT);
      x4=x3;
      x3=x2;
      x2=x1;
      x1=x0;
   }

}
#endif
#endif

#ifdef ENABLE_QEXT
void comb_filter_qext(opus_val32 *y, opus_val32 *x, int T0, int T1, int N,
      opus_val16 g0, opus_val16 g1, int tapset0, int tapset1,
      const celt_coef *window, int overlap, int arch)
{
   VARDECL(opus_val32, mem_buf);
   VARDECL(opus_val32, buf);
   celt_coef new_window[120];
   int s;
   int i;
   int N2;
   int overlap2;
   SAVE_STACK;
   /* Using ALLOC() instead of a regular stack allocation to minimize real stack use when using the pseudostack.
      This is useful on some embedded systems. */
   ALLOC(mem_buf, COMBFILTER_MAXPERIOD+960, opus_val32);
   ALLOC(buf, COMBFILTER_MAXPERIOD+960, opus_val32);
   N2 = N/2;
   overlap2=overlap/2;
   /* At 96 kHz, we double the period and the spacing between taps, which is equivalent
      to creating a mirror image of the filter around 24 kHz. It also means we can process
      the even and odd samples completely independently. */
   for (s=0;s<2;s++) {
      opus_val32 *yptr;
      for (i=0;i<overlap2;i++) new_window[i] = window[2*i+s];
      for (i=0;i<COMBFILTER_MAXPERIOD+N2;i++) mem_buf[i] = x[2*i+s-2*COMBFILTER_MAXPERIOD];
      if (x==y) {
         yptr = mem_buf+COMBFILTER_MAXPERIOD;
      } else {
         for (i=0;i<N2;i++) buf[i] = y[2*i+s];
         yptr = buf;
      }
      comb_filter(yptr, mem_buf+COMBFILTER_MAXPERIOD, T0, T1, N2, g0, g1, tapset0, tapset1, new_window, overlap2, arch);
      for (i=0;i<N2;i++) y[2*i+s] = yptr[i];
   }
   RESTORE_STACK;
   return;
}
#endif

#ifndef OVERRIDE_comb_filter
void comb_filter(opus_val32 *y, opus_val32 *x, int T0, int T1, int N,
      opus_val16 g0, opus_val16 g1, int tapset0, int tapset1,
      const celt_coef *window, int overlap, int arch)
{
   int i;
   /* printf ("%d %d %f %f\n", T0, T1, g0, g1); */
   celt_coef g00, g01, g02, g10, g11, g12;
   opus_val32 x0, x1, x2, x3, x4;
   static const opus_val16 gains[3][3] = {
         {QCONST16(0.3066406250f, 15), QCONST16(0.2170410156f, 15), QCONST16(0.1296386719f, 15)},
         {QCONST16(0.4638671875f, 15), QCONST16(0.2680664062f, 15), QCONST16(0.f, 15)},
         {QCONST16(0.7998046875f, 15), QCONST16(0.1000976562f, 15), QCONST16(0.f, 15)}};
#ifdef ENABLE_QEXT
   if (overlap==240) {
      comb_filter_qext(y, x, T0, T1, N, g0, g1, tapset0, tapset1, window, overlap, arch);
      return;
   }
#endif
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
   g00 = MULT_COEF_TAPS(g0, gains[tapset0][0]);
   g01 = MULT_COEF_TAPS(g0, gains[tapset0][1]);
   g02 = MULT_COEF_TAPS(g0, gains[tapset0][2]);
   g10 = MULT_COEF_TAPS(g1, gains[tapset1][0]);
   g11 = MULT_COEF_TAPS(g1, gains[tapset1][1]);
   g12 = MULT_COEF_TAPS(g1, gains[tapset1][2]);
   x1 = x[-T1+1];
   x2 = x[-T1  ];
   x3 = x[-T1-1];
   x4 = x[-T1-2];
   /* If the filter didn't change, we don't need the overlap */
   if (g0==g1 && T0==T1 && tapset0==tapset1)
      overlap=0;
   for (i=0;i<overlap;i++)
   {
      celt_coef f;
      x0=x[i-T1+2];
      f = MULT_COEF(window[i],window[i]);
      y[i] = x[i]
               + MULT_COEF_32(MULT_COEF((COEF_ONE-f),g00),x[i-T0])
               + MULT_COEF_32(MULT_COEF((COEF_ONE-f),g01),ADD32(x[i-T0+1],x[i-T0-1]))
               + MULT_COEF_32(MULT_COEF((COEF_ONE-f),g02),ADD32(x[i-T0+2],x[i-T0-2]))
               + MULT_COEF_32(MULT_COEF(f,g10),x2)
               + MULT_COEF_32(MULT_COEF(f,g11),ADD32(x1,x3))
               + MULT_COEF_32(MULT_COEF(f,g12),ADD32(x0,x4));
#ifdef FIXED_POINT
      /* A bit of bias seems to help here. */
      y[i] = SUB32(y[i], 3);
#endif
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
