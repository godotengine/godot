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

#ifdef OPUS_ENABLED
#include "opus/opus_config.h"
#endif

#define CELT_C

#include "opus/celt/os_support.h"
#include "opus/celt/mdct.h"
#include <math.h>
#include "opus/celt/celt.h"
#include "opus/celt/pitch.h"
#include "opus/celt/bands.h"
#include "opus/celt/opus_modes.h"
#include "opus/celt/entcode.h"
#include "opus/celt/quant_bands.h"
#include "opus/celt/rate.h"
#include "opus/celt/stack_alloc.h"
#include "opus/celt/mathops.h"
#include "opus/celt/float_cast.h"
#include <stdarg.h>
#include "opus/celt/celt_lpc.h"
#include "opus/celt/vq.h"

#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION "unknown"
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

#ifndef OVERRIDE_COMB_FILTER_CONST
static void comb_filter_const(opus_val32 *y, opus_val32 *x, int T, int N,
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
      x4=x3;
      x3=x2;
      x2=x1;
      x1=x0;
   }

}
#endif

void comb_filter(opus_val32 *y, opus_val32 *x, int T0, int T1, int N,
      opus_val16 g0, opus_val16 g1, int tapset0, int tapset1,
      const opus_val16 *window, int overlap)
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
   g00 = MULT16_16_Q15(g0, gains[tapset0][0]);
   g01 = MULT16_16_Q15(g0, gains[tapset0][1]);
   g02 = MULT16_16_Q15(g0, gains[tapset0][2]);
   g10 = MULT16_16_Q15(g1, gains[tapset1][0]);
   g11 = MULT16_16_Q15(g1, gains[tapset1][1]);
   g12 = MULT16_16_Q15(g1, gains[tapset1][2]);
   x1 = x[-T1+1];
   x2 = x[-T1  ];
   x3 = x[-T1-1];
   x4 = x[-T1-2];
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
   comb_filter_const(y+i, x+i, T1, N-i, g10, g11, g12);
}

const signed char tf_select_table[4][8] = {
      {0, -1, 0, -1,    0,-1, 0,-1},
      {0, -1, 0, -2,    1, 0, 1,-1},
      {0, -2, 0, -3,    2, 0, 1,-1},
      {0, -2, 0, -3,    3, 0, 1,-1},
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
#ifdef OPUS_FIXED_POINT
          "-fixed"
#endif
#ifdef FUZZING
          "-fuzzing"
#endif
          ;
}
