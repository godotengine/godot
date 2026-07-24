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

#ifndef CELT_MIPSR1_H__
#define CELT_MIPSR1_H__

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define CELT_C

#if defined (__mips_dsp) && __mips == 32

#define OVERRIDE_COMB_FILTER_CONST
#define OVERRIDE_comb_filter
#elif defined(__mips_isa_rev) && __mips_isa_rev < 6

#define OVERRIDE_COMB_FILTER_CONST
#define OVERRIDE_comb_filter
#endif

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

#if defined (__mips_dsp) && __mips == 32

#define MIPS_MULT __builtin_mips_mult
#define MIPS_MADD __builtin_mips_madd
#define MIPS_EXTR __builtin_mips_extr_w

#elif defined(__mips_isa_rev) && __mips_isa_rev < 6

static inline long long MIPS_MULT(int a, int b) {
    long long acc;

    asm volatile (
            "mult %[a], %[b]  \n"
        : [acc] "=x"(acc)
        : [a] "r"(a), [b] "r"(b)
        :
    );
    return acc;
}

static inline long long MIPS_MADD(long long acc, int a, int b) {
    asm volatile (
            "madd %[a], %[b]  \n"
        : [acc] "+x"(acc)
        : [a] "r"(a), [b] "r"(b)
        :
    );
    return acc;
}

static inline opus_val32 MIPS_EXTR(long long acc, int shift) {
    return (opus_val32)(acc >> shift);
}

#endif

#if defined (OVERRIDE_comb_filter)
void comb_filter(opus_val32 *y, opus_val32 *x, int T0, int T1, int N,
      opus_val16 g0, opus_val16 g1, int tapset0, int tapset1,
      const opus_val16 *window, int overlap, int arch)
{
   int i;
   opus_val32 x0, x1, x2, x3, x4;

   (void)arch;

   /* printf ("%d %d %f %f\n", T0, T1, g0, g1); */
   opus_val16 g00, g01, g02, g10, g11, g12;
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
      opus_val32 res;
      long long acc;
      f = MULT16_16_Q15(window[i],window[i]);
      x0= x[i-T1+2];

      acc = MIPS_MULT((int)MULT16_16_Q15((Q15ONE-f),g00), (int)x[i-T0]);
      acc = MIPS_MADD(acc, (int)MULT16_16_Q15((Q15ONE-f),g01), (int)ADD32(x[i-T0-1],x[i-T0+1]));
      acc = MIPS_MADD(acc, (int)MULT16_16_Q15((Q15ONE-f),g02), (int)ADD32(x[i-T0-2],x[i-T0+2]));
      acc = MIPS_MADD(acc, (int)MULT16_16_Q15(f,g10), (int)x2);
      acc = MIPS_MADD(acc, (int)MULT16_16_Q15(f,g11), (int)ADD32(x3,x1));
      acc = MIPS_MADD(acc, (int)MULT16_16_Q15(f,g12), (int)ADD32(x4,x0));
      res = MIPS_EXTR(acc, 15);

      y[i] = x[i] + res;

      x4=x3;
      x3=x2;
      x2=x1;
      x1=x0;
   }

   x4 = x[i-T1-2];
   x3 = x[i-T1-1];
   x2 = x[i-T1];
   x1 = x[i-T1+1];

   if (g1==0)
   {
      /* OPT: Happens to work without the OPUS_MOVE(), but only because the current encoder already copies x to y */
      if (x!=y)
         OPUS_MOVE(y+overlap, x+overlap, N-overlap);
      return;
   }

   for (i=overlap;i<N;i++)
   {
      opus_val32 res;
      long long acc;
      x0=x[i-T1+2];

      acc = MIPS_MULT((int)g10, (int)x2);
      acc = MIPS_MADD(acc, (int)g11, (int)ADD32(x3,x1));
      acc = MIPS_MADD(acc, (int)g12, (int)ADD32(x4,x0));
      res = MIPS_EXTR(acc, 15);

      y[i] = x[i] + res;
      x4=x3;
      x3=x2;
      x2=x1;
      x1=x0;
   }
}
#endif /* OVERRIDE_comb_filter */

#endif /* CELT_MIPSR1_H__ */
