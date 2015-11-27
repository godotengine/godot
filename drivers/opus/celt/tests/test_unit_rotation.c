/* Copyright (c) 2008-2011 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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

#ifdef OPUS_HAVE_CONFIG_H
#include "opus/opus_config.h"
#endif

#ifndef CUSTOM_MODES
#define CUSTOM_MODES
#endif

#define CELT_C

#include <stdio.h>
#include <stdlib.h>
#include "opus/celt/vq.c"
#include "opus/celt/cwrs.c"
#include "opus/celt/entcode.c"
#include "opus/celt/entenc.c"
#include "opus/celt/entdec.c"
#include "opus/celt/mathops.c"
#include "opus/celt/bands.h"
#include <math.h>
#define MAX_SIZE 100

int ret=0;
void test_rotation(int N, int K)
{
   int i;
   double err = 0, ener = 0, snr, snr0;
   opus_val16 x0[MAX_SIZE];
   opus_val16 x1[MAX_SIZE];
   for (i=0;i<N;i++)
      x1[i] = x0[i] = rand()%32767-16384;
   exp_rotation(x1, N, 1, 1, K, SPREAD_NORMAL);
   for (i=0;i<N;i++)
   {
      err += (x0[i]-(double)x1[i])*(x0[i]-(double)x1[i]);
      ener += x0[i]*(double)x0[i];
   }
   snr0 = 20*log10(ener/err);
   err = ener = 0;
   exp_rotation(x1, N, -1, 1, K, SPREAD_NORMAL);
   for (i=0;i<N;i++)
   {
      err += (x0[i]-(double)x1[i])*(x0[i]-(double)x1[i]);
      ener += x0[i]*(double)x0[i];
   }
   snr = 20*log10(ener/err);
   printf ("SNR for size %d (%d pulses) is %f (was %f without inverse)\n", N, K, snr, snr0);
   if (snr < 60 || snr0 > 20)
   {
      fprintf(stderr, "FAIL!\n");
      ret = 1;
   }
}

int main(void)
{
   ALLOC_STACK;
   test_rotation(15, 3);
   test_rotation(23, 5);
   test_rotation(50, 3);
   test_rotation(80, 1);
   return ret;
}
