/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Written by Jean-Marc Valin */
/**
   @file pitch.h
   @brief Pitch analysis
 */

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

#ifndef PITCH_MIPSR1_H
#define PITCH_MIPSR1_H

#include "fixed_generic_mipsr1.h"

#if defined (__mips_dsp) && __mips == 32

#define accumulator_t opus_int64
#define MIPS_MAC(acc,a,b) \
    __builtin_mips_madd((acc), (int)(a), (int)(b))

#define MIPS_MAC16x16_2X(acc,a2x,b2x) \
    __builtin_mips_dpaq_s_w_ph((acc), (a2x), (b2x))

#define OVERRIDE_CELT_INNER_PROD
#define OVERRIDE_DUAL_INNER_PROD
#define OVERRIDE_XCORR_KERNEL

#else /* any other MIPS */

/* using madd is slower due to single accumulator */
#define accumulator_t opus_int32
#define MIPS_MAC MAC16_16

#define OVERRIDE_CELT_INNER_PROD
#define OVERRIDE_DUAL_INNER_PROD
#define OVERRIDE_XCORR_KERNEL

#endif /* any other MIPS */


#if defined(OVERRIDE_CELT_INNER_PROD)

static OPUS_INLINE opus_val32 celt_inner_prod(const opus_val16 *x,
      const opus_val16 *y, int N, int arch)
{
   int j;
   accumulator_t acc = 0;

#if defined (MIPS_MAC16x16_2X)
   const v2i16 *x2x;
   const v2i16 *y2x;
   int loops;

   /* misaligned */
   if (((long)x | (long)y) & 3)
       goto fallback;

   x2x = __builtin_assume_aligned(x, 4);
   y2x = __builtin_assume_aligned(y, 4);
   loops = N / 8;
   for (j = 0; j < loops; j++)
   {
      acc = MIPS_MAC16x16_2X(acc, x2x[0], y2x[0]);
      acc = MIPS_MAC16x16_2X(acc, x2x[1], y2x[1]);
      acc = MIPS_MAC16x16_2X(acc, x2x[2], y2x[2]);
      acc = MIPS_MAC16x16_2X(acc, x2x[3], y2x[3]);
      x2x += 4; y2x += 4;
   }

   switch (N & 7) {
   case 7:
      acc = MIPS_MAC16x16_2X(acc, x2x[0], y2x[0]);
      acc = MIPS_MAC16x16_2X(acc, x2x[1], y2x[1]);
      acc = MIPS_MAC16x16_2X(acc, x2x[2], y2x[2]);
      acc = MIPS_MAC(acc, x[N-1], y[N-1]);
      break;
   case 6:
      acc = MIPS_MAC16x16_2X(acc, x2x[0], y2x[0]);
      acc = MIPS_MAC16x16_2X(acc, x2x[1], y2x[1]);
      acc = MIPS_MAC16x16_2X(acc, x2x[2], y2x[2]);
      break;
   case 5:
      acc = MIPS_MAC16x16_2X(acc, x2x[0], y2x[0]);
      acc = MIPS_MAC16x16_2X(acc, x2x[1], y2x[1]);
      acc = MIPS_MAC(acc, x[N-1], y[N-1]);
      break;
   case 4:
      acc = MIPS_MAC16x16_2X(acc, x2x[0], y2x[0]);
      acc = MIPS_MAC16x16_2X(acc, x2x[1], y2x[1]);
      break;
   case 3:
      acc = MIPS_MAC16x16_2X(acc, x2x[0], y2x[0]);
      acc = MIPS_MAC(acc, x[N-1], y[N-1]);
      break;
   case 2:
      acc = MIPS_MAC16x16_2X(acc, x2x[0], y2x[0]);
      break;
   case 1:
      acc = MIPS_MAC(acc, x[N-1], y[N-1]);
      break;
   case 0:
      break;
   }
   return __builtin_mips_extr_w(acc, 1);

fallback:
#endif
   for (j = 0; j < N - 3; j += 4)
   {
      acc = MIPS_MAC(acc, x[j],   y[j]);
      acc = MIPS_MAC(acc, x[j+1], y[j+1]);
      acc = MIPS_MAC(acc, x[j+2], y[j+2]);
      acc = MIPS_MAC(acc, x[j+3], y[j+3]);
   }

   switch (N & 3) {
   case 3:
      acc = MIPS_MAC(acc, x[j],   y[j]);
      acc = MIPS_MAC(acc, x[j+1], y[j+1]);
      acc = MIPS_MAC(acc, x[j+2], y[j+2]);
      break;
   case 2:
      acc = MIPS_MAC(acc, x[j],   y[j]);
      acc = MIPS_MAC(acc, x[j+1], y[j+1]);
      break;
   case 1:
      acc = MIPS_MAC(acc, x[j],   y[j]);
      break;
   case 0:
      break;
   }

   (void)arch;

   return (opus_val32)acc;
}
#endif /* OVERRIDE_CELT_INNER_PROD */

#if defined(OVERRIDE_DUAL_INNER_PROD)
static inline void dual_inner_prod(const opus_val16 *x, const opus_val16 *y01, const opus_val16 *y02,
      int N, opus_val32 *xy1, opus_val32 *xy2, int arch)
{
   int j;
   accumulator_t acc1 = 0;
   accumulator_t acc2 = 0;

#if defined (MIPS_MAC16x16_2X)
   const v2i16 *x2x;
   const v2i16 *y01_2x;
   const v2i16 *y02_2x;

   /* misaligned */
   if (((long)x | (long)y01 | (long)y02) & 3)
       goto fallback;

   x2x = __builtin_assume_aligned(x, 4);
   y01_2x = __builtin_assume_aligned(y01, 4);
   y02_2x = __builtin_assume_aligned(y02, 4);
   N /= 2;

   for (j = 0; j < N - 3; j += 4)
   {
      acc1 = MIPS_MAC16x16_2X(acc1, x2x[j],   y01_2x[j]);
      acc2 = MIPS_MAC16x16_2X(acc2, x2x[j],   y02_2x[j]);
      acc1 = MIPS_MAC16x16_2X(acc1, x2x[j+1], y01_2x[j+1]);
      acc2 = MIPS_MAC16x16_2X(acc2, x2x[j+1], y02_2x[j+1]);
      acc1 = MIPS_MAC16x16_2X(acc1, x2x[j+2], y01_2x[j+2]);
      acc2 = MIPS_MAC16x16_2X(acc2, x2x[j+2], y02_2x[j+2]);
      acc1 = MIPS_MAC16x16_2X(acc1, x2x[j+3], y01_2x[j+3]);
      acc2 = MIPS_MAC16x16_2X(acc2, x2x[j+3], y02_2x[j+3]);
   }

   switch (N & 3) {
   case 3:
      acc1 = MIPS_MAC16x16_2X(acc1, x2x[j],   y01_2x[j]);
      acc2 = MIPS_MAC16x16_2X(acc2, x2x[j],   y02_2x[j]);
      acc1 = MIPS_MAC16x16_2X(acc1, x2x[j+1], y01_2x[j+1]);
      acc2 = MIPS_MAC16x16_2X(acc2, x2x[j+1], y02_2x[j+1]);
      acc1 = MIPS_MAC16x16_2X(acc1, x2x[j+2], y01_2x[j+2]);
      acc2 = MIPS_MAC16x16_2X(acc2, x2x[j+2], y02_2x[j+2]);
      break;
   case 2:
      acc1 = MIPS_MAC16x16_2X(acc1, x2x[j],   y01_2x[j]);
      acc2 = MIPS_MAC16x16_2X(acc2, x2x[j],   y02_2x[j]);
      acc1 = MIPS_MAC16x16_2X(acc1, x2x[j+1], y01_2x[j+1]);
      acc2 = MIPS_MAC16x16_2X(acc2, x2x[j+1], y02_2x[j+1]);
      break;
   case 1:
      acc1 = MIPS_MAC16x16_2X(acc1, x2x[j],   y01_2x[j]);
      acc2 = MIPS_MAC16x16_2X(acc2, x2x[j],   y02_2x[j]);
      break;
   case 0:
      break;
   }

   *xy1 = __builtin_mips_extr_w(acc1, 1);
   *xy2 = __builtin_mips_extr_w(acc2, 1);
   return;

fallback:
#endif
   /* Compute the norm of X+Y and X-Y as |X|^2 + |Y|^2 +/- sum(xy) */
   for (j = 0; j < N - 3; j += 4)
   {
      acc1 = MIPS_MAC(acc1, x[j],   y01[j]);
      acc2 = MIPS_MAC(acc2, x[j],   y02[j]);
      acc1 = MIPS_MAC(acc1, x[j+1], y01[j+1]);
      acc2 = MIPS_MAC(acc2, x[j+1], y02[j+1]);
      acc1 = MIPS_MAC(acc1, x[j+2], y01[j+2]);
      acc2 = MIPS_MAC(acc2, x[j+2], y02[j+2]);
      acc1 = MIPS_MAC(acc1, x[j+3], y01[j+3]);
      acc2 = MIPS_MAC(acc2, x[j+3], y02[j+3]);
   }

   if (j < N) {
      acc1 = MIPS_MAC(acc1, x[j],   y01[j]);
      acc2 = MIPS_MAC(acc2, x[j],   y02[j]);
      acc1 = MIPS_MAC(acc1, x[j+1], y01[j+1]);
      acc2 = MIPS_MAC(acc2, x[j+1], y02[j+1]);
   }

   (void)arch;

   *xy1 = (opus_val32)acc1;
   *xy2 = (opus_val32)acc2;
}
#endif /* OVERRIDE_DUAL_INNER_PROD */

#if defined(OVERRIDE_XCORR_KERNEL)

static inline void xcorr_kernel_mips(const opus_val16 * x,
      const opus_val16 * y, opus_val32 sum[4], int len)
{
   int j;
   opus_val16 y_0, y_1, y_2, y_3;

    accumulator_t sum_0, sum_1, sum_2, sum_3;
    sum_0 =  (accumulator_t)sum[0];
    sum_1 =  (accumulator_t)sum[1];
    sum_2 =  (accumulator_t)sum[2];
    sum_3 =  (accumulator_t)sum[3];

    y_0=*y++;
    y_1=*y++;
    y_2=*y++;
    for (j=0;j<len-3;j+=4)
    {
        opus_val16 tmp;
        tmp = *x++;
        y_3=*y++;

        sum_0 = MIPS_MAC(sum_0, tmp, y_0);
        sum_1 = MIPS_MAC(sum_1, tmp, y_1);
        sum_2 = MIPS_MAC(sum_2, tmp, y_2);
        sum_3 = MIPS_MAC(sum_3, tmp, y_3);

        tmp=*x++;
        y_0=*y++;

        sum_0 = MIPS_MAC(sum_0, tmp, y_1);
        sum_1 = MIPS_MAC(sum_1, tmp, y_2);
        sum_2 = MIPS_MAC(sum_2, tmp, y_3);
        sum_3 = MIPS_MAC(sum_3, tmp, y_0);

       tmp=*x++;
       y_1=*y++;

       sum_0 = MIPS_MAC(sum_0, tmp, y_2);
       sum_1 = MIPS_MAC(sum_1, tmp, y_3);
       sum_2 = MIPS_MAC(sum_2, tmp, y_0);
       sum_3 = MIPS_MAC(sum_3, tmp, y_1);


      tmp=*x++;
      y_2=*y++;

      sum_0 = MIPS_MAC(sum_0, tmp, y_3);
      sum_1 = MIPS_MAC(sum_1, tmp, y_0);
      sum_2 = MIPS_MAC(sum_2, tmp, y_1);
      sum_3 = MIPS_MAC(sum_3, tmp, y_2);
   }

   switch (len & 3) {
   case 3:
      sum_0 = MIPS_MAC(sum_0, x[2], y_2);
      sum_1 = MIPS_MAC(sum_1, x[2], y[0]);
      sum_2 = MIPS_MAC(sum_2, x[2], y[1]);
      sum_3 = MIPS_MAC(sum_3, x[2], y[2]);

      sum_0 = MIPS_MAC(sum_0, x[1], y_1);
      sum_1 = MIPS_MAC(sum_1, x[1], y_2);
      sum_2 = MIPS_MAC(sum_2, x[1], y[0]);
      sum_3 = MIPS_MAC(sum_3, x[1], y[1]);

      sum_0 = MIPS_MAC(sum_0, x[0], y_0);
      sum_1 = MIPS_MAC(sum_1, x[0], y_1);
      sum_2 = MIPS_MAC(sum_2, x[0], y_2);
      sum_3 = MIPS_MAC(sum_3, x[0], y[0]);
      break;
   case 2:
      sum_0 = MIPS_MAC(sum_0, x[1], y_1);
      sum_1 = MIPS_MAC(sum_1, x[1], y_2);
      sum_2 = MIPS_MAC(sum_2, x[1], y[0]);
      sum_3 = MIPS_MAC(sum_3, x[1], y[1]);

      sum_0 = MIPS_MAC(sum_0, x[0], y_0);
      sum_1 = MIPS_MAC(sum_1, x[0], y_1);
      sum_2 = MIPS_MAC(sum_2, x[0], y_2);
      sum_3 = MIPS_MAC(sum_3, x[0], y[0]);
      break;
   case 1:
      sum_0 = MIPS_MAC(sum_0, x[0], y_0);
      sum_1 = MIPS_MAC(sum_1, x[0], y_1);
      sum_2 = MIPS_MAC(sum_2, x[0], y_2);
      sum_3 = MIPS_MAC(sum_3, x[0], y[0]);
      break;
   case 0:
      break;
   }

   sum[0] = (opus_val32)sum_0;
   sum[1] = (opus_val32)sum_1;
   sum[2] = (opus_val32)sum_2;
   sum[3] = (opus_val32)sum_3;
}

#define xcorr_kernel(x, y, sum, len, arch) \
    ((void)(arch), xcorr_kernel_mips(x, y, sum, len))

#undef accumulator_t
#undef MIPS_MAC

#endif /* OVERRIDE_XCORR_KERNEL */

#endif /* PITCH_MIPSR1_H */
