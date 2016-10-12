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

#define OVERRIDE_DUAL_INNER_PROD
static inline void dual_inner_prod(const opus_val16 *x, const opus_val16 *y01, const opus_val16 *y02,
      int N, opus_val32 *xy1, opus_val32 *xy2, int arch)
{
   int j;
   opus_val32 xy01=0;
   opus_val32 xy02=0;

   (void)arch;

   asm volatile("MULT $ac1, $0, $0");
   asm volatile("MULT $ac2, $0, $0");
   /* Compute the norm of X+Y and X-Y as |X|^2 + |Y|^2 +/- sum(xy) */
   for (j=0;j<N;j++)
   {
      asm volatile("MADD $ac1, %0, %1" : : "r" ((int)x[j]), "r" ((int)y01[j]));
      asm volatile("MADD $ac2, %0, %1" : : "r" ((int)x[j]), "r" ((int)y02[j]));
      ++j;
      asm volatile("MADD $ac1, %0, %1" : : "r" ((int)x[j]), "r" ((int)y01[j]));
      asm volatile("MADD $ac2, %0, %1" : : "r" ((int)x[j]), "r" ((int)y02[j]));
   }
   asm volatile ("mflo %0, $ac1": "=r"(xy01));
   asm volatile ("mflo %0, $ac2": "=r"(xy02));
   *xy1 = xy01;
   *xy2 = xy02;
}

static inline void xcorr_kernel_mips(const opus_val16 * x,
      const opus_val16 * y, opus_val32 sum[4], int len)
{
   int j;
   opus_val16 y_0, y_1, y_2, y_3;

    opus_int64 sum_0, sum_1, sum_2, sum_3;
    sum_0 =  (opus_int64)sum[0];
    sum_1 =  (opus_int64)sum[1];
    sum_2 =  (opus_int64)sum[2];
    sum_3 =  (opus_int64)sum[3];

    y_3=0; /* gcc doesn't realize that y_3 can't be used uninitialized */
    y_0=*y++;
    y_1=*y++;
    y_2=*y++;
    for (j=0;j<len-3;j+=4)
    {
        opus_val16 tmp;
        tmp = *x++;
        y_3=*y++;

        sum_0 = __builtin_mips_madd( sum_0, tmp, y_0);
        sum_1 = __builtin_mips_madd( sum_1, tmp, y_1);
        sum_2 = __builtin_mips_madd( sum_2, tmp, y_2);
        sum_3 = __builtin_mips_madd( sum_3, tmp, y_3);

        tmp=*x++;
        y_0=*y++;

        sum_0 = __builtin_mips_madd( sum_0, tmp, y_1 );
        sum_1 = __builtin_mips_madd( sum_1, tmp, y_2 );
        sum_2 = __builtin_mips_madd( sum_2, tmp, y_3);
        sum_3 = __builtin_mips_madd( sum_3, tmp, y_0);

       tmp=*x++;
       y_1=*y++;

       sum_0 = __builtin_mips_madd( sum_0, tmp, y_2 );
       sum_1 = __builtin_mips_madd( sum_1, tmp, y_3 );
       sum_2 = __builtin_mips_madd( sum_2, tmp, y_0);
       sum_3 = __builtin_mips_madd( sum_3, tmp, y_1);


      tmp=*x++;
      y_2=*y++;

       sum_0 = __builtin_mips_madd( sum_0, tmp, y_3 );
       sum_1 = __builtin_mips_madd( sum_1, tmp, y_0 );
       sum_2 = __builtin_mips_madd( sum_2, tmp, y_1);
       sum_3 = __builtin_mips_madd( sum_3, tmp, y_2);

   }
   if (j++<len)
   {
      opus_val16 tmp = *x++;
      y_3=*y++;

       sum_0 = __builtin_mips_madd( sum_0, tmp, y_0 );
       sum_1 = __builtin_mips_madd( sum_1, tmp, y_1 );
       sum_2 = __builtin_mips_madd( sum_2, tmp, y_2);
       sum_3 = __builtin_mips_madd( sum_3, tmp, y_3);
   }

   if (j++<len)
   {
      opus_val16 tmp=*x++;
      y_0=*y++;

      sum_0 = __builtin_mips_madd( sum_0, tmp, y_1 );
      sum_1 = __builtin_mips_madd( sum_1, tmp, y_2 );
      sum_2 = __builtin_mips_madd( sum_2, tmp, y_3);
      sum_3 = __builtin_mips_madd( sum_3, tmp, y_0);
   }

   if (j<len)
   {
      opus_val16 tmp=*x++;
      y_1=*y++;

       sum_0 = __builtin_mips_madd( sum_0, tmp, y_2 );
       sum_1 = __builtin_mips_madd( sum_1, tmp, y_3 );
       sum_2 = __builtin_mips_madd( sum_2, tmp, y_0);
       sum_3 = __builtin_mips_madd( sum_3, tmp, y_1);

   }

   sum[0] = (opus_val32)sum_0;
   sum[1] = (opus_val32)sum_1;
   sum[2] = (opus_val32)sum_2;
   sum[3] = (opus_val32)sum_3;
}

#define OVERRIDE_XCORR_KERNEL
#define xcorr_kernel(x, y, sum, len, arch) \
    ((void)(arch), xcorr_kernel_mips(x, y, sum, len))

#endif /* PITCH_MIPSR1_H */
