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

#ifndef PITCH_H
#define PITCH_H

#include "modes.h"
#include "cpu_support.h"

#if (defined(OPUS_X86_MAY_HAVE_SSE) && !defined(FIXED_POINT)) \
  || ((defined(OPUS_X86_MAY_HAVE_SSE4_1) || defined(OPUS_X86_MAY_HAVE_SSE2)) && defined(FIXED_POINT))
#include "x86/pitch_sse.h"
#endif

#if defined(MIPSr1_ASM)
#include "mips/pitch_mipsr1.h"
#endif

#if ((defined(OPUS_ARM_ASM) && defined(FIXED_POINT)) \
  || defined(OPUS_ARM_MAY_HAVE_NEON_INTR))
# include "arm/pitch_arm.h"
#endif

void pitch_downsample(celt_sig * OPUS_RESTRICT x[], opus_val16 * OPUS_RESTRICT x_lp,
      int len, int C, int arch);

void pitch_search(const opus_val16 * OPUS_RESTRICT x_lp, opus_val16 * OPUS_RESTRICT y,
                  int len, int max_pitch, int *pitch, int arch);

opus_val16 remove_doubling(opus_val16 *x, int maxperiod, int minperiod,
      int N, int *T0, int prev_period, opus_val16 prev_gain, int arch);


/* OPT: This is the kernel you really want to optimize. It gets used a lot
   by the prefilter and by the PLC. */
static OPUS_INLINE void xcorr_kernel_c(const opus_val16 * x, const opus_val16 * y, opus_val32 sum[4], int len)
{
   int j;
   opus_val16 y_0, y_1, y_2, y_3;
   celt_assert(len>=3);
   y_3=0; /* gcc doesn't realize that y_3 can't be used uninitialized */
   y_0=*y++;
   y_1=*y++;
   y_2=*y++;
   for (j=0;j<len-3;j+=4)
   {
      opus_val16 tmp;
      tmp = *x++;
      y_3=*y++;
      sum[0] = MAC16_16(sum[0],tmp,y_0);
      sum[1] = MAC16_16(sum[1],tmp,y_1);
      sum[2] = MAC16_16(sum[2],tmp,y_2);
      sum[3] = MAC16_16(sum[3],tmp,y_3);
      tmp=*x++;
      y_0=*y++;
      sum[0] = MAC16_16(sum[0],tmp,y_1);
      sum[1] = MAC16_16(sum[1],tmp,y_2);
      sum[2] = MAC16_16(sum[2],tmp,y_3);
      sum[3] = MAC16_16(sum[3],tmp,y_0);
      tmp=*x++;
      y_1=*y++;
      sum[0] = MAC16_16(sum[0],tmp,y_2);
      sum[1] = MAC16_16(sum[1],tmp,y_3);
      sum[2] = MAC16_16(sum[2],tmp,y_0);
      sum[3] = MAC16_16(sum[3],tmp,y_1);
      tmp=*x++;
      y_2=*y++;
      sum[0] = MAC16_16(sum[0],tmp,y_3);
      sum[1] = MAC16_16(sum[1],tmp,y_0);
      sum[2] = MAC16_16(sum[2],tmp,y_1);
      sum[3] = MAC16_16(sum[3],tmp,y_2);
   }
   if (j++<len)
   {
      opus_val16 tmp = *x++;
      y_3=*y++;
      sum[0] = MAC16_16(sum[0],tmp,y_0);
      sum[1] = MAC16_16(sum[1],tmp,y_1);
      sum[2] = MAC16_16(sum[2],tmp,y_2);
      sum[3] = MAC16_16(sum[3],tmp,y_3);
   }
   if (j++<len)
   {
      opus_val16 tmp=*x++;
      y_0=*y++;
      sum[0] = MAC16_16(sum[0],tmp,y_1);
      sum[1] = MAC16_16(sum[1],tmp,y_2);
      sum[2] = MAC16_16(sum[2],tmp,y_3);
      sum[3] = MAC16_16(sum[3],tmp,y_0);
   }
   if (j<len)
   {
      opus_val16 tmp=*x++;
      y_1=*y++;
      sum[0] = MAC16_16(sum[0],tmp,y_2);
      sum[1] = MAC16_16(sum[1],tmp,y_3);
      sum[2] = MAC16_16(sum[2],tmp,y_0);
      sum[3] = MAC16_16(sum[3],tmp,y_1);
   }
}

#ifndef OVERRIDE_XCORR_KERNEL
#define xcorr_kernel(x, y, sum, len, arch) \
    ((void)(arch),xcorr_kernel_c(x, y, sum, len))
#endif /* OVERRIDE_XCORR_KERNEL */


static OPUS_INLINE void dual_inner_prod_c(const opus_val16 *x, const opus_val16 *y01, const opus_val16 *y02,
      int N, opus_val32 *xy1, opus_val32 *xy2)
{
   int i;
   opus_val32 xy01=0;
   opus_val32 xy02=0;
   for (i=0;i<N;i++)
   {
      xy01 = MAC16_16(xy01, x[i], y01[i]);
      xy02 = MAC16_16(xy02, x[i], y02[i]);
   }
   *xy1 = xy01;
   *xy2 = xy02;
}

#ifndef OVERRIDE_DUAL_INNER_PROD
# define dual_inner_prod(x, y01, y02, N, xy1, xy2, arch) \
    ((void)(arch),dual_inner_prod_c(x, y01, y02, N, xy1, xy2))
#endif

/*We make sure a C version is always available for cases where the overhead of
  vectorization and passing around an arch flag aren't worth it.*/
static OPUS_INLINE opus_val32 celt_inner_prod_c(const opus_val16 *x,
      const opus_val16 *y, int N)
{
   int i;
   opus_val32 xy=0;
   for (i=0;i<N;i++)
      xy = MAC16_16(xy, x[i], y[i]);
   return xy;
}

#if !defined(OVERRIDE_CELT_INNER_PROD)
# define celt_inner_prod(x, y, N, arch) \
    ((void)(arch),celt_inner_prod_c(x, y, N))
#endif

#ifdef NON_STATIC_COMB_FILTER_CONST_C
void comb_filter_const_c(opus_val32 *y, opus_val32 *x, int T, int N,
     opus_val16 g10, opus_val16 g11, opus_val16 g12);
#endif


#ifdef FIXED_POINT
opus_val32
#else
void
#endif
celt_pitch_xcorr_c(const opus_val16 *_x, const opus_val16 *_y,
      opus_val32 *xcorr, int len, int max_pitch);

#if !defined(OVERRIDE_PITCH_XCORR)
/*Is run-time CPU detection enabled on this platform?*/
# if defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_ASM) \
   || (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) \
   && !defined(OPUS_ARM_PRESUME_NEON_INTR)))
extern
#  if defined(FIXED_POINT)
opus_val32
#  else
void
#  endif
(*const CELT_PITCH_XCORR_IMPL[OPUS_ARCHMASK+1])(const opus_val16 *,
      const opus_val16 *, opus_val32 *, int, int);

#  define OVERRIDE_PITCH_XCORR
#  define celt_pitch_xcorr(_x, _y, xcorr, len, max_pitch, arch) \
  ((*CELT_PITCH_XCORR_IMPL[(arch)&OPUS_ARCHMASK])(_x, _y, \
        xcorr, len, max_pitch))
# else

#ifdef FIXED_POINT
opus_val32
#else
void
#endif
celt_pitch_xcorr(const opus_val16 *_x, const opus_val16 *_y,
      opus_val32 *xcorr, int len, int max_pitch, int arch);

# endif
#endif

#endif
