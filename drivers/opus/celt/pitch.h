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

#include "opus/celt/opus_modes.h"
#include "opus/celt/cpu_support.h"

#if defined(__SSE__) && !defined(OPUS_FIXED_POINT)
#include "x86/pitch_sse.h"
#endif

#if defined(OPUS_ARM_ASM) && defined(OPUS_FIXED_POINT)
# include "arm/pitch_arm.h"
#endif

void pitch_downsample(celt_sig * OPUS_RESTRICT x[], opus_val16 * OPUS_RESTRICT x_lp,
      int len, int C, int arch);

void pitch_search(const opus_val16 * OPUS_RESTRICT x_lp, opus_val16 * OPUS_RESTRICT y,
                  int len, int max_pitch, int *pitch, int arch);

opus_val16 remove_doubling(opus_val16 *x, int maxperiod, int minperiod,
      int N, int *T0, int prev_period, opus_val16 prev_gain);

/* OPT: This is the kernel you really want to optimize. It gets used a lot
   by the prefilter and by the PLC. */
#ifndef OVERRIDE_XCORR_KERNEL
static OPUS_INLINE void xcorr_kernel(const opus_val16 * x, const opus_val16 * y, opus_val32 sum[4], int len)
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
#endif /* OVERRIDE_XCORR_KERNEL */

#ifndef OVERRIDE_DUAL_INNER_PROD
static OPUS_INLINE void dual_inner_prod(const opus_val16 *x, const opus_val16 *y01, const opus_val16 *y02,
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
#endif

#ifdef OPUS_FIXED_POINT
opus_val32
#else
void
#endif
celt_pitch_xcorr_c(const opus_val16 *_x, const opus_val16 *_y,
      opus_val32 *xcorr, int len, int max_pitch);

#if !defined(OVERRIDE_PITCH_XCORR)
/*Is run-time CPU detection enabled on this platform?*/
# if defined(OPUS_HAVE_RTCD)
extern
#  if defined(OPUS_FIXED_POINT)
opus_val32
#  else
void
#  endif
(*const CELT_PITCH_XCORR_IMPL[OPUS_ARCHMASK+1])(const opus_val16 *,
      const opus_val16 *, opus_val32 *, int, int);

#  define celt_pitch_xcorr(_x, _y, xcorr, len, max_pitch, arch) \
  ((*CELT_PITCH_XCORR_IMPL[(arch)&OPUS_ARCHMASK])(_x, _y, \
        xcorr, len, max_pitch))
# else
#  define celt_pitch_xcorr(_x, _y, xcorr, len, max_pitch, arch) \
  ((void)(arch),celt_pitch_xcorr_c(_x, _y, xcorr, len, max_pitch))
# endif
#endif

#endif
