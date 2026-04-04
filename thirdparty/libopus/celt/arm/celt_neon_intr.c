/* Copyright (c) 2014-2015 Xiph.Org Foundation
   Copyright (c) 2024 Arm Limited
   Written by Viswanath Puttagunta */
/**
   @file celt_neon_intr.c
   @brief ARM Neon Intrinsic optimizations for celt
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <arm_neon.h>
#include "../float_cast.h"
#include "../mathops.h"
#include "../pitch.h"
#include <stddef.h>
#if defined(OPUS_CHECK_ASM)
#include <stdlib.h>
#endif

#if !defined(DISABLE_FLOAT_API) && defined(OPUS_ARM_MAY_HAVE_NEON_INTR)

void celt_float2int16_neon(const float * OPUS_RESTRICT in, short * OPUS_RESTRICT out, int cnt)
{
   int i = 0;

#if defined(__ARM_NEON)
   const int BLOCK_SIZE = 16;
   const int blockedSize = cnt / BLOCK_SIZE * BLOCK_SIZE;

   for (; i < blockedSize; i += BLOCK_SIZE)
   {
      float32x4_t orig_a = vld1q_f32(&in[i +  0]);
      float32x4_t orig_b = vld1q_f32(&in[i +  4]);
      float32x4_t orig_c = vld1q_f32(&in[i +  8]);
      float32x4_t orig_d = vld1q_f32(&in[i + 12]);

      int16x4_t asShort_a = vqmovn_s32(vroundf(vmulq_n_f32(orig_a, CELT_SIG_SCALE)));
      int16x4_t asShort_b = vqmovn_s32(vroundf(vmulq_n_f32(orig_b, CELT_SIG_SCALE)));
      int16x4_t asShort_c = vqmovn_s32(vroundf(vmulq_n_f32(orig_c, CELT_SIG_SCALE)));
      int16x4_t asShort_d = vqmovn_s32(vroundf(vmulq_n_f32(orig_d, CELT_SIG_SCALE)));

      vst1_s16(&out[i +  0], asShort_a);
      vst1_s16(&out[i +  4], asShort_b);
      vst1_s16(&out[i +  8], asShort_c);
      vst1_s16(&out[i + 12], asShort_d);
# if defined(OPUS_CHECK_ASM)
      short out_c[BLOCK_SIZE];
      int j;
      for(j = 0; j < BLOCK_SIZE; j++)
      {
         out_c[j] = FLOAT2INT16(in[i + j]);
         celt_assert(abs((out_c[j] - out[i + j])) <= 1);
      }
# endif
   }
#endif

   for (; i < cnt; i++)
   {
      out[i] = FLOAT2INT16(in[i]);
   }
}

int opus_limit2_checkwithin1_neon(float *samples, int cnt)
{
   const float hardclipMin = -2.0f;
   const float hardclipMax = 2.0f;

   int i = 0;
   int exceeding1 = 0;
   int nextIndex = 0;

#if defined(__ARM_NEON)
   const int BLOCK_SIZE = 16;
   const int blockedSize = cnt / BLOCK_SIZE * BLOCK_SIZE;

   float32x4_t min_all_0 = vdupq_n_f32(0.0f);
   float32x4_t min_all_1 = vdupq_n_f32(0.0f);
   float32x4_t max_all_0 = vdupq_n_f32(0.0f);
   float32x4_t max_all_1 = vdupq_n_f32(0.0f);

   float max, min;

   for (i = 0; i < blockedSize; i += BLOCK_SIZE)
   {
      const float32x4_t orig_a = vld1q_f32(&samples[i +  0]);
      const float32x4_t orig_b = vld1q_f32(&samples[i +  4]);
      const float32x4_t orig_c = vld1q_f32(&samples[i +  8]);
      const float32x4_t orig_d = vld1q_f32(&samples[i + 12]);
      max_all_0 = vmaxq_f32(max_all_0, vmaxq_f32(orig_a, orig_b));
      max_all_1 = vmaxq_f32(max_all_1, vmaxq_f32(orig_c, orig_d));
      min_all_0 = vminq_f32(min_all_0, vminq_f32(orig_a, orig_b));
      min_all_1 = vminq_f32(min_all_1, vminq_f32(orig_c, orig_d));
   }

   max = vmaxvf(vmaxq_f32(max_all_0, max_all_1));
   min = vminvf(vminq_f32(min_all_0, min_all_1));

   if (min < hardclipMin || max > hardclipMax)
   {
      const float32x4_t hardclipMinReg = vdupq_n_f32(hardclipMin);
      const float32x4_t hardclipMaxReg = vdupq_n_f32(hardclipMax);
      for (i = 0; i < blockedSize; i += BLOCK_SIZE)
      {
         const float32x4_t orig_a = vld1q_f32(&samples[i +  0]);
         const float32x4_t orig_b = vld1q_f32(&samples[i +  4]);
         const float32x4_t orig_c = vld1q_f32(&samples[i +  8]);
         const float32x4_t orig_d = vld1q_f32(&samples[i + 12]);
         const float32x4_t clipped_a = vminq_f32(hardclipMaxReg, vmaxq_f32(orig_a, hardclipMinReg));
         const float32x4_t clipped_b = vminq_f32(hardclipMaxReg, vmaxq_f32(orig_b, hardclipMinReg));
         const float32x4_t clipped_c = vminq_f32(hardclipMaxReg, vmaxq_f32(orig_c, hardclipMinReg));
         const float32x4_t clipped_d = vminq_f32(hardclipMaxReg, vmaxq_f32(orig_d, hardclipMinReg));
         vst1q_f32(&samples[i + 0], clipped_a);
         vst1q_f32(&samples[i + 4], clipped_b);
         vst1q_f32(&samples[i + 8], clipped_c);
         vst1q_f32(&samples[i + 12], clipped_d);
      }
   }

   nextIndex = blockedSize;
   exceeding1 |= max > 1.0f || min < -1.0f;

#endif

   for (i = nextIndex; i < cnt; i++)
   {
      const float origVal = samples[i];
      float clippedVal = origVal;
      clippedVal = MAX16(hardclipMin, clippedVal);
      clippedVal = MIN16(hardclipMax, clippedVal);
      samples[i] = clippedVal;

      exceeding1 |= origVal > 1.0f || origVal < -1.0f;
   }

   return !exceeding1;
}

#endif


#if defined(FIXED_POINT)
#include <string.h>

void xcorr_kernel_neon_fixed(const opus_val16 * x, const opus_val16 * y, opus_val32 sum[4], int len)
{
   int j;
   int32x4_t a = vld1q_s32(sum);
   /* Load y[0...3] */
   /* This requires len>0 to always be valid (which we assert in the C code). */
   int16x4_t y0 = vld1_s16(y);
   y += 4;

   /* This loop loads one y value more than we actually need.
      Therefore we have to stop as soon as there are 8 or fewer samples left
       (instead of 7), to avoid reading past the end of the array. */
   for (j = 0; j + 8 < len; j += 8)
   {
      /* Load x[0...7] */
      int16x8_t xx = vld1q_s16(x);
      int16x4_t x0 = vget_low_s16(xx);
      int16x4_t x4 = vget_high_s16(xx);
      /* Load y[4...11] */
      int16x8_t yy = vld1q_s16(y);
      int16x4_t y4 = vget_low_s16(yy);
      int16x4_t y8 = vget_high_s16(yy);
      int32x4_t a0 = vmlal_lane_s16(a, y0, x0, 0);
      int32x4_t a1 = vmlal_lane_s16(a0, y4, x4, 0);

      int16x4_t y1 = vext_s16(y0, y4, 1);
      int16x4_t y5 = vext_s16(y4, y8, 1);
      int32x4_t a2 = vmlal_lane_s16(a1, y1, x0, 1);
      int32x4_t a3 = vmlal_lane_s16(a2, y5, x4, 1);

      int16x4_t y2 = vext_s16(y0, y4, 2);
      int16x4_t y6 = vext_s16(y4, y8, 2);
      int32x4_t a4 = vmlal_lane_s16(a3, y2, x0, 2);
      int32x4_t a5 = vmlal_lane_s16(a4, y6, x4, 2);

      int16x4_t y3 = vext_s16(y0, y4, 3);
      int16x4_t y7 = vext_s16(y4, y8, 3);
      int32x4_t a6 = vmlal_lane_s16(a5, y3, x0, 3);
      int32x4_t a7 = vmlal_lane_s16(a6, y7, x4, 3);

      y0 = y8;
      a = a7;
      x += 8;
      y += 8;
   }
   if (j + 4 < len) {
      /* Load x[0...3] */
      int16x4_t x0 = vld1_s16(x);
      /* Load y[4...7] */
      int16x4_t y4 = vld1_s16(y);
      int32x4_t a0 = vmlal_lane_s16(a, y0, x0, 0);
      int16x4_t y1 = vext_s16(y0, y4, 1);
      int32x4_t a1 = vmlal_lane_s16(a0, y1, x0, 1);
      int16x4_t y2 = vext_s16(y0, y4, 2);
      int32x4_t a2 = vmlal_lane_s16(a1, y2, x0, 2);
      int16x4_t y3 = vext_s16(y0, y4, 3);
      int32x4_t a3 = vmlal_lane_s16(a2, y3, x0, 3);
      y0 = y4;
      a = a3;
      x += 4;
      y += 4;
      j += 4;
   }
   if (j + 2 < len) {
      /* Load x[0...1] */
      int16x4x2_t xx = vld2_dup_s16(x);
      int16x4_t x0 = xx.val[0];
      int16x4_t x1 = xx.val[1];
      /* Load y[4...5].
         We would like to use vld1_dup_s32(), but casting the pointer would
          break strict aliasing rules and potentially have alignment issues.
         Fortunately the compiler seems capable of translating this memcpy()
          and vdup_n_s32() into the equivalent vld1_dup_s32().*/
      int32_t yy;
      memcpy(&yy, y, sizeof(yy));
      int16x4_t y4 = vreinterpret_s16_s32(vdup_n_s32(yy));
      int32x4_t a0 = vmlal_s16(a, y0, x0);
      int16x4_t y1 = vext_s16(y0, y4, 1);
      /* Replace bottom copy of {y[5], y[4]} in y4 with {y[3], y[2]} from y0,
          using VSRI instead of VEXT, since it's a data-processing
          instruction. */
      y0 = vreinterpret_s16_s64(vsri_n_s64(vreinterpret_s64_s16(y4),
       vreinterpret_s64_s16(y0), 32));
      int32x4_t a1 = vmlal_s16(a0, y1, x1);
      a = a1;
      x += 2;
      y += 2;
      j += 2;
   }
   if (j + 1 < len) {
      /* Load next x. */
      int16x4_t x0 = vld1_dup_s16(x);
      int32x4_t a0 = vmlal_s16(a, y0, x0);
      /* Load last y. */
      int16x4_t y4 = vld1_dup_s16(y);
      y0 = vreinterpret_s16_s64(vsri_n_s64(vreinterpret_s64_s16(y4),
       vreinterpret_s64_s16(y0), 16));
      a = a0;
      x++;
   }
   /* Load last x. */
   int16x4_t x0 = vld1_dup_s16(x);
   int32x4_t a0 = vmlal_s16(a, y0, x0);
   vst1q_s32(sum, a0);
}

#else

#if defined(__ARM_FEATURE_FMA) && defined(__ARM_ARCH_ISA_A64)
/* If we can, force the compiler to use an FMA instruction rather than break
 *    vmlaq_f32() into fmul/fadd. */
#ifdef vmlaq_lane_f32
#undef vmlaq_lane_f32
#endif
#define vmlaq_lane_f32(a,b,c,lane) vfmaq_lane_f32(a,b,c,lane)
#endif


/*
 * Function: xcorr_kernel_neon_float
 * ---------------------------------
 * Computes 4 correlation values and stores them in sum[4]
 */
static void xcorr_kernel_neon_float(const float32_t *x, const float32_t *y,
      float32_t sum[4], int len) {
   float32x4_t YY[3];
   float32x4_t YEXT[3];
   float32x4_t XX[2];
   float32x2_t XX_2;
   float32x4_t SUMM;
   const float32_t *xi = x;
   const float32_t *yi = y;

   celt_assert(len>0);

   YY[0] = vld1q_f32(yi);
   SUMM = vdupq_n_f32(0);

   /* Consume 8 elements in x vector and 12 elements in y
    * vector. However, the 12'th element never really gets
    * touched in this loop. So, if len == 8, then we only
    * must access y[0] to y[10]. y[11] must not be accessed
    * hence make sure len > 8 and not len >= 8
    */
   while (len > 8) {
      yi += 4;
      YY[1] = vld1q_f32(yi);
      yi += 4;
      YY[2] = vld1q_f32(yi);

      XX[0] = vld1q_f32(xi);
      xi += 4;
      XX[1] = vld1q_f32(xi);
      xi += 4;

      SUMM = vmlaq_lane_f32(SUMM, YY[0], vget_low_f32(XX[0]), 0);
      YEXT[0] = vextq_f32(YY[0], YY[1], 1);
      SUMM = vmlaq_lane_f32(SUMM, YEXT[0], vget_low_f32(XX[0]), 1);
      YEXT[1] = vextq_f32(YY[0], YY[1], 2);
      SUMM = vmlaq_lane_f32(SUMM, YEXT[1], vget_high_f32(XX[0]), 0);
      YEXT[2] = vextq_f32(YY[0], YY[1], 3);
      SUMM = vmlaq_lane_f32(SUMM, YEXT[2], vget_high_f32(XX[0]), 1);

      SUMM = vmlaq_lane_f32(SUMM, YY[1], vget_low_f32(XX[1]), 0);
      YEXT[0] = vextq_f32(YY[1], YY[2], 1);
      SUMM = vmlaq_lane_f32(SUMM, YEXT[0], vget_low_f32(XX[1]), 1);
      YEXT[1] = vextq_f32(YY[1], YY[2], 2);
      SUMM = vmlaq_lane_f32(SUMM, YEXT[1], vget_high_f32(XX[1]), 0);
      YEXT[2] = vextq_f32(YY[1], YY[2], 3);
      SUMM = vmlaq_lane_f32(SUMM, YEXT[2], vget_high_f32(XX[1]), 1);

      YY[0] = YY[2];
      len -= 8;
   }

   /* Consume 4 elements in x vector and 8 elements in y
    * vector. However, the 8'th element in y never really gets
    * touched in this loop. So, if len == 4, then we only
    * must access y[0] to y[6]. y[7] must not be accessed
    * hence make sure len>4 and not len>=4
    */
   if (len > 4) {
      yi += 4;
      YY[1] = vld1q_f32(yi);

      XX[0] = vld1q_f32(xi);
      xi += 4;

      SUMM = vmlaq_lane_f32(SUMM, YY[0], vget_low_f32(XX[0]), 0);
      YEXT[0] = vextq_f32(YY[0], YY[1], 1);
      SUMM = vmlaq_lane_f32(SUMM, YEXT[0], vget_low_f32(XX[0]), 1);
      YEXT[1] = vextq_f32(YY[0], YY[1], 2);
      SUMM = vmlaq_lane_f32(SUMM, YEXT[1], vget_high_f32(XX[0]), 0);
      YEXT[2] = vextq_f32(YY[0], YY[1], 3);
      SUMM = vmlaq_lane_f32(SUMM, YEXT[2], vget_high_f32(XX[0]), 1);

      YY[0] = YY[1];
      len -= 4;
   }

   while (--len > 0) {
      XX_2 = vld1_dup_f32(xi++);
      SUMM = vmlaq_lane_f32(SUMM, YY[0], XX_2, 0);
      YY[0]= vld1q_f32(++yi);
   }

   XX_2 = vld1_dup_f32(xi);
   SUMM = vmlaq_lane_f32(SUMM, YY[0], XX_2, 0);

   vst1q_f32(sum, SUMM);
}

void celt_pitch_xcorr_float_neon(const opus_val16 *_x, const opus_val16 *_y,
                        opus_val32 *xcorr, int len, int max_pitch, int arch) {
   int i;
   (void)arch;
   celt_assert(max_pitch > 0);
   celt_sig_assert((((size_t)_x)&3)==0);

   for (i = 0; i < (max_pitch-3); i += 4) {
      xcorr_kernel_neon_float((const float32_t *)_x, (const float32_t *)_y+i,
            (float32_t *)xcorr+i, len);
   }

   /* In case max_pitch isn't a multiple of 4, do non-unrolled version. */
   for (; i < max_pitch; i++) {
      xcorr[i] = celt_inner_prod_neon(_x, _y+i, len);
   }
}
#endif
