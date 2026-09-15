/***********************************************************************
Copyright (c) 2017 Google Inc.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <arm_neon.h>
#include "pitch.h"

#ifdef FIXED_POINT

opus_val32 celt_inner_prod_neon(const opus_val16 *x, const opus_val16 *y, int N)
{
    int i;
    opus_val32 xy;
    int16x8_t x_s16x8, y_s16x8;
    int32x4_t xy_s32x4 = vdupq_n_s32(0);
    int64x2_t xy_s64x2;
    int64x1_t xy_s64x1;

    for (i = 0; i < N - 7; i += 8) {
        x_s16x8  = vld1q_s16(&x[i]);
        y_s16x8  = vld1q_s16(&y[i]);
        xy_s32x4 = vmlal_s16(xy_s32x4, vget_low_s16 (x_s16x8), vget_low_s16 (y_s16x8));
        xy_s32x4 = vmlal_s16(xy_s32x4, vget_high_s16(x_s16x8), vget_high_s16(y_s16x8));
    }

    if (N - i >= 4) {
        const int16x4_t x_s16x4 = vld1_s16(&x[i]);
        const int16x4_t y_s16x4 = vld1_s16(&y[i]);
        xy_s32x4 = vmlal_s16(xy_s32x4, x_s16x4, y_s16x4);
        i += 4;
    }

    xy_s64x2 = vpaddlq_s32(xy_s32x4);
    xy_s64x1 = vadd_s64(vget_low_s64(xy_s64x2), vget_high_s64(xy_s64x2));
    xy       = vget_lane_s32(vreinterpret_s32_s64(xy_s64x1), 0);

    for (; i < N; i++) {
        xy = MAC16_16(xy, x[i], y[i]);
    }

#ifdef OPUS_CHECK_ASM
    celt_assert(celt_inner_prod_c(x, y, N) == xy);
#endif

    return xy;
}

void dual_inner_prod_neon(const opus_val16 *x, const opus_val16 *y01, const opus_val16 *y02,
        int N, opus_val32 *xy1, opus_val32 *xy2)
{
    int i;
    opus_val32 xy01, xy02;
    int16x8_t x_s16x8, y01_s16x8, y02_s16x8;
    int32x4_t xy01_s32x4 = vdupq_n_s32(0);
    int32x4_t xy02_s32x4 = vdupq_n_s32(0);
    int64x2_t xy01_s64x2, xy02_s64x2;
    int64x1_t xy01_s64x1, xy02_s64x1;

    for (i = 0; i < N - 7; i += 8) {
        x_s16x8    = vld1q_s16(&x[i]);
        y01_s16x8  = vld1q_s16(&y01[i]);
        y02_s16x8  = vld1q_s16(&y02[i]);
        xy01_s32x4 = vmlal_s16(xy01_s32x4, vget_low_s16 (x_s16x8), vget_low_s16 (y01_s16x8));
        xy02_s32x4 = vmlal_s16(xy02_s32x4, vget_low_s16 (x_s16x8), vget_low_s16 (y02_s16x8));
        xy01_s32x4 = vmlal_s16(xy01_s32x4, vget_high_s16(x_s16x8), vget_high_s16(y01_s16x8));
        xy02_s32x4 = vmlal_s16(xy02_s32x4, vget_high_s16(x_s16x8), vget_high_s16(y02_s16x8));
    }

    if (N - i >= 4) {
        const int16x4_t x_s16x4   = vld1_s16(&x[i]);
        const int16x4_t y01_s16x4 = vld1_s16(&y01[i]);
        const int16x4_t y02_s16x4 = vld1_s16(&y02[i]);
        xy01_s32x4 = vmlal_s16(xy01_s32x4, x_s16x4, y01_s16x4);
        xy02_s32x4 = vmlal_s16(xy02_s32x4, x_s16x4, y02_s16x4);
        i += 4;
    }

    xy01_s64x2 = vpaddlq_s32(xy01_s32x4);
    xy02_s64x2 = vpaddlq_s32(xy02_s32x4);
    xy01_s64x1 = vadd_s64(vget_low_s64(xy01_s64x2), vget_high_s64(xy01_s64x2));
    xy02_s64x1 = vadd_s64(vget_low_s64(xy02_s64x2), vget_high_s64(xy02_s64x2));
    xy01       = vget_lane_s32(vreinterpret_s32_s64(xy01_s64x1), 0);
    xy02       = vget_lane_s32(vreinterpret_s32_s64(xy02_s64x1), 0);

    for (; i < N; i++) {
        xy01 = MAC16_16(xy01, x[i], y01[i]);
        xy02 = MAC16_16(xy02, x[i], y02[i]);
    }
    *xy1 = xy01;
    *xy2 = xy02;

#ifdef OPUS_CHECK_ASM
    {
        opus_val32 xy1_c, xy2_c;
        dual_inner_prod_c(x, y01, y02, N, &xy1_c, &xy2_c);
        celt_assert(xy1_c == *xy1);
        celt_assert(xy2_c == *xy2);
    }
#endif
}

#else /* !FIXED_POINT */

/* ========================================================================== */

#ifdef __ARM_FEATURE_FMA
/* If we can, force the compiler to use an FMA instruction rather than break
   vmlaq_f32() into fmul/fadd. */
#define vmlaq_f32(a,b,c) vfmaq_f32(a,b,c)
#endif


#ifdef OPUS_CHECK_ASM

/* This part of code simulates floating-point NEON operations. */

/* celt_inner_prod_neon_float_c_simulation() simulates the floating-point   */
/* operations of celt_inner_prod_neon(), and both functions should have bit */
/* exact output.                                                            */
static opus_val32 celt_inner_prod_neon_float_c_simulation(const opus_val16 *x, const opus_val16 *y, float *err, int N)
{
   int i;
   *err = 0;
   opus_val32 xy, xy0 = 0, xy1 = 0, xy2 = 0, xy3 = 0;
   for (i = 0; i < N - 3; i += 4) {
      xy0 = MAC16_16(xy0, x[i + 0], y[i + 0]);
      xy1 = MAC16_16(xy1, x[i + 1], y[i + 1]);
      xy2 = MAC16_16(xy2, x[i + 2], y[i + 2]);
      xy3 = MAC16_16(xy3, x[i + 3], y[i + 3]);
      *err += ABS32(xy0)+ABS32(xy1)+ABS32(xy2)+ABS32(xy3);
   }
   xy0 += xy2;
   xy1 += xy3;
   xy = xy0 + xy1;
   *err += ABS32(xy1)+ABS32(xy0)+ABS32(xy);
   for (; i < N; i++) {
      xy = MAC16_16(xy, x[i], y[i]);
      *err += ABS32(xy);
   }
   *err = *err*2e-7 + N*1e-37;
   return xy;
}

/* dual_inner_prod_neon_float_c_simulation() simulates the floating-point   */
/* operations of dual_inner_prod_neon(), and both functions should have bit */
/* exact output.                                                            */
static void dual_inner_prod_neon_float_c_simulation(const opus_val16 *x, const opus_val16 *y01, const opus_val16 *y02,
      int N, opus_val32 *xy1, opus_val32 *xy2, float *err)
{
   *xy1 = celt_inner_prod_neon_float_c_simulation(x, y01, &err[0], N);
   *xy2 = celt_inner_prod_neon_float_c_simulation(x, y02, &err[1], N);
}

#endif /* OPUS_CHECK_ASM */

/* ========================================================================== */

opus_val32 celt_inner_prod_neon(const opus_val16 *x, const opus_val16 *y, int N)
{
    int i;
    opus_val32 xy;
    float32x4_t xy_f32x4 = vdupq_n_f32(0);
    float32x2_t xy_f32x2;

    for (i = 0; i < N - 7; i += 8) {
        float32x4_t x_f32x4, y_f32x4;
        x_f32x4  = vld1q_f32(&x[i]);
        y_f32x4  = vld1q_f32(&y[i]);
        xy_f32x4 = vmlaq_f32(xy_f32x4, x_f32x4, y_f32x4);
        x_f32x4  = vld1q_f32(&x[i + 4]);
        y_f32x4  = vld1q_f32(&y[i + 4]);
        xy_f32x4 = vmlaq_f32(xy_f32x4, x_f32x4, y_f32x4);
    }

    if (N - i >= 4) {
        const float32x4_t x_f32x4 = vld1q_f32(&x[i]);
        const float32x4_t y_f32x4 = vld1q_f32(&y[i]);
        xy_f32x4 = vmlaq_f32(xy_f32x4, x_f32x4, y_f32x4);
        i += 4;
    }

    xy_f32x2 = vadd_f32(vget_low_f32(xy_f32x4), vget_high_f32(xy_f32x4));
    xy_f32x2 = vpadd_f32(xy_f32x2, xy_f32x2);
    xy       = vget_lane_f32(xy_f32x2, 0);

    for (; i < N; i++) {
        xy = MAC16_16(xy, x[i], y[i]);
    }

#ifdef OPUS_CHECK_ASM
    {
        float err, res;
        res = celt_inner_prod_neon_float_c_simulation(x, y, &err, N);
        /*if (ABS32(res - xy) > err) fprintf(stderr, "%g %g %g\n", res, xy, err);*/
        celt_assert(ABS32(res - xy) <= err);
    }
#endif

    return xy;
}

void dual_inner_prod_neon(const opus_val16 *x, const opus_val16 *y01, const opus_val16 *y02,
        int N, opus_val32 *xy1, opus_val32 *xy2)
{
    int i;
    opus_val32 xy01, xy02;
    float32x4_t xy01_f32x4 = vdupq_n_f32(0);
    float32x4_t xy02_f32x4 = vdupq_n_f32(0);
    float32x2_t xy01_f32x2, xy02_f32x2;

    for (i = 0; i < N - 7; i += 8) {
        float32x4_t x_f32x4, y01_f32x4, y02_f32x4;
        x_f32x4    = vld1q_f32(&x[i]);
        y01_f32x4  = vld1q_f32(&y01[i]);
        y02_f32x4  = vld1q_f32(&y02[i]);
        xy01_f32x4 = vmlaq_f32(xy01_f32x4, x_f32x4, y01_f32x4);
        xy02_f32x4 = vmlaq_f32(xy02_f32x4, x_f32x4, y02_f32x4);
        x_f32x4    = vld1q_f32(&x[i + 4]);
        y01_f32x4  = vld1q_f32(&y01[i + 4]);
        y02_f32x4  = vld1q_f32(&y02[i + 4]);
        xy01_f32x4 = vmlaq_f32(xy01_f32x4, x_f32x4, y01_f32x4);
        xy02_f32x4 = vmlaq_f32(xy02_f32x4, x_f32x4, y02_f32x4);
    }

    if (N - i >= 4) {
        const float32x4_t x_f32x4   = vld1q_f32(&x[i]);
        const float32x4_t y01_f32x4 = vld1q_f32(&y01[i]);
        const float32x4_t y02_f32x4 = vld1q_f32(&y02[i]);
        xy01_f32x4 = vmlaq_f32(xy01_f32x4, x_f32x4, y01_f32x4);
        xy02_f32x4 = vmlaq_f32(xy02_f32x4, x_f32x4, y02_f32x4);
        i += 4;
    }

    xy01_f32x2 = vadd_f32(vget_low_f32(xy01_f32x4), vget_high_f32(xy01_f32x4));
    xy02_f32x2 = vadd_f32(vget_low_f32(xy02_f32x4), vget_high_f32(xy02_f32x4));
    xy01_f32x2 = vpadd_f32(xy01_f32x2, xy01_f32x2);
    xy02_f32x2 = vpadd_f32(xy02_f32x2, xy02_f32x2);
    xy01       = vget_lane_f32(xy01_f32x2, 0);
    xy02       = vget_lane_f32(xy02_f32x2, 0);

    for (; i < N; i++) {
        xy01 = MAC16_16(xy01, x[i], y01[i]);
        xy02 = MAC16_16(xy02, x[i], y02[i]);
    }
    *xy1 = xy01;
    *xy2 = xy02;

#ifdef OPUS_CHECK_ASM
    {
        opus_val32 xy1_c, xy2_c;
        float err[2];
        dual_inner_prod_neon_float_c_simulation(x, y01, y02, N, &xy1_c, &xy2_c, err);
        /*if (ABS32(xy1_c - *xy1) > err[0]) fprintf(stderr, "dual1 fail: %g %g %g\n", xy1_c, *xy1, err[0]);
        if (ABS32(xy2_c - *xy2) > err[1]) fprintf(stderr, "dual2 fail: %g %g %g\n", xy2_c, *xy2, err[1]);*/
        celt_assert(ABS32(xy1_c - *xy1) <= err[0]);
        celt_assert(ABS32(xy2_c - *xy2) <= err[1]);
    }
#endif
}

#endif /* FIXED_POINT */
