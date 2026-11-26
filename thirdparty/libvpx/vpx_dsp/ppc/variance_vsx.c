/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/ppc/types_vsx.h"

uint32_t vpx_get4x4sse_cs_vsx(const uint8_t *src_ptr, int src_stride,
                              const uint8_t *ref_ptr, int ref_stride) {
  int distortion;

  const int16x8_t a0 = unpack_to_s16_h(read4x2(src_ptr, src_stride));
  const int16x8_t a1 =
      unpack_to_s16_h(read4x2(src_ptr + src_stride * 2, src_stride));
  const int16x8_t b0 = unpack_to_s16_h(read4x2(ref_ptr, ref_stride));
  const int16x8_t b1 =
      unpack_to_s16_h(read4x2(ref_ptr + ref_stride * 2, ref_stride));
  const int16x8_t d0 = vec_sub(a0, b0);
  const int16x8_t d1 = vec_sub(a1, b1);
  const int32x4_t ds = vec_msum(d1, d1, vec_msum(d0, d0, vec_splat_s32(0)));
  const int32x4_t d = vec_splat(vec_sums(ds, vec_splat_s32(0)), 3);

  vec_ste(d, 0, &distortion);

  return distortion;
}

// TODO(lu_zero): Unroll
uint32_t vpx_get_mb_ss_vsx(const int16_t *src_ptr) {
  unsigned int i, sum = 0;
  int32x4_t s = vec_splat_s32(0);

  for (i = 0; i < 256; i += 8) {
    const int16x8_t v = vec_vsx_ld(0, src_ptr + i);
    s = vec_msum(v, v, s);
  }

  s = vec_splat(vec_sums(s, vec_splat_s32(0)), 3);

  vec_ste((uint32x4_t)s, 0, &sum);

  return sum;
}

void vpx_comp_avg_pred_vsx(uint8_t *comp_pred, const uint8_t *pred, int width,
                           int height, const uint8_t *ref, int ref_stride) {
  int i, j;
  /* comp_pred and pred must be 16 byte aligned. */
  assert(((intptr_t)comp_pred & 0xf) == 0);
  assert(((intptr_t)pred & 0xf) == 0);
  if (width >= 16) {
    for (i = 0; i < height; ++i) {
      for (j = 0; j < width; j += 16) {
        const uint8x16_t v = vec_avg(vec_vsx_ld(j, pred), vec_vsx_ld(j, ref));
        vec_vsx_st(v, j, comp_pred);
      }
      comp_pred += width;
      pred += width;
      ref += ref_stride;
    }
  } else if (width == 8) {
    // Process 2 lines at time
    for (i = 0; i < height / 2; ++i) {
      const uint8x16_t r0 = vec_vsx_ld(0, ref);
      const uint8x16_t r1 = vec_vsx_ld(0, ref + ref_stride);
      const uint8x16_t r = xxpermdi(r0, r1, 0);
      const uint8x16_t v = vec_avg(vec_vsx_ld(0, pred), r);
      vec_vsx_st(v, 0, comp_pred);
      comp_pred += 16;  // width * 2;
      pred += 16;       // width * 2;
      ref += ref_stride * 2;
    }
  } else {
    assert(width == 4);
    // process 4 lines at time
    for (i = 0; i < height / 4; ++i) {
      const uint32x4_t r0 = (uint32x4_t)vec_vsx_ld(0, ref);
      const uint32x4_t r1 = (uint32x4_t)vec_vsx_ld(0, ref + ref_stride);
      const uint32x4_t r2 = (uint32x4_t)vec_vsx_ld(0, ref + ref_stride * 2);
      const uint32x4_t r3 = (uint32x4_t)vec_vsx_ld(0, ref + ref_stride * 3);
      const uint8x16_t r =
          (uint8x16_t)xxpermdi(vec_mergeh(r0, r1), vec_mergeh(r2, r3), 0);
      const uint8x16_t v = vec_avg(vec_vsx_ld(0, pred), r);
      vec_vsx_st(v, 0, comp_pred);
      comp_pred += 16;  // width * 4;
      pred += 16;       // width * 4;
      ref += ref_stride * 4;
    }
  }
}

static INLINE void variance_inner_32(const uint8_t *src_ptr,
                                     const uint8_t *ref_ptr,
                                     int32x4_t *sum_squared, int32x4_t *sum) {
  int32x4_t s = *sum;
  int32x4_t ss = *sum_squared;

  const uint8x16_t va0 = vec_vsx_ld(0, src_ptr);
  const uint8x16_t vb0 = vec_vsx_ld(0, ref_ptr);
  const uint8x16_t va1 = vec_vsx_ld(16, src_ptr);
  const uint8x16_t vb1 = vec_vsx_ld(16, ref_ptr);

  const int16x8_t a0 = unpack_to_s16_h(va0);
  const int16x8_t b0 = unpack_to_s16_h(vb0);
  const int16x8_t a1 = unpack_to_s16_l(va0);
  const int16x8_t b1 = unpack_to_s16_l(vb0);
  const int16x8_t a2 = unpack_to_s16_h(va1);
  const int16x8_t b2 = unpack_to_s16_h(vb1);
  const int16x8_t a3 = unpack_to_s16_l(va1);
  const int16x8_t b3 = unpack_to_s16_l(vb1);
  const int16x8_t d0 = vec_sub(a0, b0);
  const int16x8_t d1 = vec_sub(a1, b1);
  const int16x8_t d2 = vec_sub(a2, b2);
  const int16x8_t d3 = vec_sub(a3, b3);

  s = vec_sum4s(d0, s);
  ss = vec_msum(d0, d0, ss);
  s = vec_sum4s(d1, s);
  ss = vec_msum(d1, d1, ss);
  s = vec_sum4s(d2, s);
  ss = vec_msum(d2, d2, ss);
  s = vec_sum4s(d3, s);
  ss = vec_msum(d3, d3, ss);
  *sum = s;
  *sum_squared = ss;
}

static INLINE void variance(const uint8_t *src_ptr, int src_stride,
                            const uint8_t *ref_ptr, int ref_stride, int w,
                            int h, uint32_t *sse, int *sum) {
  int i;

  int32x4_t s = vec_splat_s32(0);
  int32x4_t ss = vec_splat_s32(0);

  switch (w) {
    case 4:
      for (i = 0; i < h / 2; ++i) {
        const int16x8_t a0 = unpack_to_s16_h(read4x2(src_ptr, src_stride));
        const int16x8_t b0 = unpack_to_s16_h(read4x2(ref_ptr, ref_stride));
        const int16x8_t d = vec_sub(a0, b0);
        s = vec_sum4s(d, s);
        ss = vec_msum(d, d, ss);
        src_ptr += src_stride * 2;
        ref_ptr += ref_stride * 2;
      }
      break;
    case 8:
      for (i = 0; i < h; ++i) {
        const int16x8_t a0 = unpack_to_s16_h(vec_vsx_ld(0, src_ptr));
        const int16x8_t b0 = unpack_to_s16_h(vec_vsx_ld(0, ref_ptr));
        const int16x8_t d = vec_sub(a0, b0);

        s = vec_sum4s(d, s);
        ss = vec_msum(d, d, ss);
        src_ptr += src_stride;
        ref_ptr += ref_stride;
      }
      break;
    case 16:
      for (i = 0; i < h; ++i) {
        const uint8x16_t va = vec_vsx_ld(0, src_ptr);
        const uint8x16_t vb = vec_vsx_ld(0, ref_ptr);
        const int16x8_t a0 = unpack_to_s16_h(va);
        const int16x8_t b0 = unpack_to_s16_h(vb);
        const int16x8_t a1 = unpack_to_s16_l(va);
        const int16x8_t b1 = unpack_to_s16_l(vb);
        const int16x8_t d0 = vec_sub(a0, b0);
        const int16x8_t d1 = vec_sub(a1, b1);

        s = vec_sum4s(d0, s);
        ss = vec_msum(d0, d0, ss);
        s = vec_sum4s(d1, s);
        ss = vec_msum(d1, d1, ss);

        src_ptr += src_stride;
        ref_ptr += ref_stride;
      }
      break;
    case 32:
      for (i = 0; i < h; ++i) {
        variance_inner_32(src_ptr, ref_ptr, &ss, &s);
        src_ptr += src_stride;
        ref_ptr += ref_stride;
      }
      break;
    case 64:
      for (i = 0; i < h; ++i) {
        variance_inner_32(src_ptr, ref_ptr, &ss, &s);
        variance_inner_32(src_ptr + 32, ref_ptr + 32, &ss, &s);

        src_ptr += src_stride;
        ref_ptr += ref_stride;
      }
      break;
  }

  s = vec_splat(vec_sums(s, vec_splat_s32(0)), 3);

  vec_ste(s, 0, sum);

  ss = vec_splat(vec_sums(ss, vec_splat_s32(0)), 3);

  vec_ste((uint32x4_t)ss, 0, sse);
}

/* Identical to the variance call except it takes an additional parameter, sum,
 * and returns that value using pass-by-reference instead of returning
 * sse - sum^2 / w*h
 */
#define GET_VAR(W, H)                                                    \
  void vpx_get##W##x##H##var_vsx(const uint8_t *src_ptr, int src_stride, \
                                 const uint8_t *ref_ptr, int ref_stride, \
                                 uint32_t *sse, int *sum) {              \
    variance(src_ptr, src_stride, ref_ptr, ref_stride, W, H, sse, sum);  \
  }

/* Identical to the variance call except it does not calculate the
 * sse - sum^2 / w*h and returns sse in addition to modifying the passed in
 * variable.
 */
#define MSE(W, H)                                                         \
  uint32_t vpx_mse##W##x##H##_vsx(const uint8_t *src_ptr, int src_stride, \
                                  const uint8_t *ref_ptr, int ref_stride, \
                                  uint32_t *sse) {                        \
    int sum;                                                              \
    variance(src_ptr, src_stride, ref_ptr, ref_stride, W, H, sse, &sum);  \
    return *sse;                                                          \
  }

#define VAR(W, H)                                                              \
  uint32_t vpx_variance##W##x##H##_vsx(const uint8_t *src_ptr, int src_stride, \
                                       const uint8_t *ref_ptr, int ref_stride, \
                                       uint32_t *sse) {                        \
    int sum;                                                                   \
    variance(src_ptr, src_stride, ref_ptr, ref_stride, W, H, sse, &sum);       \
    return *sse - (uint32_t)(((int64_t)sum * sum) / ((W) * (H)));              \
  }

#define VARIANCES(W, H) VAR(W, H)

VARIANCES(64, 64)
VARIANCES(64, 32)
VARIANCES(32, 64)
VARIANCES(32, 32)
VARIANCES(32, 16)
VARIANCES(16, 32)
VARIANCES(16, 16)
VARIANCES(16, 8)
VARIANCES(8, 16)
VARIANCES(8, 8)
VARIANCES(8, 4)
VARIANCES(4, 8)
VARIANCES(4, 4)

GET_VAR(16, 16)
GET_VAR(8, 8)

MSE(16, 16)
MSE(16, 8)
MSE(8, 16)
MSE(8, 8)
