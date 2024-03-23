// Copyright 2022 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Speed-critical functions for Sharp YUV.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "sharpyuv/sharpyuv_dsp.h"

#if defined(WEBP_USE_NEON)
#include <assert.h>
#include <stdlib.h>
#include <arm_neon.h>

static uint16_t clip_NEON(int v, int max) {
  return (v < 0) ? 0 : (v > max) ? max : (uint16_t)v;
}

static uint64_t SharpYuvUpdateY_NEON(const uint16_t* ref, const uint16_t* src,
                                     uint16_t* dst, int len, int bit_depth) {
  const int max_y = (1 << bit_depth) - 1;
  int i;
  const int16x8_t zero = vdupq_n_s16(0);
  const int16x8_t max = vdupq_n_s16(max_y);
  uint64x2_t sum = vdupq_n_u64(0);
  uint64_t diff;

  for (i = 0; i + 8 <= len; i += 8) {
    const int16x8_t A = vreinterpretq_s16_u16(vld1q_u16(ref + i));
    const int16x8_t B = vreinterpretq_s16_u16(vld1q_u16(src + i));
    const int16x8_t C = vreinterpretq_s16_u16(vld1q_u16(dst + i));
    const int16x8_t D = vsubq_s16(A, B);       // diff_y
    const int16x8_t F = vaddq_s16(C, D);       // new_y
    const uint16x8_t H =
        vreinterpretq_u16_s16(vmaxq_s16(vminq_s16(F, max), zero));
    const int16x8_t I = vabsq_s16(D);          // abs(diff_y)
    vst1q_u16(dst + i, H);
    sum = vpadalq_u32(sum, vpaddlq_u16(vreinterpretq_u16_s16(I)));
  }
  diff = vgetq_lane_u64(sum, 0) + vgetq_lane_u64(sum, 1);
  for (; i < len; ++i) {
    const int diff_y = ref[i] - src[i];
    const int new_y = (int)(dst[i]) + diff_y;
    dst[i] = clip_NEON(new_y, max_y);
    diff += (uint64_t)(abs(diff_y));
  }
  return diff;
}

static void SharpYuvUpdateRGB_NEON(const int16_t* ref, const int16_t* src,
                                   int16_t* dst, int len) {
  int i;
  for (i = 0; i + 8 <= len; i += 8) {
    const int16x8_t A = vld1q_s16(ref + i);
    const int16x8_t B = vld1q_s16(src + i);
    const int16x8_t C = vld1q_s16(dst + i);
    const int16x8_t D = vsubq_s16(A, B);   // diff_uv
    const int16x8_t E = vaddq_s16(C, D);   // new_uv
    vst1q_s16(dst + i, E);
  }
  for (; i < len; ++i) {
    const int diff_uv = ref[i] - src[i];
    dst[i] += diff_uv;
  }
}

static void SharpYuvFilterRow16_NEON(const int16_t* A, const int16_t* B,
                                     int len, const uint16_t* best_y,
                                     uint16_t* out, int bit_depth) {
  const int max_y = (1 << bit_depth) - 1;
  int i;
  const int16x8_t max = vdupq_n_s16(max_y);
  const int16x8_t zero = vdupq_n_s16(0);
  for (i = 0; i + 8 <= len; i += 8) {
    const int16x8_t a0 = vld1q_s16(A + i + 0);
    const int16x8_t a1 = vld1q_s16(A + i + 1);
    const int16x8_t b0 = vld1q_s16(B + i + 0);
    const int16x8_t b1 = vld1q_s16(B + i + 1);
    const int16x8_t a0b1 = vaddq_s16(a0, b1);
    const int16x8_t a1b0 = vaddq_s16(a1, b0);
    const int16x8_t a0a1b0b1 = vaddq_s16(a0b1, a1b0);  // A0+A1+B0+B1
    const int16x8_t a0b1_2 = vaddq_s16(a0b1, a0b1);    // 2*(A0+B1)
    const int16x8_t a1b0_2 = vaddq_s16(a1b0, a1b0);    // 2*(A1+B0)
    const int16x8_t c0 = vshrq_n_s16(vaddq_s16(a0b1_2, a0a1b0b1), 3);
    const int16x8_t c1 = vshrq_n_s16(vaddq_s16(a1b0_2, a0a1b0b1), 3);
    const int16x8_t e0 = vrhaddq_s16(c1, a0);
    const int16x8_t e1 = vrhaddq_s16(c0, a1);
    const int16x8x2_t f = vzipq_s16(e0, e1);
    const int16x8_t g0 = vreinterpretq_s16_u16(vld1q_u16(best_y + 2 * i + 0));
    const int16x8_t g1 = vreinterpretq_s16_u16(vld1q_u16(best_y + 2 * i + 8));
    const int16x8_t h0 = vaddq_s16(g0, f.val[0]);
    const int16x8_t h1 = vaddq_s16(g1, f.val[1]);
    const int16x8_t i0 = vmaxq_s16(vminq_s16(h0, max), zero);
    const int16x8_t i1 = vmaxq_s16(vminq_s16(h1, max), zero);
    vst1q_u16(out + 2 * i + 0, vreinterpretq_u16_s16(i0));
    vst1q_u16(out + 2 * i + 8, vreinterpretq_u16_s16(i1));
  }
  for (; i < len; ++i) {
    const int a0b1 = A[i + 0] + B[i + 1];
    const int a1b0 = A[i + 1] + B[i + 0];
    const int a0a1b0b1 = a0b1 + a1b0 + 8;
    const int v0 = (8 * A[i + 0] + 2 * a1b0 + a0a1b0b1) >> 4;
    const int v1 = (8 * A[i + 1] + 2 * a0b1 + a0a1b0b1) >> 4;
    out[2 * i + 0] = clip_NEON(best_y[2 * i + 0] + v0, max_y);
    out[2 * i + 1] = clip_NEON(best_y[2 * i + 1] + v1, max_y);
  }
}

static void SharpYuvFilterRow32_NEON(const int16_t* A, const int16_t* B,
                                     int len, const uint16_t* best_y,
                                     uint16_t* out, int bit_depth) {
  const int max_y = (1 << bit_depth) - 1;
  int i;
  const uint16x8_t max = vdupq_n_u16(max_y);
  for (i = 0; i + 4 <= len; i += 4) {
    const int16x4_t a0 = vld1_s16(A + i + 0);
    const int16x4_t a1 = vld1_s16(A + i + 1);
    const int16x4_t b0 = vld1_s16(B + i + 0);
    const int16x4_t b1 = vld1_s16(B + i + 1);
    const int32x4_t a0b1 = vaddl_s16(a0, b1);
    const int32x4_t a1b0 = vaddl_s16(a1, b0);
    const int32x4_t a0a1b0b1 = vaddq_s32(a0b1, a1b0);  // A0+A1+B0+B1
    const int32x4_t a0b1_2 = vaddq_s32(a0b1, a0b1);    // 2*(A0+B1)
    const int32x4_t a1b0_2 = vaddq_s32(a1b0, a1b0);    // 2*(A1+B0)
    const int32x4_t c0 = vshrq_n_s32(vaddq_s32(a0b1_2, a0a1b0b1), 3);
    const int32x4_t c1 = vshrq_n_s32(vaddq_s32(a1b0_2, a0a1b0b1), 3);
    const int32x4_t e0 = vrhaddq_s32(c1, vmovl_s16(a0));
    const int32x4_t e1 = vrhaddq_s32(c0, vmovl_s16(a1));
    const int32x4x2_t f = vzipq_s32(e0, e1);

    const int16x8_t g = vreinterpretq_s16_u16(vld1q_u16(best_y + 2 * i));
    const int32x4_t h0 = vaddw_s16(f.val[0], vget_low_s16(g));
    const int32x4_t h1 = vaddw_s16(f.val[1], vget_high_s16(g));
    const uint16x8_t i_16 = vcombine_u16(vqmovun_s32(h0), vqmovun_s32(h1));
    const uint16x8_t i_clamped = vminq_u16(i_16, max);
    vst1q_u16(out + 2 * i + 0, i_clamped);
  }
  for (; i < len; ++i) {
    const int a0b1 = A[i + 0] + B[i + 1];
    const int a1b0 = A[i + 1] + B[i + 0];
    const int a0a1b0b1 = a0b1 + a1b0 + 8;
    const int v0 = (8 * A[i + 0] + 2 * a1b0 + a0a1b0b1) >> 4;
    const int v1 = (8 * A[i + 1] + 2 * a0b1 + a0a1b0b1) >> 4;
    out[2 * i + 0] = clip_NEON(best_y[2 * i + 0] + v0, max_y);
    out[2 * i + 1] = clip_NEON(best_y[2 * i + 1] + v1, max_y);
  }
}

static void SharpYuvFilterRow_NEON(const int16_t* A, const int16_t* B, int len,
                                   const uint16_t* best_y, uint16_t* out,
                                   int bit_depth) {
  if (bit_depth <= 10) {
    SharpYuvFilterRow16_NEON(A, B, len, best_y, out, bit_depth);
  } else {
    SharpYuvFilterRow32_NEON(A, B, len, best_y, out, bit_depth);
  }
}

//------------------------------------------------------------------------------

extern void InitSharpYuvNEON(void);

WEBP_TSAN_IGNORE_FUNCTION void InitSharpYuvNEON(void) {
  SharpYuvUpdateY = SharpYuvUpdateY_NEON;
  SharpYuvUpdateRGB = SharpYuvUpdateRGB_NEON;
  SharpYuvFilterRow = SharpYuvFilterRow_NEON;
}

#else  // !WEBP_USE_NEON

extern void InitSharpYuvNEON(void);

void InitSharpYuvNEON(void) {}

#endif  // WEBP_USE_NEON
