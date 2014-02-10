// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// NEON version of YUV to RGB upsampling functions.
//
// Author: mans@mansr.com (Mans Rullgard)
// Based on SSE code by: somnath@google.com (Somnath Banerjee)

#include "./dsp.h"

#if defined(WEBP_USE_NEON)

#include <assert.h>
#include <arm_neon.h>
#include <string.h>
#include "./yuv.h"

#ifdef FANCY_UPSAMPLING

//-----------------------------------------------------------------------------
// U/V upsampling

// Loads 9 pixels each from rows r1 and r2 and generates 16 pixels.
#define UPSAMPLE_16PIXELS(r1, r2, out) {                                \
  uint8x8_t a = vld1_u8(r1);                                            \
  uint8x8_t b = vld1_u8(r1 + 1);                                        \
  uint8x8_t c = vld1_u8(r2);                                            \
  uint8x8_t d = vld1_u8(r2 + 1);                                        \
                                                                        \
  uint16x8_t al = vshll_n_u8(a, 1);                                     \
  uint16x8_t bl = vshll_n_u8(b, 1);                                     \
  uint16x8_t cl = vshll_n_u8(c, 1);                                     \
  uint16x8_t dl = vshll_n_u8(d, 1);                                     \
                                                                        \
  uint8x8_t diag1, diag2;                                               \
  uint16x8_t sl;                                                        \
                                                                        \
  /* a + b + c + d */                                                   \
  sl = vaddl_u8(a,  b);                                                 \
  sl = vaddw_u8(sl, c);                                                 \
  sl = vaddw_u8(sl, d);                                                 \
                                                                        \
  al = vaddq_u16(sl, al); /* 3a +  b +  c +  d */                       \
  bl = vaddq_u16(sl, bl); /*  a + 3b +  c +  d */                       \
                                                                        \
  al = vaddq_u16(al, dl); /* 3a +  b +  c + 3d */                       \
  bl = vaddq_u16(bl, cl); /*  a + 3b + 3c +  d */                       \
                                                                        \
  diag2 = vshrn_n_u16(al, 3);                                           \
  diag1 = vshrn_n_u16(bl, 3);                                           \
                                                                        \
  a = vrhadd_u8(a, diag1);                                              \
  b = vrhadd_u8(b, diag2);                                              \
  c = vrhadd_u8(c, diag2);                                              \
  d = vrhadd_u8(d, diag1);                                              \
                                                                        \
  {                                                                     \
    const uint8x8x2_t a_b = {{ a, b }};                                 \
    const uint8x8x2_t c_d = {{ c, d }};                                 \
    vst2_u8(out,      a_b);                                             \
    vst2_u8(out + 32, c_d);                                             \
  }                                                                     \
}

// Turn the macro into a function for reducing code-size when non-critical
static void Upsample16Pixels(const uint8_t *r1, const uint8_t *r2,
                             uint8_t *out) {
  UPSAMPLE_16PIXELS(r1, r2, out);
}

#define UPSAMPLE_LAST_BLOCK(tb, bb, num_pixels, out) {                  \
  uint8_t r1[9], r2[9];                                                 \
  memcpy(r1, (tb), (num_pixels));                                       \
  memcpy(r2, (bb), (num_pixels));                                       \
  /* replicate last byte */                                             \
  memset(r1 + (num_pixels), r1[(num_pixels) - 1], 9 - (num_pixels));    \
  memset(r2 + (num_pixels), r2[(num_pixels) - 1], 9 - (num_pixels));    \
  Upsample16Pixels(r1, r2, out);                                        \
}

//-----------------------------------------------------------------------------
// YUV->RGB conversion

static const int16_t kCoeffs[4] = { kYScale, kVToR, kUToG, kVToG };

#define v255 vmov_n_u8(255)

#define STORE_Rgb(out, r, g, b) do {                                    \
  const uint8x8x3_t r_g_b = {{ r, g, b }};                              \
  vst3_u8(out, r_g_b);                                                  \
} while (0)

#define STORE_Bgr(out, r, g, b) do {                                    \
  const uint8x8x3_t b_g_r = {{ b, g, r }};                              \
  vst3_u8(out, b_g_r);                                                  \
} while (0)

#define STORE_Rgba(out, r, g, b) do {                                   \
  const uint8x8x4_t r_g_b_v255 = {{ r, g, b, v255 }};                   \
  vst4_u8(out, r_g_b_v255);                                             \
} while (0)

#define STORE_Bgra(out, r, g, b) do {                                   \
  const uint8x8x4_t b_g_r_v255 = {{ b, g, r, v255 }};                   \
  vst4_u8(out, b_g_r_v255);                                             \
} while (0)

#define CONVERT8(FMT, XSTEP, N, src_y, src_uv, out, cur_x) {            \
  int i;                                                                \
  for (i = 0; i < N; i += 8) {                                          \
    const int off = ((cur_x) + i) * XSTEP;                              \
    uint8x8_t y  = vld1_u8((src_y) + (cur_x)  + i);                     \
    uint8x8_t u  = vld1_u8((src_uv) + i);                               \
    uint8x8_t v  = vld1_u8((src_uv) + i + 16);                          \
    const int16x8_t yy = vreinterpretq_s16_u16(vsubl_u8(y, u16));       \
    const int16x8_t uu = vreinterpretq_s16_u16(vsubl_u8(u, u128));      \
    const int16x8_t vv = vreinterpretq_s16_u16(vsubl_u8(v, u128));      \
    int32x4_t yl = vmull_lane_s16(vget_low_s16(yy),  cf16, 0);          \
    int32x4_t yh = vmull_lane_s16(vget_high_s16(yy), cf16, 0);          \
    const int32x4_t rl = vmlal_lane_s16(yl, vget_low_s16(vv),  cf16, 1);\
    const int32x4_t rh = vmlal_lane_s16(yh, vget_high_s16(vv), cf16, 1);\
    int32x4_t gl = vmlsl_lane_s16(yl, vget_low_s16(uu),  cf16, 2);      \
    int32x4_t gh = vmlsl_lane_s16(yh, vget_high_s16(uu), cf16, 2);      \
    const int32x4_t bl = vmovl_s16(vget_low_s16(uu));                   \
    const int32x4_t bh = vmovl_s16(vget_high_s16(uu));                  \
    gl = vmlsl_lane_s16(gl, vget_low_s16(vv),  cf16, 3);                \
    gh = vmlsl_lane_s16(gh, vget_high_s16(vv), cf16, 3);                \
    yl = vmlaq_lane_s32(yl, bl, cf32, 0);                               \
    yh = vmlaq_lane_s32(yh, bh, cf32, 0);                               \
    /* vrshrn_n_s32() already incorporates the rounding constant */     \
    y = vqmovun_s16(vcombine_s16(vrshrn_n_s32(rl, YUV_FIX2),            \
                                 vrshrn_n_s32(rh, YUV_FIX2)));          \
    u = vqmovun_s16(vcombine_s16(vrshrn_n_s32(gl, YUV_FIX2),            \
                                 vrshrn_n_s32(gh, YUV_FIX2)));          \
    v = vqmovun_s16(vcombine_s16(vrshrn_n_s32(yl, YUV_FIX2),            \
                                 vrshrn_n_s32(yh, YUV_FIX2)));          \
    STORE_ ## FMT(out + off, y, u, v);                                  \
  }                                                                     \
}

#define CONVERT1(FUNC, XSTEP, N, src_y, src_uv, rgb, cur_x) {           \
  int i;                                                                \
  for (i = 0; i < N; i++) {                                             \
    const int off = ((cur_x) + i) * XSTEP;                              \
    const int y = src_y[(cur_x) + i];                                   \
    const int u = (src_uv)[i];                                          \
    const int v = (src_uv)[i + 16];                                     \
    FUNC(y, u, v, rgb + off);                                           \
  }                                                                     \
}

#define CONVERT2RGB_8(FMT, XSTEP, top_y, bottom_y, uv,                  \
                      top_dst, bottom_dst, cur_x, len) {                \
  CONVERT8(FMT, XSTEP, len, top_y, uv, top_dst, cur_x)                  \
  if (bottom_y != NULL) {                                               \
    CONVERT8(FMT, XSTEP, len, bottom_y, (uv) + 32, bottom_dst, cur_x)   \
  }                                                                     \
}

#define CONVERT2RGB_1(FUNC, XSTEP, top_y, bottom_y, uv,                 \
                      top_dst, bottom_dst, cur_x, len) {                \
  CONVERT1(FUNC, XSTEP, len, top_y, uv, top_dst, cur_x);                \
  if (bottom_y != NULL) {                                               \
    CONVERT1(FUNC, XSTEP, len, bottom_y, (uv) + 32, bottom_dst, cur_x); \
  }                                                                     \
}

#define NEON_UPSAMPLE_FUNC(FUNC_NAME, FMT, XSTEP)                       \
static void FUNC_NAME(const uint8_t *top_y, const uint8_t *bottom_y,    \
                      const uint8_t *top_u, const uint8_t *top_v,       \
                      const uint8_t *cur_u, const uint8_t *cur_v,       \
                      uint8_t *top_dst, uint8_t *bottom_dst, int len) { \
  int block;                                                            \
  /* 16 byte aligned array to cache reconstructed u and v */            \
  uint8_t uv_buf[2 * 32 + 15];                                          \
  uint8_t *const r_uv = (uint8_t*)((uintptr_t)(uv_buf + 15) & ~15);     \
  const int uv_len = (len + 1) >> 1;                                    \
  /* 9 pixels must be read-able for each block */                       \
  const int num_blocks = (uv_len - 1) >> 3;                             \
  const int leftover = uv_len - num_blocks * 8;                         \
  const int last_pos = 1 + 16 * num_blocks;                             \
                                                                        \
  const int u_diag = ((top_u[0] + cur_u[0]) >> 1) + 1;                  \
  const int v_diag = ((top_v[0] + cur_v[0]) >> 1) + 1;                  \
                                                                        \
  const int16x4_t cf16 = vld1_s16(kCoeffs);                             \
  const int32x2_t cf32 = vmov_n_s32(kUToB);                             \
  const uint8x8_t u16  = vmov_n_u8(16);                                 \
  const uint8x8_t u128 = vmov_n_u8(128);                                \
                                                                        \
  /* Treat the first pixel in regular way */                            \
  assert(top_y != NULL);                                                \
  {                                                                     \
    const int u0 = (top_u[0] + u_diag) >> 1;                            \
    const int v0 = (top_v[0] + v_diag) >> 1;                            \
    VP8YuvTo ## FMT(top_y[0], u0, v0, top_dst);                         \
  }                                                                     \
  if (bottom_y != NULL) {                                               \
    const int u0 = (cur_u[0] + u_diag) >> 1;                            \
    const int v0 = (cur_v[0] + v_diag) >> 1;                            \
    VP8YuvTo ## FMT(bottom_y[0], u0, v0, bottom_dst);                   \
  }                                                                     \
                                                                        \
  for (block = 0; block < num_blocks; ++block) {                        \
    UPSAMPLE_16PIXELS(top_u, cur_u, r_uv);                              \
    UPSAMPLE_16PIXELS(top_v, cur_v, r_uv + 16);                         \
    CONVERT2RGB_8(FMT, XSTEP, top_y, bottom_y, r_uv,                    \
                  top_dst, bottom_dst, 16 * block + 1, 16);             \
    top_u += 8;                                                         \
    cur_u += 8;                                                         \
    top_v += 8;                                                         \
    cur_v += 8;                                                         \
  }                                                                     \
                                                                        \
  UPSAMPLE_LAST_BLOCK(top_u, cur_u, leftover, r_uv);                    \
  UPSAMPLE_LAST_BLOCK(top_v, cur_v, leftover, r_uv + 16);               \
  CONVERT2RGB_1(VP8YuvTo ## FMT, XSTEP, top_y, bottom_y, r_uv,          \
                top_dst, bottom_dst, last_pos, len - last_pos);         \
}

// NEON variants of the fancy upsampler.
NEON_UPSAMPLE_FUNC(UpsampleRgbLinePairNEON,  Rgb,  3)
NEON_UPSAMPLE_FUNC(UpsampleBgrLinePairNEON,  Bgr,  3)
NEON_UPSAMPLE_FUNC(UpsampleRgbaLinePairNEON, Rgba, 4)
NEON_UPSAMPLE_FUNC(UpsampleBgraLinePairNEON, Bgra, 4)

#endif  // FANCY_UPSAMPLING

#endif   // WEBP_USE_NEON

//------------------------------------------------------------------------------

#ifdef FANCY_UPSAMPLING

extern WebPUpsampleLinePairFunc WebPUpsamplers[/* MODE_LAST */];

void WebPInitUpsamplersNEON(void) {
#if defined(WEBP_USE_NEON)
  WebPUpsamplers[MODE_RGB]  = UpsampleRgbLinePairNEON;
  WebPUpsamplers[MODE_RGBA] = UpsampleRgbaLinePairNEON;
  WebPUpsamplers[MODE_BGR]  = UpsampleBgrLinePairNEON;
  WebPUpsamplers[MODE_BGRA] = UpsampleBgraLinePairNEON;
#endif   // WEBP_USE_NEON
}

void WebPInitPremultiplyNEON(void) {
#if defined(WEBP_USE_NEON)
  WebPUpsamplers[MODE_rgbA] = UpsampleRgbaLinePairNEON;
  WebPUpsamplers[MODE_bgrA] = UpsampleBgraLinePairNEON;
#endif   // WEBP_USE_NEON
}

#else

// this empty function is to avoid an empty .o
void WebPInitPremultiplyNEON(void) {}

#endif  // FANCY_UPSAMPLING

