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
#include "./neon.h"
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
    uint8x8x2_t a_b, c_d;                                               \
    INIT_VECTOR2(a_b, a, b);                                            \
    INIT_VECTOR2(c_d, c, d);                                            \
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

// note: we represent the 33050 large constant as 32768 + 282
static const int16_t kCoeffs1[4] = { 19077, 26149, 6419, 13320 };

#define v255 vdup_n_u8(255)
#define v_0x0f vdup_n_u8(15)

#define STORE_Rgb(out, r, g, b) do {                                    \
  uint8x8x3_t r_g_b;                                                    \
  INIT_VECTOR3(r_g_b, r, g, b);                                         \
  vst3_u8(out, r_g_b);                                                  \
} while (0)

#define STORE_Bgr(out, r, g, b) do {                                    \
  uint8x8x3_t b_g_r;                                                    \
  INIT_VECTOR3(b_g_r, b, g, r);                                         \
  vst3_u8(out, b_g_r);                                                  \
} while (0)

#define STORE_Rgba(out, r, g, b) do {                                   \
  uint8x8x4_t r_g_b_v255;                                               \
  INIT_VECTOR4(r_g_b_v255, r, g, b, v255);                              \
  vst4_u8(out, r_g_b_v255);                                             \
} while (0)

#define STORE_Bgra(out, r, g, b) do {                                   \
  uint8x8x4_t b_g_r_v255;                                               \
  INIT_VECTOR4(b_g_r_v255, b, g, r, v255);                              \
  vst4_u8(out, b_g_r_v255);                                             \
} while (0)

#define STORE_Argb(out, r, g, b) do {                                   \
  uint8x8x4_t v255_r_g_b;                                               \
  INIT_VECTOR4(v255_r_g_b, v255, r, g, b);                              \
  vst4_u8(out, v255_r_g_b);                                             \
} while (0)

#if !defined(WEBP_SWAP_16BIT_CSP)
#define ZIP_U8(lo, hi) vzip_u8((lo), (hi))
#else
#define ZIP_U8(lo, hi) vzip_u8((hi), (lo))
#endif

#define STORE_Rgba4444(out, r, g, b) do {                               \
  const uint8x8_t r1 = vshl_n_u8(vshr_n_u8(r, 4), 4);  /* 4bits */      \
  const uint8x8_t g1 = vshr_n_u8(g, 4);                                 \
  const uint8x8_t ba = vorr_u8(b, v_0x0f);                              \
  const uint8x8_t rg = vorr_u8(r1, g1);                                 \
  const uint8x8x2_t rgba4444 = ZIP_U8(rg, ba);                          \
  vst1q_u8(out, vcombine_u8(rgba4444.val[0], rgba4444.val[1]));         \
} while (0)

#define STORE_Rgb565(out, r, g, b) do {                                 \
  const uint8x8_t r1 = vshl_n_u8(vshr_n_u8(r, 3), 3);  /* 5bits */      \
  const uint8x8_t g1 = vshr_n_u8(g, 5);                /* upper 3bits */\
  const uint8x8_t g2 = vshl_n_u8(vshr_n_u8(g, 2), 5);  /* lower 3bits */\
  const uint8x8_t b1 = vshr_n_u8(b, 3);                /* 5bits */      \
  const uint8x8_t rg = vorr_u8(r1, g1);                                 \
  const uint8x8_t gb = vorr_u8(g2, b1);                                 \
  const uint8x8x2_t rgb565 = ZIP_U8(rg, gb);                            \
  vst1q_u8(out, vcombine_u8(rgb565.val[0], rgb565.val[1]));             \
} while (0)

#define CONVERT8(FMT, XSTEP, N, src_y, src_uv, out, cur_x) do {         \
  int i;                                                                \
  for (i = 0; i < N; i += 8) {                                          \
    const int off = ((cur_x) + i) * XSTEP;                              \
    const uint8x8_t y  = vld1_u8((src_y) + (cur_x)  + i);               \
    const uint8x8_t u  = vld1_u8((src_uv) + i +  0);                    \
    const uint8x8_t v  = vld1_u8((src_uv) + i + 16);                    \
    const int16x8_t Y0 = vreinterpretq_s16_u16(vshll_n_u8(y, 7));       \
    const int16x8_t U0 = vreinterpretq_s16_u16(vshll_n_u8(u, 7));       \
    const int16x8_t V0 = vreinterpretq_s16_u16(vshll_n_u8(v, 7));       \
    const int16x8_t Y1 = vqdmulhq_lane_s16(Y0, coeff1, 0);              \
    const int16x8_t R0 = vqdmulhq_lane_s16(V0, coeff1, 1);              \
    const int16x8_t G0 = vqdmulhq_lane_s16(U0, coeff1, 2);              \
    const int16x8_t G1 = vqdmulhq_lane_s16(V0, coeff1, 3);              \
    const int16x8_t B0 = vqdmulhq_n_s16(U0, 282);                       \
    const int16x8_t R1 = vqaddq_s16(Y1, R_Rounder);                     \
    const int16x8_t G2 = vqaddq_s16(Y1, G_Rounder);                     \
    const int16x8_t B1 = vqaddq_s16(Y1, B_Rounder);                     \
    const int16x8_t R2 = vqaddq_s16(R0, R1);                            \
    const int16x8_t G3 = vqaddq_s16(G0, G1);                            \
    const int16x8_t B2 = vqaddq_s16(B0, B1);                            \
    const int16x8_t G4 = vqsubq_s16(G2, G3);                            \
    const int16x8_t B3 = vqaddq_s16(B2, U0);                            \
    const uint8x8_t R = vqshrun_n_s16(R2, YUV_FIX2);                    \
    const uint8x8_t G = vqshrun_n_s16(G4, YUV_FIX2);                    \
    const uint8x8_t B = vqshrun_n_s16(B3, YUV_FIX2);                    \
    STORE_ ## FMT(out + off, R, G, B);                                  \
  }                                                                     \
} while (0)

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
  CONVERT8(FMT, XSTEP, len, top_y, uv, top_dst, cur_x);                 \
  if (bottom_y != NULL) {                                               \
    CONVERT8(FMT, XSTEP, len, bottom_y, (uv) + 32, bottom_dst, cur_x);  \
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
  const int16x4_t coeff1 = vld1_s16(kCoeffs1);                          \
  const int16x8_t R_Rounder = vdupq_n_s16(-14234);                      \
  const int16x8_t G_Rounder = vdupq_n_s16(8708);                        \
  const int16x8_t B_Rounder = vdupq_n_s16(-17685);                      \
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
NEON_UPSAMPLE_FUNC(UpsampleRgbLinePair,  Rgb,  3)
NEON_UPSAMPLE_FUNC(UpsampleBgrLinePair,  Bgr,  3)
NEON_UPSAMPLE_FUNC(UpsampleRgbaLinePair, Rgba, 4)
NEON_UPSAMPLE_FUNC(UpsampleBgraLinePair, Bgra, 4)
NEON_UPSAMPLE_FUNC(UpsampleArgbLinePair, Argb, 4)
NEON_UPSAMPLE_FUNC(UpsampleRgba4444LinePair, Rgba4444, 2)
NEON_UPSAMPLE_FUNC(UpsampleRgb565LinePair, Rgb565, 2)

//------------------------------------------------------------------------------
// Entry point

extern WebPUpsampleLinePairFunc WebPUpsamplers[/* MODE_LAST */];

extern void WebPInitUpsamplersNEON(void);

WEBP_TSAN_IGNORE_FUNCTION void WebPInitUpsamplersNEON(void) {
  WebPUpsamplers[MODE_RGB]  = UpsampleRgbLinePair;
  WebPUpsamplers[MODE_RGBA] = UpsampleRgbaLinePair;
  WebPUpsamplers[MODE_BGR]  = UpsampleBgrLinePair;
  WebPUpsamplers[MODE_BGRA] = UpsampleBgraLinePair;
  WebPUpsamplers[MODE_ARGB] = UpsampleArgbLinePair;
  WebPUpsamplers[MODE_rgbA] = UpsampleRgbaLinePair;
  WebPUpsamplers[MODE_bgrA] = UpsampleBgraLinePair;
  WebPUpsamplers[MODE_Argb] = UpsampleArgbLinePair;
  WebPUpsamplers[MODE_RGB_565] = UpsampleRgb565LinePair;
  WebPUpsamplers[MODE_RGBA_4444] = UpsampleRgba4444LinePair;
  WebPUpsamplers[MODE_rgbA_4444] = UpsampleRgba4444LinePair;
}

#endif  // FANCY_UPSAMPLING

#endif  // WEBP_USE_NEON

#if !(defined(FANCY_UPSAMPLING) && defined(WEBP_USE_NEON))
WEBP_DSP_INIT_STUB(WebPInitUpsamplersNEON)
#endif
