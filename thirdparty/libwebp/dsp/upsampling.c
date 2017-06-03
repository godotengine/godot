// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// YUV to RGB upsampling functions.
//
// Author: somnath@google.com (Somnath Banerjee)

#include "./dsp.h"
#include "./yuv.h"

#include <assert.h>

//------------------------------------------------------------------------------
// Fancy upsampler

#ifdef FANCY_UPSAMPLING

// Fancy upsampling functions to convert YUV to RGB
WebPUpsampleLinePairFunc WebPUpsamplers[MODE_LAST];

// Given samples laid out in a square as:
//  [a b]
//  [c d]
// we interpolate u/v as:
//  ([9*a + 3*b + 3*c +   d    3*a + 9*b + 3*c +   d] + [8 8]) / 16
//  ([3*a +   b + 9*c + 3*d      a + 3*b + 3*c + 9*d]   [8 8]) / 16

// We process u and v together stashed into 32bit (16bit each).
#define LOAD_UV(u, v) ((u) | ((v) << 16))

#define UPSAMPLE_FUNC(FUNC_NAME, FUNC, XSTEP)                                  \
static void FUNC_NAME(const uint8_t* top_y, const uint8_t* bottom_y,           \
                      const uint8_t* top_u, const uint8_t* top_v,              \
                      const uint8_t* cur_u, const uint8_t* cur_v,              \
                      uint8_t* top_dst, uint8_t* bottom_dst, int len) {        \
  int x;                                                                       \
  const int last_pixel_pair = (len - 1) >> 1;                                  \
  uint32_t tl_uv = LOAD_UV(top_u[0], top_v[0]);   /* top-left sample */        \
  uint32_t l_uv  = LOAD_UV(cur_u[0], cur_v[0]);   /* left-sample */            \
  assert(top_y != NULL);                                                       \
  {                                                                            \
    const uint32_t uv0 = (3 * tl_uv + l_uv + 0x00020002u) >> 2;                \
    FUNC(top_y[0], uv0 & 0xff, (uv0 >> 16), top_dst);                          \
  }                                                                            \
  if (bottom_y != NULL) {                                                      \
    const uint32_t uv0 = (3 * l_uv + tl_uv + 0x00020002u) >> 2;                \
    FUNC(bottom_y[0], uv0 & 0xff, (uv0 >> 16), bottom_dst);                    \
  }                                                                            \
  for (x = 1; x <= last_pixel_pair; ++x) {                                     \
    const uint32_t t_uv = LOAD_UV(top_u[x], top_v[x]);  /* top sample */       \
    const uint32_t uv   = LOAD_UV(cur_u[x], cur_v[x]);  /* sample */           \
    /* precompute invariant values associated with first and second diagonals*/\
    const uint32_t avg = tl_uv + t_uv + l_uv + uv + 0x00080008u;               \
    const uint32_t diag_12 = (avg + 2 * (t_uv + l_uv)) >> 3;                   \
    const uint32_t diag_03 = (avg + 2 * (tl_uv + uv)) >> 3;                    \
    {                                                                          \
      const uint32_t uv0 = (diag_12 + tl_uv) >> 1;                             \
      const uint32_t uv1 = (diag_03 + t_uv) >> 1;                              \
      FUNC(top_y[2 * x - 1], uv0 & 0xff, (uv0 >> 16),                          \
           top_dst + (2 * x - 1) * XSTEP);                                     \
      FUNC(top_y[2 * x - 0], uv1 & 0xff, (uv1 >> 16),                          \
           top_dst + (2 * x - 0) * XSTEP);                                     \
    }                                                                          \
    if (bottom_y != NULL) {                                                    \
      const uint32_t uv0 = (diag_03 + l_uv) >> 1;                              \
      const uint32_t uv1 = (diag_12 + uv) >> 1;                                \
      FUNC(bottom_y[2 * x - 1], uv0 & 0xff, (uv0 >> 16),                       \
           bottom_dst + (2 * x - 1) * XSTEP);                                  \
      FUNC(bottom_y[2 * x + 0], uv1 & 0xff, (uv1 >> 16),                       \
           bottom_dst + (2 * x + 0) * XSTEP);                                  \
    }                                                                          \
    tl_uv = t_uv;                                                              \
    l_uv = uv;                                                                 \
  }                                                                            \
  if (!(len & 1)) {                                                            \
    {                                                                          \
      const uint32_t uv0 = (3 * tl_uv + l_uv + 0x00020002u) >> 2;              \
      FUNC(top_y[len - 1], uv0 & 0xff, (uv0 >> 16),                            \
           top_dst + (len - 1) * XSTEP);                                       \
    }                                                                          \
    if (bottom_y != NULL) {                                                    \
      const uint32_t uv0 = (3 * l_uv + tl_uv + 0x00020002u) >> 2;              \
      FUNC(bottom_y[len - 1], uv0 & 0xff, (uv0 >> 16),                         \
           bottom_dst + (len - 1) * XSTEP);                                    \
    }                                                                          \
  }                                                                            \
}

// All variants implemented.
UPSAMPLE_FUNC(UpsampleRgbLinePair,  VP8YuvToRgb,  3)
UPSAMPLE_FUNC(UpsampleBgrLinePair,  VP8YuvToBgr,  3)
UPSAMPLE_FUNC(UpsampleRgbaLinePair, VP8YuvToRgba, 4)
UPSAMPLE_FUNC(UpsampleBgraLinePair, VP8YuvToBgra, 4)
UPSAMPLE_FUNC(UpsampleArgbLinePair, VP8YuvToArgb, 4)
UPSAMPLE_FUNC(UpsampleRgba4444LinePair, VP8YuvToRgba4444, 2)
UPSAMPLE_FUNC(UpsampleRgb565LinePair,  VP8YuvToRgb565,  2)

#undef LOAD_UV
#undef UPSAMPLE_FUNC

#endif  // FANCY_UPSAMPLING

//------------------------------------------------------------------------------

#if !defined(FANCY_UPSAMPLING)
#define DUAL_SAMPLE_FUNC(FUNC_NAME, FUNC)                                      \
static void FUNC_NAME(const uint8_t* top_y, const uint8_t* bot_y,              \
                      const uint8_t* top_u, const uint8_t* top_v,              \
                      const uint8_t* bot_u, const uint8_t* bot_v,              \
                      uint8_t* top_dst, uint8_t* bot_dst, int len) {           \
  const int half_len = len >> 1;                                               \
  int x;                                                                       \
  assert(top_dst != NULL);                                                     \
  {                                                                            \
    for (x = 0; x < half_len; ++x) {                                           \
      FUNC(top_y[2 * x + 0], top_u[x], top_v[x], top_dst + 8 * x + 0);         \
      FUNC(top_y[2 * x + 1], top_u[x], top_v[x], top_dst + 8 * x + 4);         \
    }                                                                          \
    if (len & 1) FUNC(top_y[2 * x + 0], top_u[x], top_v[x], top_dst + 8 * x);  \
  }                                                                            \
  if (bot_dst != NULL) {                                                       \
    for (x = 0; x < half_len; ++x) {                                           \
      FUNC(bot_y[2 * x + 0], bot_u[x], bot_v[x], bot_dst + 8 * x + 0);         \
      FUNC(bot_y[2 * x + 1], bot_u[x], bot_v[x], bot_dst + 8 * x + 4);         \
    }                                                                          \
    if (len & 1) FUNC(bot_y[2 * x + 0], bot_u[x], bot_v[x], bot_dst + 8 * x);  \
  }                                                                            \
}

DUAL_SAMPLE_FUNC(DualLineSamplerBGRA, VP8YuvToBgra)
DUAL_SAMPLE_FUNC(DualLineSamplerARGB, VP8YuvToArgb)
#undef DUAL_SAMPLE_FUNC

#endif  // !FANCY_UPSAMPLING

WebPUpsampleLinePairFunc WebPGetLinePairConverter(int alpha_is_last) {
  WebPInitUpsamplers();
  VP8YUVInit();
#ifdef FANCY_UPSAMPLING
  return WebPUpsamplers[alpha_is_last ? MODE_BGRA : MODE_ARGB];
#else
  return (alpha_is_last ? DualLineSamplerBGRA : DualLineSamplerARGB);
#endif
}

//------------------------------------------------------------------------------
// YUV444 converter

#define YUV444_FUNC(FUNC_NAME, FUNC, XSTEP)                                    \
extern void FUNC_NAME(const uint8_t* y, const uint8_t* u, const uint8_t* v,    \
                      uint8_t* dst, int len);                                  \
void FUNC_NAME(const uint8_t* y, const uint8_t* u, const uint8_t* v,           \
               uint8_t* dst, int len) {                                        \
  int i;                                                                       \
  for (i = 0; i < len; ++i) FUNC(y[i], u[i], v[i], &dst[i * XSTEP]);           \
}

YUV444_FUNC(WebPYuv444ToRgbC,      VP8YuvToRgb,  3)
YUV444_FUNC(WebPYuv444ToBgrC,      VP8YuvToBgr,  3)
YUV444_FUNC(WebPYuv444ToRgbaC,     VP8YuvToRgba, 4)
YUV444_FUNC(WebPYuv444ToBgraC,     VP8YuvToBgra, 4)
YUV444_FUNC(WebPYuv444ToArgbC,     VP8YuvToArgb, 4)
YUV444_FUNC(WebPYuv444ToRgba4444C, VP8YuvToRgba4444, 2)
YUV444_FUNC(WebPYuv444ToRgb565C,   VP8YuvToRgb565, 2)

#undef YUV444_FUNC

WebPYUV444Converter WebPYUV444Converters[MODE_LAST];

extern void WebPInitYUV444ConvertersMIPSdspR2(void);
extern void WebPInitYUV444ConvertersSSE2(void);

static volatile VP8CPUInfo upsampling_last_cpuinfo_used1 =
    (VP8CPUInfo)&upsampling_last_cpuinfo_used1;

WEBP_TSAN_IGNORE_FUNCTION void WebPInitYUV444Converters(void) {
  if (upsampling_last_cpuinfo_used1 == VP8GetCPUInfo) return;

  WebPYUV444Converters[MODE_RGB]       = WebPYuv444ToRgbC;
  WebPYUV444Converters[MODE_RGBA]      = WebPYuv444ToRgbaC;
  WebPYUV444Converters[MODE_BGR]       = WebPYuv444ToBgrC;
  WebPYUV444Converters[MODE_BGRA]      = WebPYuv444ToBgraC;
  WebPYUV444Converters[MODE_ARGB]      = WebPYuv444ToArgbC;
  WebPYUV444Converters[MODE_RGBA_4444] = WebPYuv444ToRgba4444C;
  WebPYUV444Converters[MODE_RGB_565]   = WebPYuv444ToRgb565C;
  WebPYUV444Converters[MODE_rgbA]      = WebPYuv444ToRgbaC;
  WebPYUV444Converters[MODE_bgrA]      = WebPYuv444ToBgraC;
  WebPYUV444Converters[MODE_Argb]      = WebPYuv444ToArgbC;
  WebPYUV444Converters[MODE_rgbA_4444] = WebPYuv444ToRgba4444C;

  if (VP8GetCPUInfo != NULL) {
#if defined(WEBP_USE_SSE2)
    if (VP8GetCPUInfo(kSSE2)) {
      WebPInitYUV444ConvertersSSE2();
    }
#endif
#if defined(WEBP_USE_MIPS_DSP_R2)
    if (VP8GetCPUInfo(kMIPSdspR2)) {
      WebPInitYUV444ConvertersMIPSdspR2();
    }
#endif
  }
  upsampling_last_cpuinfo_used1 = VP8GetCPUInfo;
}

//------------------------------------------------------------------------------
// Main calls

extern void WebPInitUpsamplersSSE2(void);
extern void WebPInitUpsamplersNEON(void);
extern void WebPInitUpsamplersMIPSdspR2(void);
extern void WebPInitUpsamplersMSA(void);

static volatile VP8CPUInfo upsampling_last_cpuinfo_used2 =
    (VP8CPUInfo)&upsampling_last_cpuinfo_used2;

WEBP_TSAN_IGNORE_FUNCTION void WebPInitUpsamplers(void) {
  if (upsampling_last_cpuinfo_used2 == VP8GetCPUInfo) return;

#ifdef FANCY_UPSAMPLING
  WebPUpsamplers[MODE_RGB]       = UpsampleRgbLinePair;
  WebPUpsamplers[MODE_RGBA]      = UpsampleRgbaLinePair;
  WebPUpsamplers[MODE_BGR]       = UpsampleBgrLinePair;
  WebPUpsamplers[MODE_BGRA]      = UpsampleBgraLinePair;
  WebPUpsamplers[MODE_ARGB]      = UpsampleArgbLinePair;
  WebPUpsamplers[MODE_RGBA_4444] = UpsampleRgba4444LinePair;
  WebPUpsamplers[MODE_RGB_565]   = UpsampleRgb565LinePair;
  WebPUpsamplers[MODE_rgbA]      = UpsampleRgbaLinePair;
  WebPUpsamplers[MODE_bgrA]      = UpsampleBgraLinePair;
  WebPUpsamplers[MODE_Argb]      = UpsampleArgbLinePair;
  WebPUpsamplers[MODE_rgbA_4444] = UpsampleRgba4444LinePair;

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (VP8GetCPUInfo != NULL) {
#if defined(WEBP_USE_SSE2)
    if (VP8GetCPUInfo(kSSE2)) {
      WebPInitUpsamplersSSE2();
    }
#endif
#if defined(WEBP_USE_NEON)
    if (VP8GetCPUInfo(kNEON)) {
      WebPInitUpsamplersNEON();
    }
#endif
#if defined(WEBP_USE_MIPS_DSP_R2)
    if (VP8GetCPUInfo(kMIPSdspR2)) {
      WebPInitUpsamplersMIPSdspR2();
    }
#endif
#if defined(WEBP_USE_MSA)
    if (VP8GetCPUInfo(kMSA)) {
      WebPInitUpsamplersMSA();
    }
#endif
  }
#endif  // FANCY_UPSAMPLING
  upsampling_last_cpuinfo_used2 = VP8GetCPUInfo;
}

//------------------------------------------------------------------------------
