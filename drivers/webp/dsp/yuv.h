// Copyright 2010 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// inline YUV<->RGB conversion function
//
// The exact naming is Y'CbCr, following the ITU-R BT.601 standard.
// More information at: http://en.wikipedia.org/wiki/YCbCr
// Y = 0.2569 * R + 0.5044 * G + 0.0979 * B + 16
// U = -0.1483 * R - 0.2911 * G + 0.4394 * B + 128
// V = 0.4394 * R - 0.3679 * G - 0.0715 * B + 128
// We use 16bit fixed point operations for RGB->YUV conversion (YUV_FIX).
//
// For the Y'CbCr to RGB conversion, the BT.601 specification reads:
//   R = 1.164 * (Y-16) + 1.596 * (V-128)
//   G = 1.164 * (Y-16) - 0.813 * (V-128) - 0.391 * (U-128)
//   B = 1.164 * (Y-16)                   + 2.018 * (U-128)
// where Y is in the [16,235] range, and U/V in the [16,240] range.
// In the table-lookup version (WEBP_YUV_USE_TABLE), the common factor
// "1.164 * (Y-16)" can be handled as an offset in the VP8kClip[] table.
// So in this case the formulae should read:
//   R = 1.164 * [Y + 1.371 * (V-128)                  ] - 18.624
//   G = 1.164 * [Y - 0.698 * (V-128) - 0.336 * (U-128)] - 18.624
//   B = 1.164 * [Y                   + 1.733 * (U-128)] - 18.624
// once factorized.
// For YUV->RGB conversion, only 14bit fixed precision is used (YUV_FIX2).
// That's the maximum possible for a convenient ARM implementation.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WEBP_DSP_YUV_H_
#define WEBP_DSP_YUV_H_

#include "./dsp.h"
#include "../dec/decode_vp8.h"

// Define the following to use the LUT-based code:
// #define WEBP_YUV_USE_TABLE

#if defined(WEBP_EXPERIMENTAL_FEATURES)
// Do NOT activate this feature for real compression. This is only experimental!
// This flag is for comparison purpose against JPEG's "YUVj" natural colorspace.
// This colorspace is close to Rec.601's Y'CbCr model with the notable
// difference of allowing larger range for luma/chroma.
// See http://en.wikipedia.org/wiki/YCbCr#JPEG_conversion paragraph, and its
// difference with http://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
// #define USE_YUVj
#endif

//------------------------------------------------------------------------------
// YUV -> RGB conversion

#ifdef __cplusplus
extern "C" {
#endif

enum {
  YUV_FIX = 16,                    // fixed-point precision for RGB->YUV
  YUV_HALF = 1 << (YUV_FIX - 1),
  YUV_MASK = (256 << YUV_FIX) - 1,
  YUV_RANGE_MIN = -227,            // min value of r/g/b output
  YUV_RANGE_MAX = 256 + 226,       // max value of r/g/b output

  YUV_FIX2 = 14,                   // fixed-point precision for YUV->RGB
  YUV_HALF2 = 1 << (YUV_FIX2 - 1),
  YUV_MASK2 = (256 << YUV_FIX2) - 1
};

// These constants are 14b fixed-point version of ITU-R BT.601 constants.
#define kYScale 19077    // 1.164 = 255 / 219
#define kVToR   26149    // 1.596 = 255 / 112 * 0.701
#define kUToG   6419     // 0.391 = 255 / 112 * 0.886 * 0.114 / 0.587
#define kVToG   13320    // 0.813 = 255 / 112 * 0.701 * 0.299 / 0.587
#define kUToB   33050    // 2.018 = 255 / 112 * 0.886
#define kRCst (-kYScale * 16 - kVToR * 128 + YUV_HALF2)
#define kGCst (-kYScale * 16 + kUToG * 128 + kVToG * 128 + YUV_HALF2)
#define kBCst (-kYScale * 16 - kUToB * 128 + YUV_HALF2)

//------------------------------------------------------------------------------

#if !defined(WEBP_YUV_USE_TABLE)

// slower on x86 by ~7-8%, but bit-exact with the SSE2 version

static WEBP_INLINE int VP8Clip8(int v) {
  return ((v & ~YUV_MASK2) == 0) ? (v >> YUV_FIX2) : (v < 0) ? 0 : 255;
}

static WEBP_INLINE int VP8YUVToR(int y, int v) {
  return VP8Clip8(kYScale * y + kVToR * v + kRCst);
}

static WEBP_INLINE int VP8YUVToG(int y, int u, int v) {
  return VP8Clip8(kYScale * y - kUToG * u - kVToG * v + kGCst);
}

static WEBP_INLINE int VP8YUVToB(int y, int u) {
  return VP8Clip8(kYScale * y + kUToB * u + kBCst);
}

static WEBP_INLINE void VP8YuvToRgb(int y, int u, int v,
                                    uint8_t* const rgb) {
  rgb[0] = VP8YUVToR(y, v);
  rgb[1] = VP8YUVToG(y, u, v);
  rgb[2] = VP8YUVToB(y, u);
}

static WEBP_INLINE void VP8YuvToBgr(int y, int u, int v,
                                    uint8_t* const bgr) {
  bgr[0] = VP8YUVToB(y, u);
  bgr[1] = VP8YUVToG(y, u, v);
  bgr[2] = VP8YUVToR(y, v);
}

static WEBP_INLINE void VP8YuvToRgb565(int y, int u, int v,
                                       uint8_t* const rgb) {
  const int r = VP8YUVToR(y, v);      // 5 usable bits
  const int g = VP8YUVToG(y, u, v);   // 6 usable bits
  const int b = VP8YUVToB(y, u);      // 5 usable bits
  const int rg = (r & 0xf8) | (g >> 5);
  const int gb = ((g << 3) & 0xe0) | (b >> 3);
#ifdef WEBP_SWAP_16BIT_CSP
  rgb[0] = gb;
  rgb[1] = rg;
#else
  rgb[0] = rg;
  rgb[1] = gb;
#endif
}

static WEBP_INLINE void VP8YuvToRgba4444(int y, int u, int v,
                                         uint8_t* const argb) {
  const int r = VP8YUVToR(y, v);        // 4 usable bits
  const int g = VP8YUVToG(y, u, v);     // 4 usable bits
  const int b = VP8YUVToB(y, u);        // 4 usable bits
  const int rg = (r & 0xf0) | (g >> 4);
  const int ba = (b & 0xf0) | 0x0f;     // overwrite the lower 4 bits
#ifdef WEBP_SWAP_16BIT_CSP
  argb[0] = ba;
  argb[1] = rg;
#else
  argb[0] = rg;
  argb[1] = ba;
#endif
}

#else

// Table-based version, not totally equivalent to the SSE2 version.
// Rounding diff is only +/-1 though.

extern int16_t VP8kVToR[256], VP8kUToB[256];
extern int32_t VP8kVToG[256], VP8kUToG[256];
extern uint8_t VP8kClip[YUV_RANGE_MAX - YUV_RANGE_MIN];
extern uint8_t VP8kClip4Bits[YUV_RANGE_MAX - YUV_RANGE_MIN];

static WEBP_INLINE void VP8YuvToRgb(int y, int u, int v,
                                    uint8_t* const rgb) {
  const int r_off = VP8kVToR[v];
  const int g_off = (VP8kVToG[v] + VP8kUToG[u]) >> YUV_FIX;
  const int b_off = VP8kUToB[u];
  rgb[0] = VP8kClip[y + r_off - YUV_RANGE_MIN];
  rgb[1] = VP8kClip[y + g_off - YUV_RANGE_MIN];
  rgb[2] = VP8kClip[y + b_off - YUV_RANGE_MIN];
}

static WEBP_INLINE void VP8YuvToBgr(int y, int u, int v,
                                    uint8_t* const bgr) {
  const int r_off = VP8kVToR[v];
  const int g_off = (VP8kVToG[v] + VP8kUToG[u]) >> YUV_FIX;
  const int b_off = VP8kUToB[u];
  bgr[0] = VP8kClip[y + b_off - YUV_RANGE_MIN];
  bgr[1] = VP8kClip[y + g_off - YUV_RANGE_MIN];
  bgr[2] = VP8kClip[y + r_off - YUV_RANGE_MIN];
}

static WEBP_INLINE void VP8YuvToRgb565(int y, int u, int v,
                                       uint8_t* const rgb) {
  const int r_off = VP8kVToR[v];
  const int g_off = (VP8kVToG[v] + VP8kUToG[u]) >> YUV_FIX;
  const int b_off = VP8kUToB[u];
  const int rg = ((VP8kClip[y + r_off - YUV_RANGE_MIN] & 0xf8) |
                  (VP8kClip[y + g_off - YUV_RANGE_MIN] >> 5));
  const int gb = (((VP8kClip[y + g_off - YUV_RANGE_MIN] << 3) & 0xe0) |
                   (VP8kClip[y + b_off - YUV_RANGE_MIN] >> 3));
#ifdef WEBP_SWAP_16BIT_CSP
  rgb[0] = gb;
  rgb[1] = rg;
#else
  rgb[0] = rg;
  rgb[1] = gb;
#endif
}

static WEBP_INLINE void VP8YuvToRgba4444(int y, int u, int v,
                                         uint8_t* const argb) {
  const int r_off = VP8kVToR[v];
  const int g_off = (VP8kVToG[v] + VP8kUToG[u]) >> YUV_FIX;
  const int b_off = VP8kUToB[u];
  const int rg = ((VP8kClip4Bits[y + r_off - YUV_RANGE_MIN] << 4) |
                   VP8kClip4Bits[y + g_off - YUV_RANGE_MIN]);
  const int ba = (VP8kClip4Bits[y + b_off - YUV_RANGE_MIN] << 4) | 0x0f;
#ifdef WEBP_SWAP_16BIT_CSP
  argb[0] = ba;
  argb[1] = rg;
#else
  argb[0] = rg;
  argb[1] = ba;
#endif
}

#endif  // WEBP_YUV_USE_TABLE

//-----------------------------------------------------------------------------
// Alpha handling variants

static WEBP_INLINE void VP8YuvToArgb(uint8_t y, uint8_t u, uint8_t v,
                                     uint8_t* const argb) {
  argb[0] = 0xff;
  VP8YuvToRgb(y, u, v, argb + 1);
}

static WEBP_INLINE void VP8YuvToBgra(uint8_t y, uint8_t u, uint8_t v,
                                     uint8_t* const bgra) {
  VP8YuvToBgr(y, u, v, bgra);
  bgra[3] = 0xff;
}

static WEBP_INLINE void VP8YuvToRgba(uint8_t y, uint8_t u, uint8_t v,
                                     uint8_t* const rgba) {
  VP8YuvToRgb(y, u, v, rgba);
  rgba[3] = 0xff;
}

// Must be called before everything, to initialize the tables.
void VP8YUVInit(void);

//-----------------------------------------------------------------------------
// SSE2 extra functions (mostly for upsampling_sse2.c)

#if defined(WEBP_USE_SSE2)

#if defined(FANCY_UPSAMPLING)
// Process 32 pixels and store the result (24b or 32b per pixel) in *dst.
void VP8YuvToRgba32(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                    uint8_t* dst);
void VP8YuvToRgb32(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                   uint8_t* dst);
void VP8YuvToBgra32(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                    uint8_t* dst);
void VP8YuvToBgr32(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                   uint8_t* dst);
#endif  // FANCY_UPSAMPLING

// Must be called to initialize tables before using the functions.
void VP8YUVInitSSE2(void);

#endif    // WEBP_USE_SSE2

//------------------------------------------------------------------------------
// RGB -> YUV conversion

// Stub functions that can be called with various rounding values:
static WEBP_INLINE int VP8ClipUV(int uv, int rounding) {
  uv = (uv + rounding + (128 << (YUV_FIX + 2))) >> (YUV_FIX + 2);
  return ((uv & ~0xff) == 0) ? uv : (uv < 0) ? 0 : 255;
}

#ifndef USE_YUVj

static WEBP_INLINE int VP8RGBToY(int r, int g, int b, int rounding) {
  const int luma = 16839 * r + 33059 * g + 6420 * b;
  return (luma + rounding + (16 << YUV_FIX)) >> YUV_FIX;  // no need to clip
}

static WEBP_INLINE int VP8RGBToU(int r, int g, int b, int rounding) {
  const int u = -9719 * r - 19081 * g + 28800 * b;
  return VP8ClipUV(u, rounding);
}

static WEBP_INLINE int VP8RGBToV(int r, int g, int b, int rounding) {
  const int v = +28800 * r - 24116 * g - 4684 * b;
  return VP8ClipUV(v, rounding);
}

#else

// This JPEG-YUV colorspace, only for comparison!
// These are also 16bit precision coefficients from Rec.601, but with full
// [0..255] output range.
static WEBP_INLINE int VP8RGBToY(int r, int g, int b, int rounding) {
  const int luma = 19595 * r + 38470 * g + 7471 * b;
  return (luma + rounding) >> YUV_FIX;  // no need to clip
}

static WEBP_INLINE int VP8_RGB_TO_U(int r, int g, int b, int rounding) {
  const int u = -11058 * r - 21710 * g + 32768 * b;
  return VP8ClipUV(u, rounding);
}

static WEBP_INLINE int VP8_RGB_TO_V(int r, int g, int b, int rounding) {
  const int v = 32768 * r - 27439 * g - 5329 * b;
  return VP8ClipUV(v, rounding);
}

#endif    // USE_YUVj

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_DSP_YUV_H_ */
