// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Image transforms and color space conversion methods for lossless decoder.
//
// Authors: Vikas Arora (vikaas.arora@gmail.com)
//          Jyrki Alakuijala (jyrki@google.com)

#ifndef WEBP_DSP_LOSSLESS_H_
#define WEBP_DSP_LOSSLESS_H_

#include "../webp/types.h"
#include "../webp/decode.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
//

typedef uint32_t (*VP8LPredClampedAddSubFunc)(uint32_t c0, uint32_t c1,
                                              uint32_t c2);
typedef uint32_t (*VP8LPredSelectFunc)(uint32_t c0, uint32_t c1, uint32_t c2);
typedef void (*VP8LSubtractGreenFromBlueAndRedFunc)(uint32_t* argb_data,
                                                    int num_pixs);
typedef void (*VP8LAddGreenToBlueAndRedFunc)(uint32_t* data_start,
                                             const uint32_t* data_end);

extern VP8LPredClampedAddSubFunc VP8LClampedAddSubtractFull;
extern VP8LPredClampedAddSubFunc VP8LClampedAddSubtractHalf;
extern VP8LPredSelectFunc VP8LSelect;
extern VP8LSubtractGreenFromBlueAndRedFunc VP8LSubtractGreenFromBlueAndRed;
extern VP8LAddGreenToBlueAndRedFunc VP8LAddGreenToBlueAndRed;

// Must be called before calling any of the above methods.
void VP8LDspInit(void);

//------------------------------------------------------------------------------
// Image transforms.

struct VP8LTransform;  // Defined in dec/vp8li.h.

// Performs inverse transform of data given transform information, start and end
// rows. Transform will be applied to rows [row_start, row_end[.
// The *in and *out pointers refer to source and destination data respectively
// corresponding to the intermediate row (row_start).
void VP8LInverseTransform(const struct VP8LTransform* const transform,
                          int row_start, int row_end,
                          const uint32_t* const in, uint32_t* const out);

// Similar to the static method ColorIndexInverseTransform() that is part of
// lossless.c, but used only for alpha decoding. It takes uint8_t (rather than
// uint32_t) arguments for 'src' and 'dst'.
void VP8LColorIndexInverseTransformAlpha(
    const struct VP8LTransform* const transform, int y_start, int y_end,
    const uint8_t* src, uint8_t* dst);

void VP8LResidualImage(int width, int height, int bits,
                       uint32_t* const argb, uint32_t* const argb_scratch,
                       uint32_t* const image);

void VP8LColorSpaceTransform(int width, int height, int bits, int step,
                             uint32_t* const argb, uint32_t* image);

//------------------------------------------------------------------------------
// Color space conversion.

// Converts from BGRA to other color spaces.
void VP8LConvertFromBGRA(const uint32_t* const in_data, int num_pixels,
                         WEBP_CSP_MODE out_colorspace, uint8_t* const rgba);

//------------------------------------------------------------------------------
// Misc methods.

// Computes sampled size of 'size' when sampling using 'sampling bits'.
static WEBP_INLINE uint32_t VP8LSubSampleSize(uint32_t size,
                                              uint32_t sampling_bits) {
  return (size + (1 << sampling_bits) - 1) >> sampling_bits;
}

// Faster logarithm for integers. Small values use a look-up table.
#define LOG_LOOKUP_IDX_MAX 256
extern const float kLog2Table[LOG_LOOKUP_IDX_MAX];
extern const float kSLog2Table[LOG_LOOKUP_IDX_MAX];
float VP8LFastLog2Slow(int v);
float VP8LFastSLog2Slow(int v);
static WEBP_INLINE float VP8LFastLog2(int v) {
  return (v < LOG_LOOKUP_IDX_MAX) ? kLog2Table[v] : VP8LFastLog2Slow(v);
}
// Fast calculation of v * log2(v) for integer input.
static WEBP_INLINE float VP8LFastSLog2(int v) {
  return (v < LOG_LOOKUP_IDX_MAX) ? kSLog2Table[v] : VP8LFastSLog2Slow(v);
}

// -----------------------------------------------------------------------------
// PrefixEncode()

// use GNU builtins where available.
#if defined(__GNUC__) && \
    ((__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || __GNUC__ >= 4)
static WEBP_INLINE int BitsLog2Floor(uint32_t n) {
  return 31 ^ __builtin_clz(n);
}
#elif defined(_MSC_VER) && _MSC_VER > 1310 && \
      (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)

static WEBP_INLINE int BitsLog2Floor(uint32_t n) {
  unsigned long first_set_bit;
  _BitScanReverse(&first_set_bit, n);
  return first_set_bit;
}
#else
// Returns (int)floor(log2(n)). n must be > 0.
static WEBP_INLINE int BitsLog2Floor(uint32_t n) {
  int log = 0;
  uint32_t value = n;
  int i;

  for (i = 4; i >= 0; --i) {
    const int shift = (1 << i);
    const uint32_t x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  return log;
}
#endif

static WEBP_INLINE int VP8LBitsLog2Ceiling(uint32_t n) {
  const int log_floor = BitsLog2Floor(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two.
    return log_floor;
  else
    return log_floor + 1;
}

// Splitting of distance and length codes into prefixes and
// extra bits. The prefixes are encoded with an entropy code
// while the extra bits are stored just as normal bits.
static WEBP_INLINE void VP8LPrefixEncodeBitsNoLUT(int distance, int* const code,
                                                  int* const extra_bits) {
  const int highest_bit = BitsLog2Floor(--distance);
  const int second_highest_bit = (distance >> (highest_bit - 1)) & 1;
  *extra_bits = highest_bit - 1;
  *code = 2 * highest_bit + second_highest_bit;
}

static WEBP_INLINE void VP8LPrefixEncodeNoLUT(int distance, int* const code,
                                              int* const extra_bits,
                                              int* const extra_bits_value) {
  const int highest_bit = BitsLog2Floor(--distance);
  const int second_highest_bit = (distance >> (highest_bit - 1)) & 1;
  *extra_bits = highest_bit - 1;
  *extra_bits_value = distance & ((1 << *extra_bits) - 1);
  *code = 2 * highest_bit + second_highest_bit;
}

#define PREFIX_LOOKUP_IDX_MAX   512
typedef struct {
  int8_t code_;
  int8_t extra_bits_;
} VP8LPrefixCode;

// These tables are derived using VP8LPrefixEncodeNoLUT.
extern const VP8LPrefixCode kPrefixEncodeCode[PREFIX_LOOKUP_IDX_MAX];
extern const uint8_t kPrefixEncodeExtraBitsValue[PREFIX_LOOKUP_IDX_MAX];
static WEBP_INLINE void VP8LPrefixEncodeBits(int distance, int* const code,
                                             int* const extra_bits) {
  if (distance < PREFIX_LOOKUP_IDX_MAX) {
    const VP8LPrefixCode prefix_code = kPrefixEncodeCode[distance];
    *code = prefix_code.code_;
    *extra_bits = prefix_code.extra_bits_;
  } else {
    VP8LPrefixEncodeBitsNoLUT(distance, code, extra_bits);
  }
}

static WEBP_INLINE void VP8LPrefixEncode(int distance, int* const code,
                                         int* const extra_bits,
                                         int* const extra_bits_value) {
  if (distance < PREFIX_LOOKUP_IDX_MAX) {
    const VP8LPrefixCode prefix_code = kPrefixEncodeCode[distance];
    *code = prefix_code.code_;
    *extra_bits = prefix_code.extra_bits_;
    *extra_bits_value = kPrefixEncodeExtraBitsValue[distance];
  } else {
    VP8LPrefixEncodeNoLUT(distance, code, extra_bits, extra_bits_value);
  }
}

// In-place difference of each component with mod 256.
static WEBP_INLINE uint32_t VP8LSubPixels(uint32_t a, uint32_t b) {
  const uint32_t alpha_and_green =
      0x00ff00ffu + (a & 0xff00ff00u) - (b & 0xff00ff00u);
  const uint32_t red_and_blue =
      0xff00ff00u + (a & 0x00ff00ffu) - (b & 0x00ff00ffu);
  return (alpha_and_green & 0xff00ff00u) | (red_and_blue & 0x00ff00ffu);
}

void VP8LBundleColorMap(const uint8_t* const row, int width,
                        int xbits, uint32_t* const dst);

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  // WEBP_DSP_LOSSLESS_H_
