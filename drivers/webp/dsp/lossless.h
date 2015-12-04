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

#include "../enc/histogram.h"
#include "../utils/utils.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef WEBP_EXPERIMENTAL_FEATURES
#include "../enc/delta_palettization.h"
#endif  // WEBP_EXPERIMENTAL_FEATURES

// Not a trivial literal symbol.
#define VP8L_NON_TRIVIAL_SYM (0xffffffff)

//------------------------------------------------------------------------------
// Decoding

typedef uint32_t (*VP8LPredictorFunc)(uint32_t left, const uint32_t* const top);
extern VP8LPredictorFunc VP8LPredictors[16];

typedef void (*VP8LProcessBlueAndRedFunc)(uint32_t* argb_data, int num_pixels);
extern VP8LProcessBlueAndRedFunc VP8LAddGreenToBlueAndRed;

typedef struct {
  // Note: the members are uint8_t, so that any negative values are
  // automatically converted to "mod 256" values.
  uint8_t green_to_red_;
  uint8_t green_to_blue_;
  uint8_t red_to_blue_;
} VP8LMultipliers;
typedef void (*VP8LTransformColorFunc)(const VP8LMultipliers* const m,
                                       uint32_t* argb_data, int num_pixels);
extern VP8LTransformColorFunc VP8LTransformColorInverse;

struct VP8LTransform;  // Defined in dec/vp8li.h.

// Performs inverse transform of data given transform information, start and end
// rows. Transform will be applied to rows [row_start, row_end[.
// The *in and *out pointers refer to source and destination data respectively
// corresponding to the intermediate row (row_start).
void VP8LInverseTransform(const struct VP8LTransform* const transform,
                          int row_start, int row_end,
                          const uint32_t* const in, uint32_t* const out);

// Color space conversion.
typedef void (*VP8LConvertFunc)(const uint32_t* src, int num_pixels,
                                uint8_t* dst);
extern VP8LConvertFunc VP8LConvertBGRAToRGB;
extern VP8LConvertFunc VP8LConvertBGRAToRGBA;
extern VP8LConvertFunc VP8LConvertBGRAToRGBA4444;
extern VP8LConvertFunc VP8LConvertBGRAToRGB565;
extern VP8LConvertFunc VP8LConvertBGRAToBGR;

// Converts from BGRA to other color spaces.
void VP8LConvertFromBGRA(const uint32_t* const in_data, int num_pixels,
                         WEBP_CSP_MODE out_colorspace, uint8_t* const rgba);

// color mapping related functions.
static WEBP_INLINE uint32_t VP8GetARGBIndex(uint32_t idx) {
  return (idx >> 8) & 0xff;
}

static WEBP_INLINE uint8_t VP8GetAlphaIndex(uint8_t idx) {
  return idx;
}

static WEBP_INLINE uint32_t VP8GetARGBValue(uint32_t val) {
  return val;
}

static WEBP_INLINE uint8_t VP8GetAlphaValue(uint32_t val) {
  return (val >> 8) & 0xff;
}

typedef void (*VP8LMapARGBFunc)(const uint32_t* src,
                                const uint32_t* const color_map,
                                uint32_t* dst, int y_start,
                                int y_end, int width);
typedef void (*VP8LMapAlphaFunc)(const uint8_t* src,
                                 const uint32_t* const color_map,
                                 uint8_t* dst, int y_start,
                                 int y_end, int width);

extern VP8LMapARGBFunc VP8LMapColor32b;
extern VP8LMapAlphaFunc VP8LMapColor8b;

// Similar to the static method ColorIndexInverseTransform() that is part of
// lossless.c, but used only for alpha decoding. It takes uint8_t (rather than
// uint32_t) arguments for 'src' and 'dst'.
void VP8LColorIndexInverseTransformAlpha(
    const struct VP8LTransform* const transform, int y_start, int y_end,
    const uint8_t* src, uint8_t* dst);

// Expose some C-only fallback functions
void VP8LTransformColorInverse_C(const VP8LMultipliers* const m,
                                 uint32_t* data, int num_pixels);

void VP8LConvertBGRAToRGB_C(const uint32_t* src, int num_pixels, uint8_t* dst);
void VP8LConvertBGRAToRGBA_C(const uint32_t* src, int num_pixels, uint8_t* dst);
void VP8LConvertBGRAToRGBA4444_C(const uint32_t* src,
                                 int num_pixels, uint8_t* dst);
void VP8LConvertBGRAToRGB565_C(const uint32_t* src,
                               int num_pixels, uint8_t* dst);
void VP8LConvertBGRAToBGR_C(const uint32_t* src, int num_pixels, uint8_t* dst);
void VP8LAddGreenToBlueAndRed_C(uint32_t* data, int num_pixels);

// Must be called before calling any of the above methods.
void VP8LDspInit(void);

//------------------------------------------------------------------------------
// Encoding

extern VP8LProcessBlueAndRedFunc VP8LSubtractGreenFromBlueAndRed;
extern VP8LTransformColorFunc VP8LTransformColor;
typedef void (*VP8LCollectColorBlueTransformsFunc)(
    const uint32_t* argb, int stride,
    int tile_width, int tile_height,
    int green_to_blue, int red_to_blue, int histo[]);
extern VP8LCollectColorBlueTransformsFunc VP8LCollectColorBlueTransforms;

typedef void (*VP8LCollectColorRedTransformsFunc)(
    const uint32_t* argb, int stride,
    int tile_width, int tile_height,
    int green_to_red, int histo[]);
extern VP8LCollectColorRedTransformsFunc VP8LCollectColorRedTransforms;

// Expose some C-only fallback functions
void VP8LTransformColor_C(const VP8LMultipliers* const m,
                          uint32_t* data, int num_pixels);
void VP8LSubtractGreenFromBlueAndRed_C(uint32_t* argb_data, int num_pixels);
void VP8LCollectColorRedTransforms_C(const uint32_t* argb, int stride,
                                     int tile_width, int tile_height,
                                     int green_to_red, int histo[]);
void VP8LCollectColorBlueTransforms_C(const uint32_t* argb, int stride,
                                      int tile_width, int tile_height,
                                      int green_to_blue, int red_to_blue,
                                      int histo[]);

//------------------------------------------------------------------------------
// Image transforms.

void VP8LResidualImage(int width, int height, int bits, int low_effort,
                       uint32_t* const argb, uint32_t* const argb_scratch,
                       uint32_t* const image);

void VP8LColorSpaceTransform(int width, int height, int bits, int quality,
                             uint32_t* const argb, uint32_t* image);

//------------------------------------------------------------------------------
// Misc methods.

// Computes sampled size of 'size' when sampling using 'sampling bits'.
static WEBP_INLINE uint32_t VP8LSubSampleSize(uint32_t size,
                                              uint32_t sampling_bits) {
  return (size + (1 << sampling_bits) - 1) >> sampling_bits;
}

// -----------------------------------------------------------------------------
// Faster logarithm for integers. Small values use a look-up table.
#define LOG_LOOKUP_IDX_MAX 256
extern const float kLog2Table[LOG_LOOKUP_IDX_MAX];
extern const float kSLog2Table[LOG_LOOKUP_IDX_MAX];
typedef float (*VP8LFastLog2SlowFunc)(uint32_t v);

extern VP8LFastLog2SlowFunc VP8LFastLog2Slow;
extern VP8LFastLog2SlowFunc VP8LFastSLog2Slow;

static WEBP_INLINE float VP8LFastLog2(uint32_t v) {
  return (v < LOG_LOOKUP_IDX_MAX) ? kLog2Table[v] : VP8LFastLog2Slow(v);
}
// Fast calculation of v * log2(v) for integer input.
static WEBP_INLINE float VP8LFastSLog2(uint32_t v) {
  return (v < LOG_LOOKUP_IDX_MAX) ? kSLog2Table[v] : VP8LFastSLog2Slow(v);
}

// -----------------------------------------------------------------------------
// Huffman-cost related functions.

typedef double (*VP8LCostFunc)(const uint32_t* population, int length);
typedef double (*VP8LCostCombinedFunc)(const uint32_t* X, const uint32_t* Y,
                                       int length);

extern VP8LCostFunc VP8LExtraCost;
extern VP8LCostCombinedFunc VP8LExtraCostCombined;

typedef struct {        // small struct to hold counters
  int counts[2];        // index: 0=zero steak, 1=non-zero streak
  int streaks[2][2];    // [zero/non-zero][streak<3 / streak>=3]
} VP8LStreaks;

typedef VP8LStreaks (*VP8LCostCountFunc)(const uint32_t* population,
                                         int length);
typedef VP8LStreaks (*VP8LCostCombinedCountFunc)(const uint32_t* X,
                                                 const uint32_t* Y, int length);

extern VP8LCostCountFunc VP8LHuffmanCostCount;
extern VP8LCostCombinedCountFunc VP8LHuffmanCostCombinedCount;

// Get the symbol entropy for the distribution 'population'.
// Set 'trivial_sym', if there's only one symbol present in the distribution.
double VP8LPopulationCost(const uint32_t* const population, int length,
                          uint32_t* const trivial_sym);

// Get the combined symbol entropy for the distributions 'X' and 'Y'.
double VP8LGetCombinedEntropy(const uint32_t* const X,
                              const uint32_t* const Y, int length);

double VP8LBitsEntropy(const uint32_t* const array, int n,
                       uint32_t* const trivial_symbol);

// Estimate how many bits the combined entropy of literals and distance
// approximately maps to.
double VP8LHistogramEstimateBits(const VP8LHistogram* const p);

// This function estimates the cost in bits excluding the bits needed to
// represent the entropy code itself.
double VP8LHistogramEstimateBitsBulk(const VP8LHistogram* const p);

typedef void (*VP8LHistogramAddFunc)(const VP8LHistogram* const a,
                                     const VP8LHistogram* const b,
                                     VP8LHistogram* const out);
extern VP8LHistogramAddFunc VP8LHistogramAdd;

// -----------------------------------------------------------------------------
// PrefixEncode()

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

// Must be called before calling any of the above methods.
void VP8LEncDspInit(void);

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  // WEBP_DSP_LOSSLESS_H_
