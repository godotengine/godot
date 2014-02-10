// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
//   Speed-critical functions.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WEBP_DSP_DSP_H_
#define WEBP_DSP_DSP_H_

#include "../webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// CPU detection

#if defined(_MSC_VER) && _MSC_VER > 1310 && \
    (defined(_M_X64) || defined(_M_IX86))
#define WEBP_MSC_SSE2  // Visual C++ SSE2 targets
#endif

#if defined(__SSE2__) || defined(WEBP_MSC_SSE2)
#define WEBP_USE_SSE2
#endif

#if defined(__ANDROID__) && defined(__ARM_ARCH_7A__)
#define WEBP_ANDROID_NEON  // Android targets that might support NEON
#endif

#if defined(__ARM_NEON__) || defined(WEBP_ANDROID_NEON)
#define WEBP_USE_NEON
#endif

typedef enum {
  kSSE2,
  kSSE3,
  kNEON
} CPUFeature;
// returns true if the CPU supports the feature.
typedef int (*VP8CPUInfo)(CPUFeature feature);
extern VP8CPUInfo VP8GetCPUInfo;

//------------------------------------------------------------------------------
// Encoding

// Transforms
// VP8Idct: Does one of two inverse transforms. If do_two is set, the transforms
//          will be done for (ref, in, dst) and (ref + 4, in + 16, dst + 4).
typedef void (*VP8Idct)(const uint8_t* ref, const int16_t* in, uint8_t* dst,
                        int do_two);
typedef void (*VP8Fdct)(const uint8_t* src, const uint8_t* ref, int16_t* out);
typedef void (*VP8WHT)(const int16_t* in, int16_t* out);
extern VP8Idct VP8ITransform;
extern VP8Fdct VP8FTransform;
extern VP8WHT VP8ITransformWHT;
extern VP8WHT VP8FTransformWHT;
// Predictions
// *dst is the destination block. *top and *left can be NULL.
typedef void (*VP8IntraPreds)(uint8_t *dst, const uint8_t* left,
                              const uint8_t* top);
typedef void (*VP8Intra4Preds)(uint8_t *dst, const uint8_t* top);
extern VP8Intra4Preds VP8EncPredLuma4;
extern VP8IntraPreds VP8EncPredLuma16;
extern VP8IntraPreds VP8EncPredChroma8;

typedef int (*VP8Metric)(const uint8_t* pix, const uint8_t* ref);
extern VP8Metric VP8SSE16x16, VP8SSE16x8, VP8SSE8x8, VP8SSE4x4;
typedef int (*VP8WMetric)(const uint8_t* pix, const uint8_t* ref,
                          const uint16_t* const weights);
extern VP8WMetric VP8TDisto4x4, VP8TDisto16x16;

typedef void (*VP8BlockCopy)(const uint8_t* src, uint8_t* dst);
extern VP8BlockCopy VP8Copy4x4;
// Quantization
struct VP8Matrix;   // forward declaration
typedef int (*VP8QuantizeBlock)(int16_t in[16], int16_t out[16],
                                int n, const struct VP8Matrix* const mtx);
extern VP8QuantizeBlock VP8EncQuantizeBlock;

// specific to 2nd transform:
typedef int (*VP8QuantizeBlockWHT)(int16_t in[16], int16_t out[16],
                                   const struct VP8Matrix* const mtx);
extern VP8QuantizeBlockWHT VP8EncQuantizeBlockWHT;

// Collect histogram for susceptibility calculation and accumulate in histo[].
struct VP8Histogram;
typedef void (*VP8CHisto)(const uint8_t* ref, const uint8_t* pred,
                          int start_block, int end_block,
                          struct VP8Histogram* const histo);
extern const int VP8DspScan[16 + 4 + 4];
extern VP8CHisto VP8CollectHistogram;

void VP8EncDspInit(void);   // must be called before using any of the above

//------------------------------------------------------------------------------
// Decoding

typedef void (*VP8DecIdct)(const int16_t* coeffs, uint8_t* dst);
// when doing two transforms, coeffs is actually int16_t[2][16].
typedef void (*VP8DecIdct2)(const int16_t* coeffs, uint8_t* dst, int do_two);
extern VP8DecIdct2 VP8Transform;
extern VP8DecIdct VP8TransformAC3;
extern VP8DecIdct VP8TransformUV;
extern VP8DecIdct VP8TransformDC;
extern VP8DecIdct VP8TransformDCUV;
extern VP8WHT VP8TransformWHT;

// *dst is the destination block, with stride BPS. Boundary samples are
// assumed accessible when needed.
typedef void (*VP8PredFunc)(uint8_t* dst);
extern const VP8PredFunc VP8PredLuma16[/* NUM_B_DC_MODES */];
extern const VP8PredFunc VP8PredChroma8[/* NUM_B_DC_MODES */];
extern const VP8PredFunc VP8PredLuma4[/* NUM_BMODES */];

// simple filter (only for luma)
typedef void (*VP8SimpleFilterFunc)(uint8_t* p, int stride, int thresh);
extern VP8SimpleFilterFunc VP8SimpleVFilter16;
extern VP8SimpleFilterFunc VP8SimpleHFilter16;
extern VP8SimpleFilterFunc VP8SimpleVFilter16i;  // filter 3 inner edges
extern VP8SimpleFilterFunc VP8SimpleHFilter16i;

// regular filter (on both macroblock edges and inner edges)
typedef void (*VP8LumaFilterFunc)(uint8_t* luma, int stride,
                                  int thresh, int ithresh, int hev_t);
typedef void (*VP8ChromaFilterFunc)(uint8_t* u, uint8_t* v, int stride,
                                    int thresh, int ithresh, int hev_t);
// on outer edge
extern VP8LumaFilterFunc VP8VFilter16;
extern VP8LumaFilterFunc VP8HFilter16;
extern VP8ChromaFilterFunc VP8VFilter8;
extern VP8ChromaFilterFunc VP8HFilter8;

// on inner edge
extern VP8LumaFilterFunc VP8VFilter16i;   // filtering 3 inner edges altogether
extern VP8LumaFilterFunc VP8HFilter16i;
extern VP8ChromaFilterFunc VP8VFilter8i;  // filtering u and v altogether
extern VP8ChromaFilterFunc VP8HFilter8i;

// must be called before anything using the above
void VP8DspInit(void);

//------------------------------------------------------------------------------
// WebP I/O

#define FANCY_UPSAMPLING   // undefined to remove fancy upsampling support

// Convert a pair of y/u/v lines together to the output rgb/a colorspace.
// bottom_y can be NULL if only one line of output is needed (at top/bottom).
typedef void (*WebPUpsampleLinePairFunc)(
    const uint8_t* top_y, const uint8_t* bottom_y,
    const uint8_t* top_u, const uint8_t* top_v,
    const uint8_t* cur_u, const uint8_t* cur_v,
    uint8_t* top_dst, uint8_t* bottom_dst, int len);

#ifdef FANCY_UPSAMPLING

// Fancy upsampling functions to convert YUV to RGB(A) modes
extern WebPUpsampleLinePairFunc WebPUpsamplers[/* MODE_LAST */];

// Initializes SSE2 version of the fancy upsamplers.
void WebPInitUpsamplersSSE2(void);

// NEON version
void WebPInitUpsamplersNEON(void);

#endif    // FANCY_UPSAMPLING

// Point-sampling methods.
typedef void (*WebPSampleLinePairFunc)(
    const uint8_t* top_y, const uint8_t* bottom_y,
    const uint8_t* u, const uint8_t* v,
    uint8_t* top_dst, uint8_t* bottom_dst, int len);

extern const WebPSampleLinePairFunc WebPSamplers[/* MODE_LAST */];

// General function for converting two lines of ARGB or RGBA.
// 'alpha_is_last' should be true if 0xff000000 is stored in memory as
// as 0x00, 0x00, 0x00, 0xff (little endian).
WebPUpsampleLinePairFunc WebPGetLinePairConverter(int alpha_is_last);

// YUV444->RGB converters
typedef void (*WebPYUV444Converter)(const uint8_t* y,
                                    const uint8_t* u, const uint8_t* v,
                                    uint8_t* dst, int len);

extern const WebPYUV444Converter WebPYUV444Converters[/* MODE_LAST */];

// Main function to be called
void WebPInitUpsamplers(void);

//------------------------------------------------------------------------------
// Pre-multiply planes with alpha values

// Apply alpha pre-multiply on an rgba, bgra or argb plane of size w * h.
// alpha_first should be 0 for argb, 1 for rgba or bgra (where alpha is last).
extern void (*WebPApplyAlphaMultiply)(
    uint8_t* rgba, int alpha_first, int w, int h, int stride);

// Same, buf specifically for RGBA4444 format
extern void (*WebPApplyAlphaMultiply4444)(
    uint8_t* rgba4444, int w, int h, int stride);

// To be called first before using the above.
void WebPInitPremultiply(void);

void WebPInitPremultiplySSE2(void);   // should not be called directly.
void WebPInitPremultiplyNEON(void);

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_DSP_DSP_H_ */
