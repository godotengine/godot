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

#ifdef HAVE_CONFIG_H
#include "../webp/config.h"
#endif

#include "../webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BPS 32   // this is the common stride for enc/dec

//------------------------------------------------------------------------------
// CPU detection

#if defined(__GNUC__)
# define LOCAL_GCC_VERSION ((__GNUC__ << 8) | __GNUC_MINOR__)
# define LOCAL_GCC_PREREQ(maj, min) \
    (LOCAL_GCC_VERSION >= (((maj) << 8) | (min)))
#else
# define LOCAL_GCC_VERSION 0
# define LOCAL_GCC_PREREQ(maj, min) 0
#endif

#ifndef __has_builtin
# define __has_builtin(x) 0
#endif

#if defined(_MSC_VER) && _MSC_VER > 1310 && \
    (defined(_M_X64) || defined(_M_IX86))
#define WEBP_MSC_SSE2  // Visual C++ SSE2 targets
#endif

#if defined(_MSC_VER) && _MSC_VER >= 1500 && \
    (defined(_M_X64) || defined(_M_IX86))
#define WEBP_MSC_SSE41  // Visual C++ SSE4.1 targets
#endif

// WEBP_HAVE_* are used to indicate the presence of the instruction set in dsp
// files without intrinsics, allowing the corresponding Init() to be called.
// Files containing intrinsics will need to be built targeting the instruction
// set so should succeed on one of the earlier tests.
#if defined(__SSE2__) || defined(WEBP_MSC_SSE2) || defined(WEBP_HAVE_SSE2)
#define WEBP_USE_SSE2
#endif

#if defined(__SSE4_1__) || defined(WEBP_MSC_SSE41) || defined(WEBP_HAVE_SSE41)
#define WEBP_USE_SSE41
#endif

#if defined(__AVX2__) || defined(WEBP_HAVE_AVX2)
#define WEBP_USE_AVX2
#endif

#if defined(__ANDROID__) && defined(__ARM_ARCH_7A__)
#define WEBP_ANDROID_NEON  // Android targets that might support NEON
#endif

// The intrinsics currently cause compiler errors with arm-nacl-gcc and the
// inline assembly would need to be modified for use with Native Client.
#if (defined(__ARM_NEON__) || defined(WEBP_ANDROID_NEON) || \
     defined(__aarch64__) || defined(WEBP_HAVE_NEON)) && \
    !defined(__native_client__)
#define WEBP_USE_NEON
#endif

#if defined(_MSC_VER) && _MSC_VER >= 1700 && defined(_M_ARM)
#define WEBP_USE_NEON
#define WEBP_USE_INTRINSICS
#endif

#if defined(__mips__) && !defined(__mips64) && \
    defined(__mips_isa_rev) && (__mips_isa_rev >= 1) && (__mips_isa_rev < 6)
#define WEBP_USE_MIPS32
#if (__mips_isa_rev >= 2)
#define WEBP_USE_MIPS32_R2
#if defined(__mips_dspr2) || (__mips_dsp_rev >= 2)
#define WEBP_USE_MIPS_DSP_R2
#endif
#endif
#endif

#if defined(__mips_msa) && defined(__mips_isa_rev) && (__mips_isa_rev >= 5)
#define WEBP_USE_MSA
#endif

// This macro prevents thread_sanitizer from reporting known concurrent writes.
#define WEBP_TSAN_IGNORE_FUNCTION
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#undef WEBP_TSAN_IGNORE_FUNCTION
#define WEBP_TSAN_IGNORE_FUNCTION __attribute__((no_sanitize_thread))
#endif
#endif

#define WEBP_UBSAN_IGNORE_UNDEF
#define WEBP_UBSAN_IGNORE_UNSIGNED_OVERFLOW
#if !defined(WEBP_FORCE_ALIGNED) && defined(__clang__) && \
    defined(__has_attribute)
#if __has_attribute(no_sanitize)
// This macro prevents the undefined behavior sanitizer from reporting
// failures. This is only meant to silence unaligned loads on platforms that
// are known to support them.
#undef WEBP_UBSAN_IGNORE_UNDEF
#define WEBP_UBSAN_IGNORE_UNDEF \
  __attribute__((no_sanitize("undefined")))

// This macro prevents the undefined behavior sanitizer from reporting
// failures related to unsigned integer overflows. This is only meant to
// silence cases where this well defined behavior is expected.
#undef WEBP_UBSAN_IGNORE_UNSIGNED_OVERFLOW
#define WEBP_UBSAN_IGNORE_UNSIGNED_OVERFLOW \
  __attribute__((no_sanitize("unsigned-integer-overflow")))
#endif
#endif

typedef enum {
  kSSE2,
  kSSE3,
  kSSE4_1,
  kAVX,
  kAVX2,
  kNEON,
  kMIPS32,
  kMIPSdspR2,
  kMSA
} CPUFeature;
// returns true if the CPU supports the feature.
typedef int (*VP8CPUInfo)(CPUFeature feature);
WEBP_EXTERN(VP8CPUInfo) VP8GetCPUInfo;

//------------------------------------------------------------------------------
// Init stub generator

// Defines an init function stub to ensure each module exposes a symbol,
// avoiding a compiler warning.
#define WEBP_DSP_INIT_STUB(func) \
  extern void func(void); \
  WEBP_TSAN_IGNORE_FUNCTION void func(void) {}

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
extern VP8Fdct VP8FTransform2;   // performs two transforms at a time
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
// The weights for VP8TDisto4x4 and VP8TDisto16x16 contain a row-major
// 4 by 4 symmetric matrix.
extern VP8WMetric VP8TDisto4x4, VP8TDisto16x16;

typedef void (*VP8BlockCopy)(const uint8_t* src, uint8_t* dst);
extern VP8BlockCopy VP8Copy4x4;
extern VP8BlockCopy VP8Copy16x8;
// Quantization
struct VP8Matrix;   // forward declaration
typedef int (*VP8QuantizeBlock)(int16_t in[16], int16_t out[16],
                                const struct VP8Matrix* const mtx);
// Same as VP8QuantizeBlock, but quantizes two consecutive blocks.
typedef int (*VP8Quantize2Blocks)(int16_t in[32], int16_t out[32],
                                  const struct VP8Matrix* const mtx);

extern VP8QuantizeBlock VP8EncQuantizeBlock;
extern VP8Quantize2Blocks VP8EncQuantize2Blocks;

// specific to 2nd transform:
typedef int (*VP8QuantizeBlockWHT)(int16_t in[16], int16_t out[16],
                                   const struct VP8Matrix* const mtx);
extern VP8QuantizeBlockWHT VP8EncQuantizeBlockWHT;

extern const int VP8DspScan[16 + 4 + 4];

// Collect histogram for susceptibility calculation.
#define MAX_COEFF_THRESH   31   // size of histogram used by CollectHistogram.
typedef struct {
  // We only need to store max_value and last_non_zero, not the distribution.
  int max_value;
  int last_non_zero;
} VP8Histogram;
typedef void (*VP8CHisto)(const uint8_t* ref, const uint8_t* pred,
                          int start_block, int end_block,
                          VP8Histogram* const histo);
extern VP8CHisto VP8CollectHistogram;
// General-purpose util function to help VP8CollectHistogram().
void VP8SetHistogramData(const int distribution[MAX_COEFF_THRESH + 1],
                         VP8Histogram* const histo);

// must be called before using any of the above
void VP8EncDspInit(void);

//------------------------------------------------------------------------------
// cost functions (encoding)

extern const uint16_t VP8EntropyCost[256];        // 8bit fixed-point log(p)
// approximate cost per level:
extern const uint16_t VP8LevelFixedCosts[2047 /*MAX_LEVEL*/ + 1];
extern const uint8_t VP8EncBands[16 + 1];

struct VP8Residual;
typedef void (*VP8SetResidualCoeffsFunc)(const int16_t* const coeffs,
                                         struct VP8Residual* const res);
extern VP8SetResidualCoeffsFunc VP8SetResidualCoeffs;

// Cost calculation function.
typedef int (*VP8GetResidualCostFunc)(int ctx0,
                                      const struct VP8Residual* const res);
extern VP8GetResidualCostFunc VP8GetResidualCost;

// must be called before anything using the above
void VP8EncDspCostInit(void);

//------------------------------------------------------------------------------
// SSIM utils

// struct for accumulating statistical moments
typedef struct {
  double w;              // sum(w_i) : sum of weights
  double xm, ym;         // sum(w_i * x_i), sum(w_i * y_i)
  double xxm, xym, yym;  // sum(w_i * x_i * x_i), etc.
} VP8DistoStats;

#define VP8_SSIM_KERNEL 3   // total size of the kernel: 2 * VP8_SSIM_KERNEL + 1
typedef void (*VP8SSIMAccumulateClippedFunc)(const uint8_t* src1, int stride1,
                                             const uint8_t* src2, int stride2,
                                             int xo, int yo,  // center position
                                             int W, int H,    // plane dimension
                                             VP8DistoStats* const stats);

// This version is called with the guarantee that you can load 8 bytes and
// 8 rows at offset src1 and src2
typedef void (*VP8SSIMAccumulateFunc)(const uint8_t* src1, int stride1,
                                      const uint8_t* src2, int stride2,
                                      VP8DistoStats* const stats);

extern VP8SSIMAccumulateFunc VP8SSIMAccumulate;         // unclipped / unchecked
extern VP8SSIMAccumulateClippedFunc VP8SSIMAccumulateClipped;   // with clipping

// must be called before using any of the above directly
void VP8SSIMDspInit(void);

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
extern VP8PredFunc VP8PredLuma16[/* NUM_B_DC_MODES */];
extern VP8PredFunc VP8PredChroma8[/* NUM_B_DC_MODES */];
extern VP8PredFunc VP8PredLuma4[/* NUM_BMODES */];

// clipping tables (for filtering)
extern const int8_t* const VP8ksclip1;  // clips [-1020, 1020] to [-128, 127]
extern const int8_t* const VP8ksclip2;  // clips [-112, 112] to [-16, 15]
extern const uint8_t* const VP8kclip1;  // clips [-255,511] to [0,255]
extern const uint8_t* const VP8kabs0;   // abs(x) for x in [-255,255]
// must be called first
void VP8InitClipTables(void);

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

// Dithering. Combines dithering values (centered around 128) with dst[],
// according to: dst[] = clip(dst[] + (((dither[]-128) + 8) >> 4)
#define VP8_DITHER_DESCALE 4
#define VP8_DITHER_DESCALE_ROUNDER (1 << (VP8_DITHER_DESCALE - 1))
#define VP8_DITHER_AMP_BITS 7
#define VP8_DITHER_AMP_CENTER (1 << VP8_DITHER_AMP_BITS)
extern void (*VP8DitherCombine8x8)(const uint8_t* dither, uint8_t* dst,
                                   int dst_stride);

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

#endif    // FANCY_UPSAMPLING

// Per-row point-sampling methods.
typedef void (*WebPSamplerRowFunc)(const uint8_t* y,
                                   const uint8_t* u, const uint8_t* v,
                                   uint8_t* dst, int len);
// Generic function to apply 'WebPSamplerRowFunc' to the whole plane:
void WebPSamplerProcessPlane(const uint8_t* y, int y_stride,
                             const uint8_t* u, const uint8_t* v, int uv_stride,
                             uint8_t* dst, int dst_stride,
                             int width, int height, WebPSamplerRowFunc func);

// Sampling functions to convert rows of YUV to RGB(A)
extern WebPSamplerRowFunc WebPSamplers[/* MODE_LAST */];

// General function for converting two lines of ARGB or RGBA.
// 'alpha_is_last' should be true if 0xff000000 is stored in memory as
// as 0x00, 0x00, 0x00, 0xff (little endian).
WebPUpsampleLinePairFunc WebPGetLinePairConverter(int alpha_is_last);

// YUV444->RGB converters
typedef void (*WebPYUV444Converter)(const uint8_t* y,
                                    const uint8_t* u, const uint8_t* v,
                                    uint8_t* dst, int len);

extern WebPYUV444Converter WebPYUV444Converters[/* MODE_LAST */];

// Must be called before using the WebPUpsamplers[] (and for premultiplied
// colorspaces like rgbA, rgbA4444, etc)
void WebPInitUpsamplers(void);
// Must be called before using WebPSamplers[]
void WebPInitSamplers(void);
// Must be called before using WebPYUV444Converters[]
void WebPInitYUV444Converters(void);

//------------------------------------------------------------------------------
// ARGB -> YUV converters

// Convert ARGB samples to luma Y.
extern void (*WebPConvertARGBToY)(const uint32_t* argb, uint8_t* y, int width);
// Convert ARGB samples to U/V with downsampling. do_store should be '1' for
// even lines and '0' for odd ones. 'src_width' is the original width, not
// the U/V one.
extern void (*WebPConvertARGBToUV)(const uint32_t* argb, uint8_t* u, uint8_t* v,
                                   int src_width, int do_store);

// Convert a row of accumulated (four-values) of rgba32 toward U/V
extern void (*WebPConvertRGBA32ToUV)(const uint16_t* rgb,
                                     uint8_t* u, uint8_t* v, int width);

// Convert RGB or BGR to Y
extern void (*WebPConvertRGB24ToY)(const uint8_t* rgb, uint8_t* y, int width);
extern void (*WebPConvertBGR24ToY)(const uint8_t* bgr, uint8_t* y, int width);

// used for plain-C fallback.
extern void WebPConvertARGBToUV_C(const uint32_t* argb, uint8_t* u, uint8_t* v,
                                  int src_width, int do_store);
extern void WebPConvertRGBA32ToUV_C(const uint16_t* rgb,
                                    uint8_t* u, uint8_t* v, int width);

// Must be called before using the above.
void WebPInitConvertARGBToYUV(void);

//------------------------------------------------------------------------------
// Rescaler

struct WebPRescaler;

// Import a row of data and save its contribution in the rescaler.
// 'channel' denotes the channel number to be imported. 'Expand' corresponds to
// the wrk->x_expand case. Otherwise, 'Shrink' is to be used.
typedef void (*WebPRescalerImportRowFunc)(struct WebPRescaler* const wrk,
                                          const uint8_t* src);

extern WebPRescalerImportRowFunc WebPRescalerImportRowExpand;
extern WebPRescalerImportRowFunc WebPRescalerImportRowShrink;

// Export one row (starting at x_out position) from rescaler.
// 'Expand' corresponds to the wrk->y_expand case.
// Otherwise 'Shrink' is to be used
typedef void (*WebPRescalerExportRowFunc)(struct WebPRescaler* const wrk);
extern WebPRescalerExportRowFunc WebPRescalerExportRowExpand;
extern WebPRescalerExportRowFunc WebPRescalerExportRowShrink;

// Plain-C implementation, as fall-back.
extern void WebPRescalerImportRowExpandC(struct WebPRescaler* const wrk,
                                         const uint8_t* src);
extern void WebPRescalerImportRowShrinkC(struct WebPRescaler* const wrk,
                                         const uint8_t* src);
extern void WebPRescalerExportRowExpandC(struct WebPRescaler* const wrk);
extern void WebPRescalerExportRowShrinkC(struct WebPRescaler* const wrk);

// Main entry calls:
extern void WebPRescalerImportRow(struct WebPRescaler* const wrk,
                                  const uint8_t* src);
// Export one row (starting at x_out position) from rescaler.
extern void WebPRescalerExportRow(struct WebPRescaler* const wrk);

// Must be called first before using the above.
void WebPRescalerDspInit(void);

//------------------------------------------------------------------------------
// Utilities for processing transparent channel.

// Apply alpha pre-multiply on an rgba, bgra or argb plane of size w * h.
// alpha_first should be 0 for argb, 1 for rgba or bgra (where alpha is last).
extern void (*WebPApplyAlphaMultiply)(
    uint8_t* rgba, int alpha_first, int w, int h, int stride);

// Same, buf specifically for RGBA4444 format
extern void (*WebPApplyAlphaMultiply4444)(
    uint8_t* rgba4444, int w, int h, int stride);

// Dispatch the values from alpha[] plane to the ARGB destination 'dst'.
// Returns true if alpha[] plane has non-trivial values different from 0xff.
extern int (*WebPDispatchAlpha)(const uint8_t* alpha, int alpha_stride,
                                int width, int height,
                                uint8_t* dst, int dst_stride);

// Transfer packed 8b alpha[] values to green channel in dst[], zero'ing the
// A/R/B values. 'dst_stride' is the stride for dst[] in uint32_t units.
extern void (*WebPDispatchAlphaToGreen)(const uint8_t* alpha, int alpha_stride,
                                        int width, int height,
                                        uint32_t* dst, int dst_stride);

// Extract the alpha values from 32b values in argb[] and pack them into alpha[]
// (this is the opposite of WebPDispatchAlpha).
// Returns true if there's only trivial 0xff alpha values.
extern int (*WebPExtractAlpha)(const uint8_t* argb, int argb_stride,
                               int width, int height,
                               uint8_t* alpha, int alpha_stride);

// Pre-Multiply operation transforms x into x * A / 255  (where x=Y,R,G or B).
// Un-Multiply operation transforms x into x * 255 / A.

// Pre-Multiply or Un-Multiply (if 'inverse' is true) argb values in a row.
extern void (*WebPMultARGBRow)(uint32_t* const ptr, int width, int inverse);

// Same a WebPMultARGBRow(), but for several rows.
void WebPMultARGBRows(uint8_t* ptr, int stride, int width, int num_rows,
                      int inverse);

// Same for a row of single values, with side alpha values.
extern void (*WebPMultRow)(uint8_t* const ptr, const uint8_t* const alpha,
                           int width, int inverse);

// Same a WebPMultRow(), but for several 'num_rows' rows.
void WebPMultRows(uint8_t* ptr, int stride,
                  const uint8_t* alpha, int alpha_stride,
                  int width, int num_rows, int inverse);

// Plain-C versions, used as fallback by some implementations.
void WebPMultRowC(uint8_t* const ptr, const uint8_t* const alpha,
                  int width, int inverse);
void WebPMultARGBRowC(uint32_t* const ptr, int width, int inverse);

// To be called first before using the above.
void WebPInitAlphaProcessing(void);

// ARGB packing function: a/r/g/b input is rgba or bgra order.
extern void (*VP8PackARGB)(const uint8_t* a, const uint8_t* r,
                           const uint8_t* g, const uint8_t* b, int len,
                           uint32_t* out);

// RGB packing function. 'step' can be 3 or 4. r/g/b input is rgb or bgr order.
extern void (*VP8PackRGB)(const uint8_t* r, const uint8_t* g, const uint8_t* b,
                          int len, int step, uint32_t* out);

// To be called first before using the above.
void VP8EncDspARGBInit(void);

//------------------------------------------------------------------------------
// Filter functions

typedef enum {     // Filter types.
  WEBP_FILTER_NONE = 0,
  WEBP_FILTER_HORIZONTAL,
  WEBP_FILTER_VERTICAL,
  WEBP_FILTER_GRADIENT,
  WEBP_FILTER_LAST = WEBP_FILTER_GRADIENT + 1,  // end marker
  WEBP_FILTER_BEST,    // meta-types
  WEBP_FILTER_FAST
} WEBP_FILTER_TYPE;

typedef void (*WebPFilterFunc)(const uint8_t* in, int width, int height,
                               int stride, uint8_t* out);
// In-place un-filtering.
// Warning! 'prev_line' pointer can be equal to 'cur_line' or 'preds'.
typedef void (*WebPUnfilterFunc)(const uint8_t* prev_line, const uint8_t* preds,
                                 uint8_t* cur_line, int width);

// Filter the given data using the given predictor.
// 'in' corresponds to a 2-dimensional pixel array of size (stride * height)
// in raster order.
// 'stride' is number of bytes per scan line (with possible padding).
// 'out' should be pre-allocated.
extern WebPFilterFunc WebPFilters[WEBP_FILTER_LAST];

// In-place reconstruct the original data from the given filtered data.
// The reconstruction will be done for 'num_rows' rows starting from 'row'
// (assuming rows upto 'row - 1' are already reconstructed).
extern WebPUnfilterFunc WebPUnfilters[WEBP_FILTER_LAST];

// To be called first before using the above.
void VP8FiltersInit(void);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_DSP_DSP_H_ */
