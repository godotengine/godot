/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/compare.h"

#include <float.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "libyuv/basic_types.h"
#include "libyuv/compare_row.h"
#include "libyuv/cpu_id.h"
#include "libyuv/row.h"
#include "libyuv/video_common.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// hash seed of 5381 recommended.
LIBYUV_API
uint32_t HashDjb2(const uint8_t* src, uint64_t count, uint32_t seed) {
  const int kBlockSize = 1 << 15;  // 32768;
  int remainder;
  uint32_t (*HashDjb2_SSE)(const uint8_t* src, int count, uint32_t seed) =
      HashDjb2_C;
#if defined(HAS_HASHDJB2_SSE41)
  if (TestCpuFlag(kCpuHasSSE41)) {
    HashDjb2_SSE = HashDjb2_SSE41;
  }
#endif
#if defined(HAS_HASHDJB2_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    HashDjb2_SSE = HashDjb2_AVX2;
  }
#endif

  while (count >= (uint64_t)(kBlockSize)) {
    seed = HashDjb2_SSE(src, kBlockSize, seed);
    src += kBlockSize;
    count -= kBlockSize;
  }
  remainder = (int)count & ~15;
  if (remainder) {
    seed = HashDjb2_SSE(src, remainder, seed);
    src += remainder;
    count -= remainder;
  }
  remainder = (int)count & 15;
  if (remainder) {
    seed = HashDjb2_C(src, remainder, seed);
  }
  return seed;
}

static uint32_t ARGBDetectRow_C(const uint8_t* argb, int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    if (argb[0] != 255) {  // First byte is not Alpha of 255, so not ARGB.
      return FOURCC_BGRA;
    }
    if (argb[3] != 255) {  // 4th byte is not Alpha of 255, so not BGRA.
      return FOURCC_ARGB;
    }
    if (argb[4] != 255) {  // Second pixel first byte is not Alpha of 255.
      return FOURCC_BGRA;
    }
    if (argb[7] != 255) {  // Second pixel 4th byte is not Alpha of 255.
      return FOURCC_ARGB;
    }
    argb += 8;
  }
  if (width & 1) {
    if (argb[0] != 255) {  // First byte is not Alpha of 255, so not ARGB.
      return FOURCC_BGRA;
    }
    if (argb[3] != 255) {  // 4th byte is not Alpha of 255, so not BGRA.
      return FOURCC_ARGB;
    }
  }
  return 0;
}

// Scan an opaque argb image and return fourcc based on alpha offset.
// Returns FOURCC_ARGB, FOURCC_BGRA, or 0 if unknown.
LIBYUV_API
uint32_t ARGBDetect(const uint8_t* argb,
                    int stride_argb,
                    int width,
                    int height) {
  uint32_t fourcc = 0;
  int h;

  // Coalesce rows.
  if (stride_argb == width * 4) {
    width *= height;
    height = 1;
    stride_argb = 0;
  }
  for (h = 0; h < height && fourcc == 0; ++h) {
    fourcc = ARGBDetectRow_C(argb, width);
    argb += stride_argb;
  }
  return fourcc;
}

// NEON version accumulates in 16 bit shorts which overflow at 65536 bytes.
// So actual maximum is 1 less loop, which is 64436 - 32 bytes.

LIBYUV_API
uint64_t ComputeHammingDistance(const uint8_t* src_a,
                                const uint8_t* src_b,
                                int count) {
  const int kBlockSize = 1 << 15;  // 32768;
  const int kSimdSize = 64;
  // SIMD for multiple of 64, and C for remainder
  int remainder = count & (kBlockSize - 1) & ~(kSimdSize - 1);
  uint64_t diff = 0;
  int i;
  uint32_t (*HammingDistance)(const uint8_t* src_a, const uint8_t* src_b,
                              int count) = HammingDistance_C;
#if defined(HAS_HAMMINGDISTANCE_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    HammingDistance = HammingDistance_NEON;
  }
#endif
#if defined(HAS_HAMMINGDISTANCE_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    HammingDistance = HammingDistance_SSSE3;
  }
#endif
#if defined(HAS_HAMMINGDISTANCE_SSE42)
  if (TestCpuFlag(kCpuHasSSE42)) {
    HammingDistance = HammingDistance_SSE42;
  }
#endif
#if defined(HAS_HAMMINGDISTANCE_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    HammingDistance = HammingDistance_AVX2;
  }
#endif
#if defined(HAS_HAMMINGDISTANCE_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    HammingDistance = HammingDistance_MSA;
  }
#endif
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : diff)
#endif
  for (i = 0; i < (count - (kBlockSize - 1)); i += kBlockSize) {
    diff += HammingDistance(src_a + i, src_b + i, kBlockSize);
  }
  src_a += count & ~(kBlockSize - 1);
  src_b += count & ~(kBlockSize - 1);
  if (remainder) {
    diff += HammingDistance(src_a, src_b, remainder);
    src_a += remainder;
    src_b += remainder;
  }
  remainder = count & (kSimdSize - 1);
  if (remainder) {
    diff += HammingDistance_C(src_a, src_b, remainder);
  }
  return diff;
}

// TODO(fbarchard): Refactor into row function.
LIBYUV_API
uint64_t ComputeSumSquareError(const uint8_t* src_a,
                               const uint8_t* src_b,
                               int count) {
  // SumSquareError returns values 0 to 65535 for each squared difference.
  // Up to 65536 of those can be summed and remain within a uint32_t.
  // After each block of 65536 pixels, accumulate into a uint64_t.
  const int kBlockSize = 65536;
  int remainder = count & (kBlockSize - 1) & ~31;
  uint64_t sse = 0;
  int i;
  uint32_t (*SumSquareError)(const uint8_t* src_a, const uint8_t* src_b,
                             int count) = SumSquareError_C;
#if defined(HAS_SUMSQUAREERROR_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SumSquareError = SumSquareError_NEON;
  }
#endif
#if defined(HAS_SUMSQUAREERROR_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    // Note only used for multiples of 16 so count is not checked.
    SumSquareError = SumSquareError_SSE2;
  }
#endif
#if defined(HAS_SUMSQUAREERROR_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    // Note only used for multiples of 32 so count is not checked.
    SumSquareError = SumSquareError_AVX2;
  }
#endif
#if defined(HAS_SUMSQUAREERROR_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    SumSquareError = SumSquareError_MSA;
  }
#endif
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sse)
#endif
  for (i = 0; i < (count - (kBlockSize - 1)); i += kBlockSize) {
    sse += SumSquareError(src_a + i, src_b + i, kBlockSize);
  }
  src_a += count & ~(kBlockSize - 1);
  src_b += count & ~(kBlockSize - 1);
  if (remainder) {
    sse += SumSquareError(src_a, src_b, remainder);
    src_a += remainder;
    src_b += remainder;
  }
  remainder = count & 31;
  if (remainder) {
    sse += SumSquareError_C(src_a, src_b, remainder);
  }
  return sse;
}

LIBYUV_API
uint64_t ComputeSumSquareErrorPlane(const uint8_t* src_a,
                                    int stride_a,
                                    const uint8_t* src_b,
                                    int stride_b,
                                    int width,
                                    int height) {
  uint64_t sse = 0;
  int h;
  // Coalesce rows.
  if (stride_a == width && stride_b == width) {
    width *= height;
    height = 1;
    stride_a = stride_b = 0;
  }
  for (h = 0; h < height; ++h) {
    sse += ComputeSumSquareError(src_a, src_b, width);
    src_a += stride_a;
    src_b += stride_b;
  }
  return sse;
}

LIBYUV_API
double SumSquareErrorToPsnr(uint64_t sse, uint64_t count) {
  double psnr;
  if (sse > 0) {
    double mse = (double)count / (double)sse;
    psnr = 10.0 * log10(255.0 * 255.0 * mse);
  } else {
    psnr = kMaxPsnr;  // Limit to prevent divide by 0
  }

  if (psnr > kMaxPsnr) {
    psnr = kMaxPsnr;
  }

  return psnr;
}

LIBYUV_API
double CalcFramePsnr(const uint8_t* src_a,
                     int stride_a,
                     const uint8_t* src_b,
                     int stride_b,
                     int width,
                     int height) {
  const uint64_t samples = (uint64_t)width * (uint64_t)height;
  const uint64_t sse = ComputeSumSquareErrorPlane(src_a, stride_a, src_b,
                                                  stride_b, width, height);
  return SumSquareErrorToPsnr(sse, samples);
}

LIBYUV_API
double I420Psnr(const uint8_t* src_y_a,
                int stride_y_a,
                const uint8_t* src_u_a,
                int stride_u_a,
                const uint8_t* src_v_a,
                int stride_v_a,
                const uint8_t* src_y_b,
                int stride_y_b,
                const uint8_t* src_u_b,
                int stride_u_b,
                const uint8_t* src_v_b,
                int stride_v_b,
                int width,
                int height) {
  const uint64_t sse_y = ComputeSumSquareErrorPlane(
      src_y_a, stride_y_a, src_y_b, stride_y_b, width, height);
  const int width_uv = (width + 1) >> 1;
  const int height_uv = (height + 1) >> 1;
  const uint64_t sse_u = ComputeSumSquareErrorPlane(
      src_u_a, stride_u_a, src_u_b, stride_u_b, width_uv, height_uv);
  const uint64_t sse_v = ComputeSumSquareErrorPlane(
      src_v_a, stride_v_a, src_v_b, stride_v_b, width_uv, height_uv);
  const uint64_t samples = (uint64_t)width * (uint64_t)height +
                           2 * ((uint64_t)width_uv * (uint64_t)height_uv);
  const uint64_t sse = sse_y + sse_u + sse_v;
  return SumSquareErrorToPsnr(sse, samples);
}

static const int64_t cc1 = 26634;   // (64^2*(.01*255)^2
static const int64_t cc2 = 239708;  // (64^2*(.03*255)^2

static double Ssim8x8_C(const uint8_t* src_a,
                        int stride_a,
                        const uint8_t* src_b,
                        int stride_b) {
  int64_t sum_a = 0;
  int64_t sum_b = 0;
  int64_t sum_sq_a = 0;
  int64_t sum_sq_b = 0;
  int64_t sum_axb = 0;

  int i;
  for (i = 0; i < 8; ++i) {
    int j;
    for (j = 0; j < 8; ++j) {
      sum_a += src_a[j];
      sum_b += src_b[j];
      sum_sq_a += src_a[j] * src_a[j];
      sum_sq_b += src_b[j] * src_b[j];
      sum_axb += src_a[j] * src_b[j];
    }

    src_a += stride_a;
    src_b += stride_b;
  }

  {
    const int64_t count = 64;
    // scale the constants by number of pixels
    const int64_t c1 = (cc1 * count * count) >> 12;
    const int64_t c2 = (cc2 * count * count) >> 12;

    const int64_t sum_a_x_sum_b = sum_a * sum_b;

    const int64_t ssim_n = (2 * sum_a_x_sum_b + c1) *
                           (2 * count * sum_axb - 2 * sum_a_x_sum_b + c2);

    const int64_t sum_a_sq = sum_a * sum_a;
    const int64_t sum_b_sq = sum_b * sum_b;

    const int64_t ssim_d =
        (sum_a_sq + sum_b_sq + c1) *
        (count * sum_sq_a - sum_a_sq + count * sum_sq_b - sum_b_sq + c2);

    if (ssim_d == 0.0) {
      return DBL_MAX;
    }
    return ssim_n * 1.0 / ssim_d;
  }
}

// We are using a 8x8 moving window with starting location of each 8x8 window
// on the 4x4 pixel grid. Such arrangement allows the windows to overlap
// block boundaries to penalize blocking artifacts.
LIBYUV_API
double CalcFrameSsim(const uint8_t* src_a,
                     int stride_a,
                     const uint8_t* src_b,
                     int stride_b,
                     int width,
                     int height) {
  int samples = 0;
  double ssim_total = 0;
  double (*Ssim8x8)(const uint8_t* src_a, int stride_a, const uint8_t* src_b,
                    int stride_b) = Ssim8x8_C;

  // sample point start with each 4x4 location
  int i;
  for (i = 0; i < height - 8; i += 4) {
    int j;
    for (j = 0; j < width - 8; j += 4) {
      ssim_total += Ssim8x8(src_a + j, stride_a, src_b + j, stride_b);
      samples++;
    }

    src_a += stride_a * 4;
    src_b += stride_b * 4;
  }

  ssim_total /= samples;
  return ssim_total;
}

LIBYUV_API
double I420Ssim(const uint8_t* src_y_a,
                int stride_y_a,
                const uint8_t* src_u_a,
                int stride_u_a,
                const uint8_t* src_v_a,
                int stride_v_a,
                const uint8_t* src_y_b,
                int stride_y_b,
                const uint8_t* src_u_b,
                int stride_u_b,
                const uint8_t* src_v_b,
                int stride_v_b,
                int width,
                int height) {
  const double ssim_y =
      CalcFrameSsim(src_y_a, stride_y_a, src_y_b, stride_y_b, width, height);
  const int width_uv = (width + 1) >> 1;
  const int height_uv = (height + 1) >> 1;
  const double ssim_u = CalcFrameSsim(src_u_a, stride_u_a, src_u_b, stride_u_b,
                                      width_uv, height_uv);
  const double ssim_v = CalcFrameSsim(src_v_a, stride_v_a, src_v_b, stride_v_b,
                                      width_uv, height_uv);
  return ssim_y * 0.8 + 0.1 * (ssim_u + ssim_v);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
