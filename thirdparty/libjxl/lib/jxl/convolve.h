// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_CONVOLVE_H_
#define LIB_JXL_CONVOLVE_H_

// 2D convolution.

#include <cstddef>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image.h"

namespace jxl {

// No valid values outside [0, xsize), but the strategy may still safely load
// the preceding vector, and/or round xsize up to the vector lane count. This
// avoids needing PadImage.
// Requires xsize >= kConvolveLanes + kConvolveMaxRadius.
static constexpr size_t kConvolveMaxRadius = 3;

// Weights must already be normalized.

struct WeightsSymmetric3 {
  // d r d (each replicated 4x)
  // r c r
  // d r d
  float c[4];
  float r[4];
  float d[4];
};

struct WeightsSymmetric5 {
  // The lower-right quadrant is: c r R  (each replicated 4x)
  //                              r d L
  //                              R L D
  float c[4];
  float r[4];
  float R[4];
  float d[4];
  float D[4];
  float L[4];
};

// Weights for separable 5x5 filters (typically but not necessarily the same
// values for horizontal and vertical directions). The kernel must already be
// normalized, but note that values for negative offsets are omitted, so the
// given values do not sum to 1.
struct WeightsSeparable5 {
  // Horizontal 1D, distances 0..2 (each replicated 4x)
  float horz[3 * 4];
  float vert[3 * 4];
};

const WeightsSymmetric3& WeightsSymmetric3Lowpass();
const WeightsSeparable5& WeightsSeparable5Lowpass();
const WeightsSymmetric5& WeightsSymmetric5Lowpass();

Status SlowSymmetric3(const ImageF& in, const Rect& rect,
                      const WeightsSymmetric3& weights, ThreadPool* pool,
                      ImageF* JXL_RESTRICT out);

Status SlowSeparable5(const ImageF& in, const Rect& in_rect,
                      const WeightsSeparable5& weights, ThreadPool* pool,
                      ImageF* out, const Rect& out_rect);

Status Symmetric3(const ImageF& in, const Rect& rect,
                  const WeightsSymmetric3& weights, ThreadPool* pool,
                  ImageF* out);

Status Symmetric5(const ImageF& in, const Rect& in_rect,
                  const WeightsSymmetric5& weights, ThreadPool* pool,
                  ImageF* JXL_RESTRICT out, const Rect& out_rect);

Status Symmetric5(const ImageF& in, const Rect& rect,
                  const WeightsSymmetric5& weights, ThreadPool* pool,
                  ImageF* JXL_RESTRICT out);

Status Separable5(const ImageF& in, const Rect& rect,
                  const WeightsSeparable5& weights, ThreadPool* pool,
                  ImageF* out);

}  // namespace jxl

#endif  // LIB_JXL_CONVOLVE_H_
