// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"

namespace jxl {

//------------------------------------------------------------------------------
// Slow

namespace {

template <class WrapX, class WrapY>
float SlowSymmetric3Pixel(const ImageF& in, const int64_t ix, const int64_t iy,
                          const int64_t xsize, const int64_t ysize,
                          const WeightsSymmetric3& weights) {
  float sum = 0.0f;

  // ix: image; kx: kernel
  for (int64_t ky = -1; ky <= 1; ky++) {
    const int64_t y = WrapY()(iy + ky, ysize);
    const float* JXL_RESTRICT row_in = in.ConstRow(static_cast<size_t>(y));

    const float wc = ky == 0 ? weights.c[0] : weights.r[0];
    const float wlr = ky == 0 ? weights.r[0] : weights.d[0];

    const int64_t xm1 = WrapX()(ix - 1, xsize);
    const int64_t xp1 = WrapX()(ix + 1, xsize);
    sum += row_in[ix] * wc + (row_in[xm1] + row_in[xp1]) * wlr;
  }
  return sum;
}

template <class WrapY>
void SlowSymmetric3Row(const ImageF& in, const int64_t iy, const int64_t xsize,
                       const int64_t ysize, const WeightsSymmetric3& weights,
                       float* JXL_RESTRICT row_out) {
  row_out[0] =
      SlowSymmetric3Pixel<WrapMirror, WrapY>(in, 0, iy, xsize, ysize, weights);
  for (int64_t ix = 1; ix < xsize - 1; ix++) {
    row_out[ix] = SlowSymmetric3Pixel<WrapUnchanged, WrapY>(in, ix, iy, xsize,
                                                            ysize, weights);
  }
  {
    const int64_t ix = xsize - 1;
    row_out[ix] = SlowSymmetric3Pixel<WrapMirror, WrapY>(in, ix, iy, xsize,
                                                         ysize, weights);
  }
}

}  // namespace

Status SlowSymmetric3(const ImageF& in, const Rect& rect,
                      const WeightsSymmetric3& weights, ThreadPool* pool,
                      ImageF* JXL_RESTRICT out) {
  const int64_t xsize = static_cast<int64_t>(rect.xsize());
  const int64_t ysize = static_cast<int64_t>(rect.ysize());
  const int64_t kRadius = 1;

  const auto process_row = [&](const uint32_t task,
                               size_t /*thread*/) -> Status {
    const int64_t iy = task;
    float* JXL_RESTRICT out_row = out->Row(static_cast<size_t>(iy));

    if (iy < kRadius || iy >= ysize - kRadius) {
      SlowSymmetric3Row<WrapMirror>(in, iy, xsize, ysize, weights, out_row);
    } else {
      SlowSymmetric3Row<WrapUnchanged>(in, iy, xsize, ysize, weights, out_row);
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, static_cast<uint32_t>(ysize),
                                ThreadPool::NoInit, process_row,
                                "SlowSymmetric3"));
  return true;
}

namespace {

// Separable kernels, any radius.
StatusOr<float> SlowSeparablePixel(const ImageF& in, const Rect& rect,
                                   const int64_t x, const int64_t y,
                                   const int64_t radius,
                                   const float* JXL_RESTRICT horz_weights,
                                   const float* JXL_RESTRICT vert_weights) {
  const size_t xsize = in.xsize();
  const size_t ysize = in.ysize();
  const WrapMirror wrap;

  float mul = 0.0f;
  for (int dy = -radius; dy <= radius; ++dy) {
    const float wy = vert_weights[std::abs(dy) * 4];
    const size_t sy = wrap(rect.y0() + y + dy, ysize);
    JXL_ENSURE(sy < ysize);
    const float* const JXL_RESTRICT row = in.ConstRow(sy);
    for (int dx = -radius; dx <= radius; ++dx) {
      const float wx = horz_weights[std::abs(dx) * 4];
      const size_t sx = wrap(rect.x0() + x + dx, xsize);
      JXL_ENSURE(sx < xsize);
      mul += row[sx] * wx * wy;
    }
  }
  return mul;
}

template <int R, typename Weights>
Status SlowSeparable(const ImageF& in, const Rect& in_rect,
                     const Weights& weights, ThreadPool* pool, ImageF* out,
                     const Rect& out_rect) {
  JXL_ENSURE(in_rect.xsize() == out_rect.xsize());
  JXL_ENSURE(in_rect.ysize() == out_rect.ysize());
  JXL_ENSURE(in_rect.IsInside(Rect(in)));
  JXL_ENSURE(out_rect.IsInside(Rect(*out)));
  const float* horz_weights = &weights.horz[0];
  const float* vert_weights = &weights.vert[0];

  const auto process_row = [&](const uint32_t task,
                               size_t /*thread*/) -> Status {
    const int64_t y = task;

    float* const JXL_RESTRICT row_out = out_rect.Row(out, y);
    for (size_t x = 0; x < in_rect.xsize(); ++x) {
      JXL_ASSIGN_OR_RETURN(row_out[x],
                           SlowSeparablePixel(in, in_rect, x, y, /*radius=*/R,
                                              horz_weights, vert_weights));
    }
    return true;
  };
  const size_t ysize = in_rect.ysize();
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, static_cast<uint32_t>(ysize),
                                ThreadPool::NoInit, process_row,
                                "SlowSeparable"));
  return true;
}

}  // namespace

Status SlowSeparable5(const ImageF& in, const Rect& in_rect,
                      const WeightsSeparable5& weights, ThreadPool* pool,
                      ImageF* out, const Rect& out_rect) {
  return SlowSeparable<2>(in, in_rect, weights, pool, out, out_rect);
}

}  // namespace jxl
