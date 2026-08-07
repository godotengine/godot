// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/base/status.h"
#include "lib/jxl/convolve.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/convolve_symmetric3.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/convolve-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Vec;

template <class WrapY, class V>
static V WeightedSum(const ImageF& in, const WrapY wrap_y, const size_t ix,
                     const int64_t iy, const size_t ysize, const V wx0,
                     const V wx1, const V wx2) {
  const HWY_FULL(float) d;
  const float* JXL_RESTRICT center = in.ConstRow(wrap_y(iy, ysize)) + ix;
  const auto in_m2 = LoadU(d, center - 2);
  const auto in_p2 = LoadU(d, center + 2);
  const auto in_m1 = LoadU(d, center - 1);
  const auto in_p1 = LoadU(d, center + 1);
  const auto in_00 = Load(d, center);
  const auto sum_2 = Mul(wx2, Add(in_m2, in_p2));
  const auto sum_1 = Mul(wx1, Add(in_m1, in_p1));
  const auto sum_0 = Mul(wx0, in_00);
  return Add(sum_2, Add(sum_1, sum_0));
}

// 3x3 convolution by symmetric kernel with a single scan through the input.
class Symmetric3Strategy {
  using D = HWY_CAPPED(float, 16);
  using V = Vec<D>;

 public:
  static constexpr int64_t kRadius = 1;

  // Only accesses pixels in [0, xsize).
  template <size_t kSizeModN, class WrapRow>
  static JXL_MAYBE_INLINE void ConvolveRow(
      const float* const JXL_RESTRICT row_m, const size_t xsize,
      const int64_t stride, const WrapRow& wrap_row,
      const WeightsSymmetric3& weights, float* const JXL_RESTRICT row_out) {
    const D d;
    // t, m, b = top, middle, bottom row;
    const float* const JXL_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const JXL_RESTRICT row_b = wrap_row(row_m + stride, stride);

    // Must load in advance - compiler doesn't understand LoadDup128 and
    // schedules them too late.
    const V w0 = LoadDup128(d, weights.c);
    const V w1 = LoadDup128(d, weights.r);
    const V w2 = LoadDup128(d, weights.d);

    // l, c, r = left, center, right. Leftmost vector: need FirstL1.
    {
      const V tc = LoadU(d, row_t + 0);
      const V mc = LoadU(d, row_m + 0);
      const V bc = LoadU(d, row_b + 0);
      const V tl = Neighbors::FirstL1(tc);
      const V tr = LoadU(d, row_t + 0 + 1);
      const V ml = Neighbors::FirstL1(mc);
      const V mr = LoadU(d, row_m + 0 + 1);
      const V bl = Neighbors::FirstL1(bc);
      const V br = LoadU(d, row_b + 0 + 1);
      const V conv =
          WeightedSum(tl, tc, tr, ml, mc, mr, bl, bc, br, w0, w1, w2);
      Store(conv, d, row_out + 0);
    }

    // Loop as long as we can load enough new values:
    const size_t N = Lanes(d);
    size_t x = N;
    for (; x + N + kRadius <= xsize; x += N) {
      const auto conv = ConvolveValid(row_t, row_m, row_b, x, w0, w1, w2);
      Store(conv, d, row_out + x);
    }

    // For final (partial) vector:
    const V tc = LoadU(d, row_t + x);
    const V mc = LoadU(d, row_m + x);
    const V bc = LoadU(d, row_b + x);

    V tr;
    V mr;
    V br;
#if HWY_TARGET == HWY_SCALAR
    tr = tc;  // Single-lane => mirrored right neighbor = center value.
    mr = mc;
    br = bc;
#else
    if (kSizeModN == 0) {
      // The above loop didn't handle the last vector because it needs an
      // additional right neighbor (generated via mirroring).
      auto mirror = SetTableIndices(d, MirrorLanes(N - 1));
      tr = TableLookupLanes(tc, mirror);
      mr = TableLookupLanes(mc, mirror);
      br = TableLookupLanes(bc, mirror);
    } else {
      auto mirror = SetTableIndices(d, MirrorLanes((xsize % N) - 1));
      // Loads last valid value into uppermost lane and mirrors.
      tr = TableLookupLanes(LoadU(d, row_t + xsize - N), mirror);
      mr = TableLookupLanes(LoadU(d, row_m + xsize - N), mirror);
      br = TableLookupLanes(LoadU(d, row_b + xsize - N), mirror);
    }
#endif

    const V tl = LoadU(d, row_t + x - 1);
    const V ml = LoadU(d, row_m + x - 1);
    const V bl = LoadU(d, row_b + x - 1);
    const V conv = WeightedSum(tl, tc, tr, ml, mc, mr, bl, bc, br, w0, w1, w2);
    Store(conv, d, row_out + x);
  }

 private:
  // Returns sum{x_i * w_i}.
  template <class V>
  static JXL_MAYBE_INLINE V WeightedSum(const V tl, const V tc, const V tr,
                                        const V ml, const V mc, const V mr,
                                        const V bl, const V bc, const V br,
                                        const V w0, const V w1, const V w2) {
    const V sum_tb = Add(tc, bc);

    // Faster than 5 mul + 4 FMA.
    const V mul0 = Mul(mc, w0);
    const V sum_lr = Add(ml, mr);

    const V x1 = Add(sum_tb, sum_lr);
    const V mul1 = MulAdd(x1, w1, mul0);

    const V sum_t2 = Add(tl, tr);
    const V sum_b2 = Add(bl, br);
    const V x2 = Add(sum_t2, sum_b2);
    const V mul2 = MulAdd(x2, w2, mul1);
    return mul2;
  }

  static JXL_MAYBE_INLINE V ConvolveValid(const float* JXL_RESTRICT row_t,
                                          const float* JXL_RESTRICT row_m,
                                          const float* JXL_RESTRICT row_b,
                                          const int64_t x, const V w0,
                                          const V w1, const V w2) {
    const D d;
    const V tc = LoadU(d, row_t + x);
    const V mc = LoadU(d, row_m + x);
    const V bc = LoadU(d, row_b + x);
    const V tl = LoadU(d, row_t + x - 1);
    const V tr = LoadU(d, row_t + x + 1);
    const V ml = LoadU(d, row_m + x - 1);
    const V mr = LoadU(d, row_m + x + 1);
    const V bl = LoadU(d, row_b + x - 1);
    const V br = LoadU(d, row_b + x + 1);
    return WeightedSum(tl, tc, tr, ml, mc, mr, bl, bc, br, w0, w1, w2);
  }
};

Status Symmetric3(const ImageF& in, const Rect& rect,
                  const WeightsSymmetric3& weights, ThreadPool* pool,
                  ImageF* out) {
  using Conv = ConvolveT<Symmetric3Strategy>;
  if (rect.xsize() >= Conv::MinWidth()) {
    JXL_ENSURE(SameSize(rect, *out));
    JXL_ENSURE(rect.xsize() >= Conv::MinWidth());
    Conv::Run(in, rect, weights, pool, out);
    return true;
  }

  JXL_RETURN_IF_ERROR(SlowSymmetric3(in, rect, weights, pool, out));
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(Symmetric3);
Status Symmetric3(const ImageF& in, const Rect& rect,
                  const WeightsSymmetric3& weights, ThreadPool* pool,
                  ImageF* out) {
  return HWY_DYNAMIC_DISPATCH(Symmetric3)(in, rect, weights, pool, out);
}

}  // namespace jxl
#endif  // HWY_ONCE
