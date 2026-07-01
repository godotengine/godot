// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/convolve.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/convolve_separable5.cc"
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

// 5x5 convolution by separable kernel with a single scan through the input.
// This is more cache-efficient than separate horizontal/vertical passes, and
// possibly faster (given enough registers) than tiling and/or transposing.
//
// Overview: imagine a 5x5 window around a central pixel. First convolve the
// rows by multiplying the pixels with the corresponding weights from
// WeightsSeparable5.horz[abs(x_offset) * 4]. Then multiply each of these
// intermediate results by the corresponding vertical weight, i.e.
// vert[abs(y_offset) * 4]. Finally, store the sum of these values as the
// convolution result at the position of the central pixel in the output.
//
// Each of these operations uses SIMD vectors. The central pixel and most
// importantly the output are aligned, so neighnoring pixels (e.g. x_offset=1)
// require unaligned loads. Because weights are supplied in identical groups of
// 4, we can use LoadDup128 to load them (slightly faster).
//
// Uses mirrored boundary handling. Until x >= kRadius, the horizontal
// convolution uses Neighbors class to shuffle vectors as if each of its lanes
// had been loaded from the mirrored offset. Similarly, the last full vector to
// write uses mirroring. In the case of scalar vectors, Neighbors is not usable
// and the value is loaded directly. Otherwise, the number of valid pixels
// modulo the vector size enables a small optimization: for smaller offsets,
// a non-mirrored load is sufficient.
class Separable5Strategy {
  using D = HWY_CAPPED(float, 16);
  using V = Vec<D>;

 public:
  static constexpr int64_t kRadius = 2;

  template <size_t kSizeModN, class WrapRow>
  static JXL_MAYBE_INLINE void ConvolveRow(
      const float* const JXL_RESTRICT row_m, const size_t xsize,
      const int64_t stride, const WrapRow& wrap_row,
      const WeightsSeparable5& weights, float* const JXL_RESTRICT row_out) {
    const D d;
    const int64_t neg_stride = -stride;  // allows LEA addressing.
    const float* const JXL_RESTRICT row_t2 =
        wrap_row(row_m + 2 * neg_stride, stride);
    const float* const JXL_RESTRICT row_t1 =
        wrap_row(row_m + 1 * neg_stride, stride);
    const float* const JXL_RESTRICT row_b1 =
        wrap_row(row_m + 1 * stride, stride);
    const float* const JXL_RESTRICT row_b2 =
        wrap_row(row_m + 2 * stride, stride);

    const V wh0 = LoadDup128(d, weights.horz + 0 * 4);
    const V wh1 = LoadDup128(d, weights.horz + 1 * 4);
    const V wh2 = LoadDup128(d, weights.horz + 2 * 4);
    const V wv0 = LoadDup128(d, weights.vert + 0 * 4);
    const V wv1 = LoadDup128(d, weights.vert + 1 * 4);
    const V wv2 = LoadDup128(d, weights.vert + 2 * 4);

    size_t x = 0;

    // More than one iteration for scalars.
    for (; x < kRadius; x += Lanes(d)) {
      const V conv0 =
          Mul(HorzConvolveFirst(row_m, x, xsize, wh0, wh1, wh2), wv0);

      const V conv1t = HorzConvolveFirst(row_t1, x, xsize, wh0, wh1, wh2);
      const V conv1b = HorzConvolveFirst(row_b1, x, xsize, wh0, wh1, wh2);
      const V conv1 = MulAdd(Add(conv1t, conv1b), wv1, conv0);

      const V conv2t = HorzConvolveFirst(row_t2, x, xsize, wh0, wh1, wh2);
      const V conv2b = HorzConvolveFirst(row_b2, x, xsize, wh0, wh1, wh2);
      const V conv2 = MulAdd(Add(conv2t, conv2b), wv2, conv1);
      Store(conv2, d, row_out + x);
    }

    // Main loop: load inputs without padding
    for (; x + Lanes(d) + kRadius <= xsize; x += Lanes(d)) {
      const V conv0 = Mul(HorzConvolve(row_m + x, wh0, wh1, wh2), wv0);

      const V conv1t = HorzConvolve(row_t1 + x, wh0, wh1, wh2);
      const V conv1b = HorzConvolve(row_b1 + x, wh0, wh1, wh2);
      const V conv1 = MulAdd(Add(conv1t, conv1b), wv1, conv0);

      const V conv2t = HorzConvolve(row_t2 + x, wh0, wh1, wh2);
      const V conv2b = HorzConvolve(row_b2 + x, wh0, wh1, wh2);
      const V conv2 = MulAdd(Add(conv2t, conv2b), wv2, conv1);
      Store(conv2, d, row_out + x);
    }

    // Last full vector to write (the above loop handled mod >= kRadius)
#if HWY_TARGET == HWY_SCALAR
    while (x < xsize) {
#else
    if (kSizeModN < kRadius) {
#endif
      const V conv0 =
          Mul(HorzConvolveLast<kSizeModN>(row_m, x, xsize, wh0, wh1, wh2), wv0);

      const V conv1t =
          HorzConvolveLast<kSizeModN>(row_t1, x, xsize, wh0, wh1, wh2);
      const V conv1b =
          HorzConvolveLast<kSizeModN>(row_b1, x, xsize, wh0, wh1, wh2);
      const V conv1 = MulAdd(Add(conv1t, conv1b), wv1, conv0);

      const V conv2t =
          HorzConvolveLast<kSizeModN>(row_t2, x, xsize, wh0, wh1, wh2);
      const V conv2b =
          HorzConvolveLast<kSizeModN>(row_b2, x, xsize, wh0, wh1, wh2);
      const V conv2 = MulAdd(Add(conv2t, conv2b), wv2, conv1);
      Store(conv2, d, row_out + x);
      x += Lanes(d);
    }

    // If mod = 0, the above vector was the last.
    if (kSizeModN != 0) {
      for (; x < xsize; ++x) {
        float mul = 0.0f;
        for (int64_t dy = -kRadius; dy <= kRadius; ++dy) {
          const float wy = weights.vert[std::abs(dy) * 4];
          const float* clamped_row = wrap_row(row_m + dy * stride, stride);
          for (int64_t dx = -kRadius; dx <= kRadius; ++dx) {
            const float wx = weights.horz[std::abs(dx) * 4];
            const int64_t clamped_x = Mirror(x + dx, xsize);
            mul += clamped_row[clamped_x] * wx * wy;
          }
        }
        row_out[x] = mul;
      }
    }
  }

 private:
  // Same as HorzConvolve for the first/last vector in a row.
  static JXL_MAYBE_INLINE V HorzConvolveFirst(
      const float* const JXL_RESTRICT row, const int64_t x, const int64_t xsize,
      const V wh0, const V wh1, const V wh2) {
    const D d;
    const V c = LoadU(d, row + x);
    const V mul0 = Mul(c, wh0);

#if HWY_TARGET == HWY_SCALAR
    const V l1 = LoadU(d, row + Mirror(x - 1, xsize));
    const V l2 = LoadU(d, row + Mirror(x - 2, xsize));
#else
    (void)xsize;
    const V l1 = Neighbors::FirstL1(c);
    const V l2 = Neighbors::FirstL2(c);
#endif

    const V r1 = LoadU(d, row + x + 1);
    const V r2 = LoadU(d, row + x + 2);

    const V mul1 = MulAdd(Add(l1, r1), wh1, mul0);
    const V mul2 = MulAdd(Add(l2, r2), wh2, mul1);
    return mul2;
  }

  template <size_t kSizeModN>
  static JXL_MAYBE_INLINE V
  HorzConvolveLast(const float* const JXL_RESTRICT row, const int64_t x,
                   const int64_t xsize, const V wh0, const V wh1, const V wh2) {
    const D d;
    const V c = LoadU(d, row + x);
    const V mul0 = Mul(c, wh0);

    const V l1 = LoadU(d, row + x - 1);
    const V l2 = LoadU(d, row + x - 2);

    V r1;
    V r2;
#if HWY_TARGET == HWY_SCALAR
    r1 = LoadU(d, row + Mirror(x + 1, xsize));
    r2 = LoadU(d, row + Mirror(x + 2, xsize));
#else
    const size_t N = Lanes(d);
    if (kSizeModN == 0) {
      r2 = TableLookupLanes(c, SetTableIndices(d, MirrorLanes(N - 2)));
      r1 = TableLookupLanes(c, SetTableIndices(d, MirrorLanes(N - 1)));
    } else {  // == 1
      const auto last = LoadU(d, row + xsize - N);
      r2 = TableLookupLanes(last, SetTableIndices(d, MirrorLanes(N - 1)));
      r1 = last;
    }
#endif

    // Sum of pixels with Manhattan distance i, multiplied by weights[i].
    const V sum1 = Add(l1, r1);
    const V mul1 = MulAdd(sum1, wh1, mul0);
    const V sum2 = Add(l2, r2);
    const V mul2 = MulAdd(sum2, wh2, mul1);
    return mul2;
  }

  // Requires kRadius valid pixels before/after pos.
  static JXL_MAYBE_INLINE V HorzConvolve(const float* const JXL_RESTRICT pos,
                                         const V wh0, const V wh1,
                                         const V wh2) {
    const D d;
    const V c = LoadU(d, pos);
    const V mul0 = Mul(c, wh0);

    // Loading anew is faster than combining vectors.
    const V l1 = LoadU(d, pos - 1);
    const V r1 = LoadU(d, pos + 1);
    const V l2 = LoadU(d, pos - 2);
    const V r2 = LoadU(d, pos + 2);
    // Sum of pixels with Manhattan distance i, multiplied by weights[i].
    const V sum1 = Add(l1, r1);
    const V mul1 = MulAdd(sum1, wh1, mul0);
    const V sum2 = Add(l2, r2);
    const V mul2 = MulAdd(sum2, wh2, mul1);
    return mul2;
  }
};

Status Separable5(const ImageF& in, const Rect& rect,
                  const WeightsSeparable5& weights, ThreadPool* pool,
                  ImageF* out) {
  using Conv = ConvolveT<Separable5Strategy>;
  if (rect.xsize() >= Conv::MinWidth()) {
    JXL_ENSURE(SameSize(rect, *out));
    JXL_ENSURE(rect.xsize() >= Conv::MinWidth());

    Conv::Run(in, rect, weights, pool, out);
    return true;
  }

  return SlowSeparable5(in, rect, weights, pool, out, Rect(*out));
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(Separable5);
Status Separable5(const ImageF& in, const Rect& rect,
                  const WeightsSeparable5& weights, ThreadPool* pool,
                  ImageF* out) {
  return HWY_DYNAMIC_DISPATCH(Separable5)(in, rect, weights, pool, out);
}

}  // namespace jxl
#endif  // HWY_ONCE
