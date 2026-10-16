// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_convolve_separable5.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/convolve-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::IndicesFromVec;
using hwy::HWY_NAMESPACE::Iota;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Min;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::Vec;

using D = HWY_CAPPED(float, 16);
using DI32 = HWY_CAPPED(int32_t, 16);
using V = Vec<D>;
using VI32 = Vec<DI32>;
using I = decltype(SetTableIndices(D(), static_cast<int32_t*>(nullptr)));

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
class Separable5Impl {
 public:
  using Simd = HWY_CAPPED(float, 16);
  static constexpr int64_t kRadius = 2;

  Separable5Impl(const ImageF* in, const Rect& rect,
                 const WeightsSeparable5* weights, ThreadPool* pool,
                 ImageF* out)
      : in(in), rect(rect), weights(weights), pool(pool), out(out) {}

  Status Run() {
#if HWY_TARGET == HWY_SCALAR
    // First/Last use mirrored loads of up to +/- kRadius.
    size_t min_width = 2 * kRadius;
#else
    size_t min_width = Lanes(Simd()) + kRadius;
#endif

    if (rect.xsize() >= min_width) {
      JXL_ENSURE(SameSize(rect, *out));

      switch (rect.xsize() % Lanes(Simd())) {
        case 0:
          RunRows<0>();
          break;
        case 1:
          RunRows<1>();
          break;
        case 2:
          RunRows<2>();
          break;
        default:
          RunRows<3>();
          break;
      }
      return true;
    } else {
      return SlowSeparable5(*in, rect, *weights, pool, out, Rect(*out));
    }
  }

  template <size_t kSizeModN, bool kBorder>
  JXL_NOINLINE void ConvolveRow(const uint32_t y) {
    const D d;
    const int64_t stride = in->PixelsPerRow();
    const int64_t neg_stride = -stride;  // allows LEA addressing.
    const size_t xsize = rect.xsize();
    const float* const JXL_RESTRICT row_m = rect.ConstRow(*in, y);
    float* const JXL_RESTRICT row_out = out->Row(y);
    const float* JXL_RESTRICT row_t2 = row_m + 2 * neg_stride;
    const float* JXL_RESTRICT row_t1 = row_m + 1 * neg_stride;
    const float* JXL_RESTRICT row_b1 = row_m + 1 * stride;
    const float* JXL_RESTRICT row_b2 = row_m + 2 * stride;

    if (kBorder) {
      size_t img_y = rect.y0() + y;
      if (in->ysize() <= 2 * kRadius) {  // Very special: double reflections
        static constexpr size_t kBorderLut[4 * 8] = {
            0, 0, 0, 0, 0, 0xBAD, 0xBAD, 0xBAD,  // 1 row
            1, 0, 0, 1, 1, 0,     0xBAD, 0xBAD,  // 2 rows
            1, 0, 0, 1, 2, 2,     1,     0xBAD,  // 3 rows
            1, 0, 0, 1, 2, 3,     3,     2,      // 4 rows
        };
        JXL_DASSERT(in->ysize() <= 4);
        size_t o = in->ysize() * 8 - 6 + img_y;
        row_t2 = in->ConstRow(kBorderLut[o - 2]) + rect.x0();
        row_t1 = in->ConstRow(kBorderLut[o - 1]) + rect.x0();
        row_b1 = in->ConstRow(kBorderLut[o + 1]) + rect.x0();
        row_b2 = in->ConstRow(kBorderLut[o + 2]) + rect.x0();
      } else if (img_y < kRadius) {
        if (img_y == 0) {
          row_t1 = row_m;
          row_t2 = row_b1;
        } else {
          JXL_DASSERT(img_y == 1);
          row_t2 = row_t1;
        }
      } else {
        JXL_DASSERT(img_y + kRadius >= in->ysize());
        if (img_y + 1 == in->ysize()) {
          row_b1 = row_m;
          row_b2 = row_t1;
        } else {
          JXL_DASSERT(img_y + 2 == in->ysize());
          row_b2 = row_b1;
        }
      }
    }

    const V wh0 = LoadDup128(d, weights->horz + 0 * 4);
    const V wh1 = LoadDup128(d, weights->horz + 1 * 4);
    const V wh2 = LoadDup128(d, weights->horz + 2 * 4);
    const V wv0 = LoadDup128(d, weights->vert + 0 * 4);
    const V wv1 = LoadDup128(d, weights->vert + 1 * 4);
    const V wv2 = LoadDup128(d, weights->vert + 2 * 4);
    const I ml1 = MirrorLanes<1>();
    const I ml2 = MirrorLanes<2>();

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
      const V conv0 = Mul(
          HorzConvolveLast<kSizeModN>(row_m, x, xsize, wh0, wh1, wh2, ml1, ml2),
          wv0);

      const V conv1t = HorzConvolveLast<kSizeModN>(row_t1, x, xsize, wh0, wh1,
                                                   wh2, ml1, ml2);
      const V conv1b = HorzConvolveLast<kSizeModN>(row_b1, x, xsize, wh0, wh1,
                                                   wh2, ml1, ml2);
      const V conv1 = MulAdd(Add(conv1t, conv1b), wv1, conv0);

      const V conv2t = HorzConvolveLast<kSizeModN>(row_t2, x, xsize, wh0, wh1,
                                                   wh2, ml1, ml2);
      const V conv2b = HorzConvolveLast<kSizeModN>(row_b2, x, xsize, wh0, wh1,
                                                   wh2, ml1, ml2);
      const V conv2 = MulAdd(Add(conv2t, conv2b), wv2, conv1);
      Store(conv2, d, row_out + x);
      x += Lanes(d);
    }

    // If mod = 0, the above vector was the last.
    if (kSizeModN != 0) {
      const float* JXL_RESTRICT rows[5] = {row_t2, row_t1, row_m, row_b1,
                                           row_b2};
      for (; x < xsize; ++x) {
        float mul = 0.0f;
        for (int64_t dy = -kRadius; dy <= kRadius; ++dy) {
          const float wy = weights->vert[std::abs(dy) * 4];
          const float* clamped_row = rows[dy + 2];
          for (int64_t dx = -kRadius; dx <= kRadius; ++dx) {
            const float wx = weights->horz[std::abs(dx) * 4];
            const int64_t clamped_x = Mirror(x + dx, xsize);
            mul += clamped_row[clamped_x] * wx * wy;
          }
        }
        row_out[x] = mul;
      }
    }
  }

 private:
  template <size_t kSizeModN>
  JXL_INLINE void RunRows() {
    // NB: borders are image-bound, not rect-bound.
    size_t ybegin = rect.y0();
    size_t yend = rect.y1();
    while (ybegin < yend && ybegin < kRadius) {
      ybegin++;
    }
    while (ybegin < yend && yend + kRadius > in->ysize()) {
      yend--;
    }
    if (ybegin > rect.y0()) {
      RunBorderRows<kSizeModN>(0, ybegin - rect.y0());
    }
    if (yend > ybegin) {
      RunInteriorRows<kSizeModN>(ybegin - rect.y0(), yend - rect.y0());
    }
    if (yend < rect.y1()) {
      RunBorderRows<kSizeModN>(yend - rect.y0(), rect.ysize());
    }
  }

  template <size_t kSizeModN>
  JXL_INLINE void RunBorderRows(const size_t ybegin, const size_t yend) {
    for (size_t y = ybegin; y < yend; ++y) {
      ConvolveRow<kSizeModN, true>(y);
    }
  }

  template <size_t kSizeModN>
  JXL_INLINE void RunInteriorRows(const size_t ybegin, const size_t yend) {
    const auto process_row = [&](const uint32_t y, size_t /*thread*/) HWY_ATTR {
      ConvolveRow<kSizeModN, false>(y);
      return true;
    };
    Status status = RunOnPool(pool, ybegin, yend, ThreadPool::NoInit,
                              process_row, "Convolve");
    JXL_DASSERT(status);
    (void)status;
  }

  // Returns IndicesFromVec(d, indices) such that TableLookupLanes on the
  // rightmost unaligned vector (rightmost sample in its most-significant lane)
  // returns the mirrored values, with the mirror outside the last valid sample.
  template <int M>
  static JXL_INLINE I MirrorLanes() {
    static_assert(M >= 1 && M <= 2, "Only M in range {1..2} is supported");
    D d;
    DI32 di32;
    const VI32 up = Min(Iota(di32, M), Set(di32, Lanes(d) - 1));
    const VI32 down =
        Max(Iota(di32, M - static_cast<int>(Lanes(d))), Zero(di32));
    return IndicesFromVec(d, Sub(up, down));
  }

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
  static JXL_MAYBE_INLINE V HorzConvolveLast(
      const float* const JXL_RESTRICT row, const int64_t x, const int64_t xsize,
      const V wh0, const V wh1, const V wh2, const I ml1, const I ml2) {
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
    (void)ml1;
    (void)ml2;
#else
    const size_t N = Lanes(d);
    if (kSizeModN == 0) {
      r2 = TableLookupLanes(c, ml2);
      r1 = TableLookupLanes(c, ml1);
    } else {  // == 1
      const auto last = LoadU(d, row + xsize - N);
      r2 = TableLookupLanes(last, ml1);
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

  const ImageF* in;
  const Rect rect;
  const WeightsSeparable5* weights;
  ThreadPool* pool;
  ImageF* out;
};

Status Separable5(const ImageF& in, const Rect& rect,
                  const WeightsSeparable5& weights, ThreadPool* pool,
                  ImageF* out) {
  Separable5Impl impl(&in, rect, &weights, pool, out);
  return impl.Run();
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
