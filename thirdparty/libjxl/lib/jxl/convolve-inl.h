// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#if defined(LIB_JXL_CONVOLVE_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_CONVOLVE_INL_H_
#undef LIB_JXL_CONVOLVE_INL_H_
#else
#define LIB_JXL_CONVOLVE_INL_H_
#endif

#include <hwy/highway.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image_ops.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Broadcast;
#if HWY_TARGET != HWY_SCALAR
using hwy::HWY_NAMESPACE::CombineShiftRightBytes;
#endif
using hwy::HWY_NAMESPACE::TableLookupLanes;
using hwy::HWY_NAMESPACE::Vec;

// Synthesizes left/right neighbors from a vector of center pixels.
class Neighbors {
 public:
  using D = HWY_CAPPED(float, 16);
  using V = Vec<D>;

  // Returns l[i] == c[Mirror(i - 1)].
  HWY_INLINE HWY_MAYBE_UNUSED static V FirstL1(const V c) {
#if HWY_CAP_GE256
    const D d;
    HWY_ALIGN constexpr int32_t lanes[16] = {0, 0, 1, 2,  3,  4,  5,  6,
                                             7, 8, 9, 10, 11, 12, 13, 14};
    const auto indices = SetTableIndices(d, lanes);
    // c = PONM'LKJI
    return TableLookupLanes(c, indices);  // ONML'KJII
#elif HWY_TARGET == HWY_SCALAR
    return c;  // Same (the first mirrored value is the last valid one)
#else  // 128 bit
    // c = LKJI
#if HWY_TARGET <= (1 << HWY_HIGHEST_TARGET_BIT_X86)
    return V{_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(2, 1, 0, 0))};  // KJII
#else
    const D d;
    // TODO(deymo): Figure out if this can be optimized using a single vsri
    // instruction to convert LKJI to KJII.
    HWY_ALIGN constexpr int lanes[4] = {0, 0, 1, 2};  // KJII
    const auto indices = SetTableIndices(d, lanes);
    return TableLookupLanes(c, indices);
#endif
#endif
  }

  // Returns l[i] == c[Mirror(i - 2)].
  HWY_INLINE HWY_MAYBE_UNUSED static V FirstL2(const V c) {
#if HWY_CAP_GE256
    const D d;
    HWY_ALIGN constexpr int32_t lanes[16] = {1, 0, 0, 1, 2,  3,  4,  5,
                                             6, 7, 8, 9, 10, 11, 12, 13};
    const auto indices = SetTableIndices(d, lanes);
    // c = PONM'LKJI
    return TableLookupLanes(c, indices);  // NMLK'JIIJ
#elif HWY_TARGET == HWY_SCALAR
    const D d;
    JXL_DEBUG_ABORT("Unsupported");
    return Zero(d);
#else  // 128 bit
    // c = LKJI
#if HWY_TARGET <= (1 << HWY_HIGHEST_TARGET_BIT_X86)
    return V{_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(1, 0, 0, 1))};  // JIIJ
#else
    const D d;
    HWY_ALIGN constexpr int lanes[4] = {1, 0, 0, 1};  // JIIJ
    const auto indices = SetTableIndices(d, lanes);
    return TableLookupLanes(c, indices);
#endif
#endif
  }

  // Returns l[i] == c[Mirror(i - 3)].
  HWY_INLINE HWY_MAYBE_UNUSED static V FirstL3(const V c) {
#if HWY_CAP_GE256
    const D d;
    HWY_ALIGN constexpr int32_t lanes[16] = {2, 1, 0, 0, 1, 2,  3,  4,
                                             5, 6, 7, 8, 9, 10, 11, 12};
    const auto indices = SetTableIndices(d, lanes);
    // c = PONM'LKJI
    return TableLookupLanes(c, indices);  // MLKJ'IIJK
#elif HWY_TARGET == HWY_SCALAR
    const D d;
    JXL_DEBUG_ABORT("Unsipported");
    return Zero(d);
#else  // 128 bit
    // c = LKJI
#if HWY_TARGET <= (1 << HWY_HIGHEST_TARGET_BIT_X86)
    return V{_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(0, 0, 1, 2))};  // IIJK
#else
    const D d;
    HWY_ALIGN constexpr int lanes[4] = {2, 1, 0, 0};  // IIJK
    const auto indices = SetTableIndices(d, lanes);
    return TableLookupLanes(c, indices);
#endif
#endif
  }
};

#if HWY_TARGET != HWY_SCALAR

// Returns indices for SetTableIndices such that TableLookupLanes on the
// rightmost unaligned vector (rightmost sample in its most-significant lane)
// returns the mirrored values, with the mirror outside the last valid sample.
inline const int32_t* MirrorLanes(const size_t mod) {
  const HWY_CAPPED(float, 16) d;
  constexpr size_t kN = MaxLanes(d);
  // typo:off
  // For mod = `image width mod 16` 0..15:
  // last full vec     mirrored (mem order)  loadedVec  mirrorVec  idxVec
  // 0123456789abcdef| fedcba9876543210      fed..210   012..def   012..def
  // 0123456789abcdef|0 0fedcba98765432      0fe..321   234..f00   123..eff
  // 0123456789abcdef|01 10fedcba987654      10f..432   456..110   234..ffe
  // 0123456789abcdef|012 210fedcba9876      210..543   67..2210   34..ffed
  // 0123456789abcdef|0123 3210fedcba98      321..654   8..33210   4..ffedc
  // 0123456789abcdef|01234 43210fedcba
  // 0123456789abcdef|012345 543210fedc
  // 0123456789abcdef|0123456 6543210fe
  // 0123456789abcdef|01234567 76543210
  // 0123456789abcdef|012345678 8765432
  // 0123456789abcdef|0123456789 987654
  // 0123456789abcdef|0123456789A A9876
  // 0123456789abcdef|0123456789AB BA98
  // 0123456789abcdef|0123456789ABC CBA
  // 0123456789abcdef|0123456789ABCD DC
  // 0123456789abcdef|0123456789ABCDE E      EDC..10f   EED..210   ffe..321
  // typo:on
#if HWY_CAP_GE512
  HWY_ALIGN static constexpr int32_t idx_lanes[2 * kN - 1] = {
      1,  2,  3,  4,  5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15,  //
      14, 13, 12, 11, 10, 9, 8, 7, 6, 5,  4,  3,  2,  1,  0};
#elif HWY_CAP_GE256
  HWY_ALIGN static constexpr int32_t idx_lanes[2 * kN - 1] = {
      1, 2, 3, 4, 5, 6, 7, 7,  //
      6, 5, 4, 3, 2, 1, 0};
#else  // 128-bit
  HWY_ALIGN static constexpr int32_t idx_lanes[2 * kN - 1] = {1, 2, 3, 3,  //
                                                              2, 1, 0};
#endif
  return idx_lanes + kN - 1 - mod;
}

#endif  // HWY_TARGET != HWY_SCALAR

// Single entry point for convolution.
// "Strategy" (Direct*/Separable*) decides kernel size and how to evaluate it.
template <class Strategy>
class ConvolveT {
  static constexpr int64_t kRadius = Strategy::kRadius;
  using Simd = HWY_CAPPED(float, 16);

 public:
  static size_t MinWidth() {
#if HWY_TARGET == HWY_SCALAR
    // First/Last use mirrored loads of up to +/- kRadius.
    return 2 * kRadius;
#else
    return Lanes(Simd()) + kRadius;
#endif
  }

  // "Image" is ImageF or Image3F.
  template <class Image, class Weights>
  static void Run(const Image& in, const Rect& rect, const Weights& weights,
                  ThreadPool* pool, Image* out) {
    JXL_DASSERT(SameSize(rect, *out));
    JXL_DASSERT(rect.xsize() >= MinWidth());

    static_assert(static_cast<int64_t>(kRadius) <= 3,
                  "Must handle [0, kRadius) and >= kRadius");
    switch (rect.xsize() % Lanes(Simd())) {
      case 0:
        return RunRows<0>(in, rect, weights, pool, out);
      case 1:
        return RunRows<1>(in, rect, weights, pool, out);
      case 2:
        return RunRows<2>(in, rect, weights, pool, out);
      default:
        return RunRows<3>(in, rect, weights, pool, out);
    }
  }

 private:
  template <size_t kSizeModN, class WrapRow, class Weights>
  static JXL_INLINE void RunRow(const float* JXL_RESTRICT in,
                                const size_t xsize, const int64_t stride,
                                const WrapRow& wrap_row, const Weights& weights,
                                float* JXL_RESTRICT out) {
    Strategy::template ConvolveRow<kSizeModN>(in, xsize, stride, wrap_row,
                                              weights, out);
  }

  template <size_t kSizeModN, class Weights>
  static JXL_INLINE void RunBorderRows(const ImageF& in, const Rect& rect,
                                       const int64_t ybegin, const int64_t yend,
                                       const Weights& weights, ImageF* out) {
    const int64_t stride = in.PixelsPerRow();
    const WrapRowMirror wrap_row(in, rect.ysize());
    for (int64_t y = ybegin; y < yend; ++y) {
      RunRow<kSizeModN>(rect.ConstRow(in, y), rect.xsize(), stride, wrap_row,
                        weights, out->Row(y));
    }
  }

  // Image3F.
  template <size_t kSizeModN, class Weights>
  static JXL_INLINE void RunBorderRows(const Image3F& in, const Rect& rect,
                                       const int64_t ybegin, const int64_t yend,
                                       const Weights& weights, Image3F* out) {
    const int64_t stride = in.PixelsPerRow();
    for (int64_t y = ybegin; y < yend; ++y) {
      for (size_t c = 0; c < 3; ++c) {
        const WrapRowMirror wrap_row(in.Plane(c), rect.ysize());
        RunRow<kSizeModN>(rect.ConstPlaneRow(in, c, y), rect.xsize(), stride,
                          wrap_row, weights, out->PlaneRow(c, y));
      }
    }
  }

  template <size_t kSizeModN, class Weights>
  static JXL_INLINE void RunInteriorRows(const ImageF& in, const Rect& rect,
                                         const int64_t ybegin,
                                         const int64_t yend,
                                         const Weights& weights,
                                         ThreadPool* pool, ImageF* out) {
    const int64_t stride = in.PixelsPerRow();
    const auto process_row = [&](const uint32_t y, size_t /*thread*/) HWY_ATTR {
      RunRow<kSizeModN>(rect.ConstRow(in, y), rect.xsize(), stride,
                        WrapRowUnchanged(), weights, out->Row(y));
      return true;
    };
    Status status = RunOnPool(pool, ybegin, yend, ThreadPool::NoInit,
                              process_row, "Convolve");
    (void)status;
    JXL_DASSERT(status);
  }

  // Image3F.
  template <size_t kSizeModN, class Weights>
  static JXL_INLINE void RunInteriorRows(const Image3F& in, const Rect& rect,
                                         const int64_t ybegin,
                                         const int64_t yend,
                                         const Weights& weights,
                                         ThreadPool* pool, Image3F* out) {
    const int64_t stride = in.PixelsPerRow();
    const auto process_row = [&](const uint32_t y, size_t /*thread*/) HWY_ATTR {
      for (size_t c = 0; c < 3; ++c) {
        RunRow<kSizeModN>(rect.ConstPlaneRow(in, c, y), rect.xsize(), stride,
                          WrapRowUnchanged(), weights, out->PlaneRow(c, y));
      }
      return true;
    };
    Status status = RunOnPool(pool, ybegin, yend, ThreadPool::NoInit,
                              process_row, "Convolve3");
    (void)status;
    JXL_DASSERT(status);
  }

  template <size_t kSizeModN, class Image, class Weights>
  static JXL_INLINE void RunRows(const Image& in, const Rect& rect,
                                 const Weights& weights, ThreadPool* pool,
                                 Image* out) {
    const int64_t ysize = rect.ysize();
    RunBorderRows<kSizeModN>(in, rect, 0,
                             std::min(static_cast<int64_t>(kRadius), ysize),
                             weights, out);
    if (ysize > 2 * static_cast<int64_t>(kRadius)) {
      RunInteriorRows<kSizeModN>(in, rect, static_cast<int64_t>(kRadius),
                                 ysize - static_cast<int64_t>(kRadius), weights,
                                 pool, out);
    }
    if (ysize > static_cast<int64_t>(kRadius)) {
      RunBorderRows<kSizeModN>(in, rect, ysize - static_cast<int64_t>(kRadius),
                               ysize, weights, out);
    }
  }
};

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // LIB_JXL_CONVOLVE_INL_H_
