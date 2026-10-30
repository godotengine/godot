// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"

#if defined(LIB_JXL_CONVOLVE_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_CONVOLVE_INL_H_
#undef LIB_JXL_CONVOLVE_INL_H_
#else
#define LIB_JXL_CONVOLVE_INL_H_
#endif

#include <hwy/highway.h>

#if HWY_TARGET <= (1 << HWY_HIGHEST_TARGET_BIT_X86)
#include <xmmintrin.h>
#endif

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
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
};

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // LIB_JXL_CONVOLVE_INL_H_
