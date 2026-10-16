// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/simd_util.h"

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/simd_util.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::Iota;
using hwy::HWY_NAMESPACE::LoadU;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::MaxOfLanes;
using hwy::HWY_NAMESPACE::Set;

size_t MaxVectorSize() {
  HWY_FULL(float) df;
  return Lanes(df) * sizeof(float);
}

uint32_t MaxValue(uint32_t* JXL_RESTRICT data, size_t len) {
  HWY_FULL(uint32_t) du;
  size_t last_full = Lanes(du) * (len / Lanes(du));
  auto max = Set(du, 0);
  for (size_t i = 0; i < last_full; i += Lanes(du)) {
    max = Max(max, LoadU(du, data + i));
  }
  if (last_full < len) {
    const auto stop = Set(du, len);
    const auto fence = Iota(du, last_full);
    const auto take = Lt(fence, stop);
    max = Max(max, IfThenElseZero(take, LoadU(du, data + last_full)));
  }
  return GetLane(MaxOfLanes(du, max));
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(MaxVectorSize);
HWY_EXPORT(MaxValue);

size_t MaxVectorSize() {
  // Ideally HWY framework should provide us this value.
  // Less than ideal is to check all available targets and choose maximal.
  // As for now, we just ask current active target, assuming it won't change.
  return HWY_DYNAMIC_DISPATCH(MaxVectorSize)();
}

uint32_t MaxValue(uint32_t* JXL_RESTRICT data, size_t len) {
  return HWY_DYNAMIC_DISPATCH(MaxValue)(data, len);
}

}  // namespace jxl
#endif
