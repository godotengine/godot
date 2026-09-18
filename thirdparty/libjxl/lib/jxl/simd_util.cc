// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/simd_util.h"

#include <cstddef>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/simd_util.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

size_t MaxVectorSize() {
  HWY_FULL(float) df;
  return Lanes(df) * sizeof(float);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(MaxVectorSize);

size_t MaxVectorSize() {
  // Ideally HWY framework should provide us this value.
  // Less than ideal is to check all available targets and choose maximal.
  // As for now, we just ask current active target, assuming it won't change.
  return HWY_DYNAMIC_DISPATCH(MaxVectorSize)();
}

}  // namespace jxl
#endif
