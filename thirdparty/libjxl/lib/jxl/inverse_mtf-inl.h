// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// SIMDified inverse-move-to-front transform.

#if defined(LIB_JXL_INVERSE_MTF_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_INVERSE_MTF_INL_H_
#undef LIB_JXL_INVERSE_MTF_INL_H_
#else
#define LIB_JXL_INVERSE_MTF_INL_H_
#endif

#include <hwy/highway.h>

#include "lib/jxl/base/sanitizers.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::FirstN;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::Load;
using hwy::HWY_NAMESPACE::LoadU;
using hwy::HWY_NAMESPACE::StoreU;

inline void MoveToFront(uint8_t* v, uint8_t index) {
  uint8_t value = v[index];
  uint8_t i = index;
  if (i < 4) {
    for (; i; --i) v[i] = v[i - 1];
  } else {
    const HWY_CAPPED(uint8_t, 64) d;
    int tail = i & (Lanes(d) - 1);
    if (tail) {
      i -= tail;
      const auto vec = Load(d, v + i);
      const auto prev = LoadU(d, v + i + 1);
      StoreU(IfThenElse(FirstN(d, tail), vec, prev), d, v + i + 1);
    }
    while (i) {
      i -= Lanes(d);
      const auto vec = Load(d, v + i);
      StoreU(vec, d, v + i + 1);
    }
  }
  v[0] = value;
}

inline void InverseMoveToFrontTransform(uint8_t* v, int v_len) {
  HWY_ALIGN uint8_t mtf[256 + 64];
  int i;
  for (i = 0; i < 256; ++i) {
    mtf[i] = static_cast<uint8_t>(i);
  }
#if JXL_MEMORY_SANITIZER
  const HWY_CAPPED(uint8_t, 64) d;
  for (size_t j = 0; j < Lanes(d); ++j) {
    mtf[256 + j] = 0;
  }
#endif  // JXL_MEMORY_SANITIZER
  for (i = 0; i < v_len; ++i) {
    uint8_t index = v[i];
    v[i] = mtf[index];
    if (index) MoveToFront(mtf, index);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // LIB_JXL_INVERSE_MTF_INL_H_

#if HWY_ONCE
#ifndef INVERSE_MTF_ONCE
#define INVERSE_MTF_ONCE

namespace jxl {
inline void InverseMoveToFrontTransform(uint8_t* v, int v_len) {
  HWY_STATIC_DISPATCH(InverseMoveToFrontTransform)(v, v_len);
}
}  // namespace jxl

#endif  // INVERSE_MTF_ONCE
#endif  // HWY_ONCE
