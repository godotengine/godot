// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_MODULAR_SIMD_H_
#define LIB_JXL_ENC_MODULAR_SIMD_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/status.h"
#include "lib/jxl/modular/modular_image.h"

namespace jxl {

StatusOr<float> EstimateCost(const Image& img);

namespace estimate_cost_detail {
constexpr size_t kLastThreshold = 501;
constexpr size_t kLastCtx = 16;
const std::array<uint8_t, kLastThreshold>& ContextMap();
}  // namespace estimate_cost_detail

}  // namespace jxl

#endif  // LIB_JXL_ENC_MODULAR_SIMD_H_
