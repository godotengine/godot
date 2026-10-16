// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_SIMD_UTIL_H_
#define LIB_JXL_SIMD_UTIL_H_

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"

namespace jxl {

// Maximal vector size in bytes.
size_t MaxVectorSize();

uint32_t MaxValue(uint32_t* JXL_RESTRICT data, size_t len);

}  // namespace jxl

#endif  // LIB_JXL_SIMD_UTIL_H_
