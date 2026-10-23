// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_ANS_SIMD_H_
#define LIB_JXL_ENC_ANS_SIMD_H_

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/memory_manager_internal.h"

namespace jxl {

// Returns "extra_bits" sum and puts tokens into `tokens`.
uint32_t EstimateTokenCost(uint32_t* JXL_RESTRICT values, size_t len,
                           HybridUintConfig cfg, AlignedMemory& tokens);

}  // namespace jxl

#endif  // LIB_JXL_ENC_ANS_SIMD_H_
