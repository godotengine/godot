// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dct_scales.h"

#include "lib/jxl/base/compiler_specific.h"

namespace jxl {

#if JXL_CXX_LANG < JXL_CXX_17
constexpr float DCTResampleScales<1, 8>::kScales[];
constexpr float DCTResampleScales<2, 16>::kScales[];
constexpr float DCTResampleScales<4, 32>::kScales[];
constexpr float DCTResampleScales<8, 64>::kScales[];
constexpr float DCTResampleScales<16, 128>::kScales[];
constexpr float DCTResampleScales<32, 256>::kScales[];
constexpr float DCTResampleScales<8, 1>::kScales[];
constexpr float DCTResampleScales<16, 2>::kScales[];
constexpr float DCTResampleScales<32, 4>::kScales[];
constexpr float DCTResampleScales<64, 8>::kScales[];
constexpr float DCTResampleScales<128, 16>::kScales[];
constexpr float DCTResampleScales<256, 32>::kScales[];
constexpr float WcMultipliers<4>::kMultipliers[];
constexpr float WcMultipliers<8>::kMultipliers[];
constexpr float WcMultipliers<16>::kMultipliers[];
constexpr float WcMultipliers<32>::kMultipliers[];
constexpr float WcMultipliers<64>::kMultipliers[];
constexpr float WcMultipliers<128>::kMultipliers[];
constexpr float WcMultipliers<256>::kMultipliers[];
#endif

}  // namespace jxl
