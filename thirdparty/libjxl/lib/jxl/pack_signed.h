// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_PACK_H_
#define LIB_JXL_PACK_H_

// Pack/UnpackSigned utilities.

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"

namespace jxl {
// Encodes non-negative (X) into (2 * X), negative (-X) into (2 * X - 1)
constexpr uint32_t PackSigned(int32_t value)
    JXL_NO_SANITIZE("unsigned-integer-overflow") {
  return (static_cast<uint32_t>(value) << 1) ^
         ((static_cast<uint32_t>(~value) >> 31) - 1);
}

// Reverse to PackSigned, i.e. UnpackSigned(PackSigned(X)) == X.
// (((~value) & 1) - 1) is either 0 or 0xFF...FF and it will have an expected
// unsigned-integer-overflow.
// NB: semantically `value` should have type `uint32_t`, but for efficiency and
// convenience its type is `size_t` (i.e. `uint32_t` or `uint64_t`).
constexpr int32_t UnpackSigned(size_t value)
    JXL_NO_SANITIZE("unsigned-integer-overflow") {
  // TODO(Ivan): fails in C++11 mode, restore with a guard?
  // JXL_DASSERT((value & 0xFFFFFFFF) == value);  // no-op in 32-bit build
  return static_cast<int32_t>((value >> 1) ^ (((~value) & 1) - 1));
}

}  // namespace jxl

#endif  // LIB_JXL_PACK_H_
