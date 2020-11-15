// Copyright 2014 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CRASHPAD_UTIL_NUMERIC_INT128_H_
#define CRASHPAD_UTIL_NUMERIC_INT128_H_

#include <stdint.h>

#include "build/build_config.h"

namespace crashpad {

//! \brief Stores a 128-bit quantity.
//!
//! This structure is organized so that 128-bit quantities are laid out in
//! memory according to the system’s natural byte order. If a system offers a
//! native 128-bit type, it should be possible to bit_cast<> between that type
//! and this one.
//!
//! This structure is designed to have the same layout, although not the same
//! field names, as the Windows SDK’s `M128A` type from `<winnt.h>`. It is
//! provided here instead of in `compat` because it is useful outside of the
//! scope of data structures defined by the Windows SDK.
struct uint128_struct {
#if defined(ARCH_CPU_LITTLE_ENDIAN) || DOXYGEN
  //! \brief The low 64 bits of the 128-bit quantity.
  uint64_t lo;

  //! \brief The high 64 bits of the 128-bit quantity.
  uint64_t hi;
#else
  uint64_t hi;
  uint64_t lo;
#endif
};

static_assert(sizeof(uint128_struct) == 16, "uint128 must be 16 bytes");

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_NUMERIC_INT128_H_
