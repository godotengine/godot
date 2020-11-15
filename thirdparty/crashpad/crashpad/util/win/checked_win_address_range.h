// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_WIN_CHECKED_WIN_ADDRESS_RANGE_H_
#define CRASHPAD_UTIL_WIN_CHECKED_WIN_ADDRESS_RANGE_H_

#include "util/numeric/checked_address_range.h"
#include "util/win/address_types.h"

namespace crashpad {

//! \brief Ensures that a range, composed of a base and a size, does not
//!     overflow the pointer type of the process it describes a range in.
//!
//! This class checks bases of type WinVMAddress and sizes of type WinVMSize
//! against a process whose pointer type is either 32 or 64 bits wide.
//!
//! Aside from varying the overall range on the basis of a process' pointer type
//! width, this class functions very similarly to CheckedRange.
using CheckedWinAddressRange =
    internal::CheckedAddressRangeGeneric<WinVMAddress, WinVMSize>;

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_CHECKED_WIN_ADDRESS_RANGE_H_
