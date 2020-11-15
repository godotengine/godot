// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_LINUX_ADDRESS_TYPES_H_
#define CRASHPAD_UTIL_LINUX_ADDRESS_TYPES_H_

#include <stdint.h>

namespace crashpad {

//! \brief Type used to represent an address in a process, potentially across
//!     bitness.
using LinuxVMAddress = uint64_t;

//! \brief Type used to represent the size of a memory range (with a
//!     LinuxVMAddress), potentially across bitness.
using LinuxVMSize = uint64_t;

//! \brief Type used to represent an offset from a LinuxVMAddress, potentially
//!     across bitness.
using LinuxVMOffset = int64_t;

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_LINUX_ADDRESS_TYPES_H_
