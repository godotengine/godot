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

#ifndef CRASHPAD_UTIL_MISC_ADDRESS_TYPES_H_
#define CRASHPAD_UTIL_MISC_ADDRESS_TYPES_H_

#include <stdint.h>

#include <type_traits>

#include "build/build_config.h"

#if defined(OS_MACOSX)
#include <mach/mach_types.h>
#elif defined(OS_WIN)
#include "util/win/address_types.h"
#elif defined(OS_LINUX) || defined(OS_ANDROID)
#include "util/linux/address_types.h"
#elif defined(OS_FUCHSIA)
#include <zircon/types.h>
#else
#error "Unhandled OS type"
#endif

namespace crashpad {

#if DOXYGEN

//! \brief Type used to represent an address in a process, potentially across
//!     bitness.
using VMAddress = uint64_t;

//! \brief Type used to represent the size of a memory range (with a
//!     VMAddress), potentially across bitness.
using VMSize = uint64_t;

#elif defined(OS_MACOSX)

using VMAddress = mach_vm_address_t;
using VMSize = mach_vm_size_t;

#elif defined(OS_WIN)

using VMAddress = WinVMAddress;
using VMSize = WinVMSize;

#elif defined(OS_LINUX) || defined(OS_ANDROID)

using VMAddress = LinuxVMAddress;
using VMSize = LinuxVMSize;

#elif defined(OS_FUCHSIA)

using VMAddress = zx_vaddr_t;
using VMSize = size_t;

#endif

//! \brief Type used to represent an offset from a VMAddress, potentially
//!     across bitness.
using VMOffset = std::make_signed<VMSize>::type;

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_ADDRESS_TYPES_H_
