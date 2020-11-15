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

#ifndef CRASHPAD_SNAPSHOT_WIN_CPU_CONTEXT_WIN_H_
#define CRASHPAD_SNAPSHOT_WIN_CPU_CONTEXT_WIN_H_

#include <windows.h>

#include "build/build_config.h"

namespace crashpad {

struct CPUContextX86;
struct CPUContextX86_64;

#if defined(ARCH_CPU_64_BITS) || DOXYGEN

//! \brief Initializes a CPUContextX86 structure from a native context structure
//!     on Windows.
void InitializeX86Context(const WOW64_CONTEXT& context, CPUContextX86* out);

//! \brief Initializes a CPUContextX86_64 structure from a native context
//!     structure on Windows.
void InitializeX64Context(const CONTEXT& context, CPUContextX86_64* out);

#else  // ARCH_CPU_64_BITS

//! \brief Initializes a CPUContextX86 structure from a native context structure
//!     on Windows.
void InitializeX86Context(const CONTEXT& context, CPUContextX86* out);

#endif  // ARCH_CPU_64_BITS

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_WIN_CPU_CONTEXT_WIN_H_
