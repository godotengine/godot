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

#ifndef CRASHPAD_UTIL_MISC_CAPTURE_CONTEXT_H_
#define CRASHPAD_UTIL_MISC_CAPTURE_CONTEXT_H_

#include "build/build_config.h"

#if defined(OS_MACOSX)
#include <mach/mach.h>
#elif defined(OS_WIN)
#include <windows.h>
#elif defined(OS_LINUX) || defined(OS_ANDROID)
#include <ucontext.h>
#elif defined(OS_FUCHSIA)
#include <signal.h>
#endif  // OS_MACOSX

namespace crashpad {

#if defined(OS_MACOSX)
#if defined(ARCH_CPU_X86_FAMILY)
using NativeCPUContext = x86_thread_state;
#endif
#elif defined(OS_WIN)
using NativeCPUContext = CONTEXT;
#elif defined(OS_LINUX) || defined(OS_ANDROID) || defined(OS_FUCHSIA)
using NativeCPUContext = ucontext_t;
#endif  // OS_MACOSX

//! \brief Saves the CPU context.
//!
//! The CPU context will be captured as accurately and completely as possible,
//! containing an atomic snapshot at the point of this function’s return. This
//! function does not modify any registers.
//!
//! This function is a replacement for `RtlCaptureContext()` and `getcontext()`
//! which contain bugs and/or limitations.
//!
//! On 32-bit x86, `RtlCaptureContext()` requires that `ebp` be used as a frame
//! pointer, and returns `ebp`, `esp`, and `eip` out of sync with the other
//! registers. Both the 32-bit x86 and 64-bit x86_64 versions of
//! `RtlCaptureContext()` capture only the state of the integer registers,
//! ignoring floating-point and vector state.
//!
//! \param[out] cpu_context The structure to store the context in.
//!
//! \note The ABI may require that this function's argument is passed by
//!     register, preventing this fuction from saving the original value of that
//!     register. This occurs in the following circumstances:
//!
//!     OS                  | Architecture | Register
//!     --------------------|--------------|---------
//!     Win                 | x86_64       | `%%rcx`
//!     macOS/Linux/Fuchsia | x86_64       | `%%rdi`
//!     Linux               | ARM/ARM64    | `r0`/`x0`
//!     Linux               | MIPS/MIPS64  | `$a0`
//!
//!     Additionally, the value `LR` on ARM/ARM64 will be the return address of
//!     this function.
//!
//!     If the value of these register prior to calling this function are needed
//!     they must be obtained separately prior to calling this function. For
//!     example:
//!     \code
//!       uint64_t rdi;
//!       asm("movq %%rdi, %0" : "=m"(rdi));
//!     \endcode
void CaptureContext(NativeCPUContext* cpu_context);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_CAPTURE_CONTEXT_H_
