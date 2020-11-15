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

#ifndef CRASHPAD_UTIL_WIN_SAFE_TERMINATE_PROCESS_H_
#define CRASHPAD_UTIL_WIN_SAFE_TERMINATE_PROCESS_H_

#include <windows.h>

#include "build/build_config.h"

namespace crashpad {

//! \brief Calls `TerminateProcess()`.
//!
//! `TerminateProcess()` has been observed in the wild as being patched badly on
//! 32-bit x86: it’s patched with code adhering to the `cdecl` (caller clean-up)
//! convention, although it’s supposed to be `stdcall` (callee clean-up). The
//! mix-up means that neither caller nor callee perform parameter clean-up from
//! the stack, causing the stack pointer to have an unexpected value on return
//! from the patched function. This typically results in a crash shortly
//! thereafter. See <a href="https://crashpad.chromium.org/bug/179">Crashpad bug
//! 179</a>.
//!
//! On 32-bit x86, this replacement function calls `TerminateProcess()` without
//! making any assumptions about the stack pointer on its return. As such, it’s
//! compatible with the badly patched `cdecl` version as well as the native
//! `stdcall` version (and other less badly patched versions).
//!
//! Elsewhere, this function calls `TerminateProcess()` directly without any
//! additional fanfare.
//!
//! Call this function instead of `TerminateProcess()` anywhere that
//! `TerminateProcess()` would normally be called.
bool SafeTerminateProcess(HANDLE process, UINT exit_code);

#if !defined(ARCH_CPU_X86)
inline bool SafeTerminateProcess(HANDLE process, UINT exit_code) {
  return TerminateProcess(process, exit_code) != FALSE;
}
#endif  // !ARCH_CPU_X86

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_SAFE_TERMINATE_PROCESS_H_
