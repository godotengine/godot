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

#ifndef CRASHPAD_UTIL_WIN_HANDLE_TO_INT_H_
#define CRASHPAD_UTIL_WIN_HANDLE_TO_INT_H_

#include <windows.h>

namespace crashpad {

//! \brief Converts a `HANDLE` to an `int`.
//!
//! `HANDLE` is a `typedef` for `void *`, but kernel `HANDLE` values aren’t
//! pointers to anything. Only 32 bits of kernel `HANDLE`s are significant, even
//! in 64-bit processes on 64-bit operating systems. See <a
//! href="https://msdn.microsoft.com/library/aa384203.aspx">Interprocess
//! Communication Between 32-bit and 64-bit Applications</a>.
//!
//! This function safely converts a kernel `HANDLE` to an `int` similarly to a
//! cast operation. It checks that the operation can be performed safely, and
//! aborts execution if it cannot.
//!
//! \param[in] handle The kernel `HANDLE` to convert.
//!
//! \return An equivalent `int`, truncated (if necessary) from \a handle. If
//!     truncation would have resulted in an `int` that could not be converted
//!     back to \a handle, aborts execution.
//!
//! \sa IntToHandle()
int HandleToInt(HANDLE handle);

//! \brief Converts an `int` to an `HANDLE`.
//!
//! `HANDLE` is a `typedef` for `void *`, but kernel `HANDLE` values aren’t
//! pointers to anything. Only 32 bits of kernel `HANDLE`s are significant, even
//! in 64-bit processes on 64-bit operating systems. See <a
//! href="https://msdn.microsoft.com/library/aa384203.aspx">Interprocess
//! Communication Between 32-bit and 64-bit Applications</a>.
//!
//! This function safely convert an `int` to a kernel `HANDLE` similarly to a
//! cast operation.
//!
//! \param[in] handle_int The `int` to convert. This must have been produced by
//!     HandleToInt(), possibly in a different process.
//!
//! \return An equivalent kernel `HANDLE`, sign-extended (if necessary) from \a
//!     handle_int.
//!
//! \sa HandleToInt()
HANDLE IntToHandle(int handle_int);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_HANDLE_TO_INT_H_
