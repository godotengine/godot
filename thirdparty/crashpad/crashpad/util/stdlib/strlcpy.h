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

#ifndef CRASHPAD_UTIL_STDLIB_STRLCPY_H_
#define CRASHPAD_UTIL_STDLIB_STRLCPY_H_

#include <sys/types.h>

#include "base/strings/string16.h"

namespace crashpad {

//! \brief Copy a `NUL`-terminated char16-based string to a fixed-size buffer.
//!
//! This function behaves identically to `strlcpy()`, but it operates on char16
//! data instead of `char` data. It copies the `NUL`-terminated string in the
//! buffer beginning at \a source to the buffer of size \a length at \a
//! destination, ensuring that the destination buffer is `NUL`-terminated. No
//! data will be written outside of the \a destination buffer, but if \a length
//! is smaller than the length of the string at \a source, the string will be
//! truncated.
//!
//! \param[out] destination A pointer to a buffer of at least size \a length
//!     char16 units (not bytes). The string will be copied to this buffer,
//!     possibly with truncation, and `NUL`-terminated. Nothing will be written
//!     following the `NUL` terminator.
//! \param[in] source A pointer to a `NUL`-terminated string of char16 data. The
//!     `NUL` terminator must be a `NUL` value in a char16 unit, not just a
//!     single `NUL` byte.
//! \param[in] length The length of the \a destination buffer in char16 units,
//!     not bytes. A maximum of \a `length - 1` char16 units from \a source will
//!     be copied to \a destination.
//!
//! \return The length of the \a source string in char16 units, not including
//!     its `NUL` terminator. When truncation occurs, the return value will be
//!     equal to or greater than than the \a length parameter.
size_t c16lcpy(base::char16* destination,
               const base::char16* source,
               size_t length);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_STDLIB_STRLCPY_H_
