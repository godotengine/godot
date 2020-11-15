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

#ifndef CRASHPAD_UTIL_STDLIB_STRNLEN_H_
#define CRASHPAD_UTIL_STDLIB_STRNLEN_H_

#include <string.h>
#include <sys/types.h>

#include "build/build_config.h"

#if defined(OS_MACOSX)
#include <AvailabilityMacros.h>
#endif

namespace crashpad {

//! \brief Returns the length of a string, not to exceed a maximum.
//!
//! \param[in] string The string whose length is to be calculated.
//! \param[in] max_length The maximum length to return.
//!
//! \return The length of \a string, determined as the index of the first `NUL`
//!     byte found, not exceeding \a max_length.
//!
//! \note This function is provided because it was introduced in POSIX.1-2008,
//!     and not all systems’ standard libraries provide an implementation.
size_t strnlen(const char* string, size_t max_length);

#if !defined(OS_MACOSX) || \
    MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_7
inline size_t strnlen(const char* string, size_t max_length) {
  return ::strnlen(string, max_length);
}
#endif

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_STDLIB_STRNLEN_H_
