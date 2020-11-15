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

#include "util/stdlib/strnlen.h"

#if defined(OS_MACOSX) && MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_7

#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_7
// Redeclare a method only available on Mac OS X 10.7 and later to suppress a
// -Wpartial-availability warning.
extern "C" {
size_t strnlen(const char* string, size_t max_length);
}  // extern "C"
#endif

namespace crashpad {

size_t strnlen(const char* string, size_t max_length) {
#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_7
  if (::strnlen) {
    return ::strnlen(string, max_length);
  }
#endif

  for (size_t index = 0; index < max_length; ++index) {
    if (string[index] == '\0') {
      return index;
    }
  }

  return max_length;
}

}  // namespace crashpad

#endif
