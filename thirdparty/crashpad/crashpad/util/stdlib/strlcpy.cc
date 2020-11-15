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

#include "util/stdlib/strlcpy.h"

#include "base/logging.h"
#include "build/build_config.h"

#if defined(OS_WIN) && defined(WCHAR_T_IS_UTF16)
#include <strsafe.h>
#endif

namespace crashpad {

#if defined(OS_WIN) && defined(WCHAR_T_IS_UTF16)

size_t c16lcpy(base::char16* destination,
               const base::char16* source,
               size_t length) {
  HRESULT result = StringCchCopyW(destination, length, source);
  CHECK(result == S_OK || result == STRSAFE_E_INSUFFICIENT_BUFFER);
  return wcslen(source);
}

#elif defined(WCHAR_T_IS_UTF32)

size_t c16lcpy(base::char16* destination,
               const base::char16* source,
               size_t length) {
  size_t source_length = base::c16len(source);
  if (source_length < length) {
    base::c16memcpy(destination, source, source_length + 1);
  } else if (length != 0) {
    base::c16memcpy(destination, source, length - 1);
    destination[length - 1] = '\0';
  }
  return source_length;
}

#endif  // WCHAR_T_IS_UTF32

}  // namespace crashpad
