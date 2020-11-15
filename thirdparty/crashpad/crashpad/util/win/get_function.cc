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

#include "util/win/get_function.h"

#include "base/logging.h"
#include "base/strings/utf_string_conversions.h"

namespace crashpad {
namespace internal {

FARPROC GetFunctionInternal(
    const wchar_t* library, const char* function, bool required) {
  HMODULE module = LoadLibrary(library);
  DPCHECK(!required || module) << "LoadLibrary " << base::UTF16ToUTF8(library);
  if (!module) {
    return nullptr;
  }

  // Strip off any leading :: that may have come from stringifying the
  // function’s name.
  if (function[0] == ':' && function[1] == ':' &&
      function[2] && function[2] != ':') {
    function += 2;
  }

  FARPROC proc = GetProcAddress(module, function);
  DPCHECK(!required || proc) << "GetProcAddress " << function;
  return proc;
}

}  // namespace internal
}  // namespace crashpad
