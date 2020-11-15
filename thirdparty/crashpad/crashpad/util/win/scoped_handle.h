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

#ifndef CRASHPAD_UTIL_WIN_SCOPED_HANDLE_H_
#define CRASHPAD_UTIL_WIN_SCOPED_HANDLE_H_

#include <windows.h>

#include "base/scoped_generic.h"

namespace crashpad {

namespace internal {

struct ScopedFileHANDLECloseTraits {
  static HANDLE InvalidValue() { return INVALID_HANDLE_VALUE; }
  static void Free(HANDLE handle);
};

struct ScopedKernelHANDLECloseTraits {
  static HANDLE InvalidValue() { return nullptr; }
  static void Free(HANDLE handle);
};

struct ScopedSearchHANDLECloseTraits {
  static HANDLE InvalidValue() { return INVALID_HANDLE_VALUE; }
  static void Free(HANDLE handle);
};

}  // namespace internal

using ScopedFileHANDLE =
    base::ScopedGeneric<HANDLE, internal::ScopedFileHANDLECloseTraits>;
using ScopedKernelHANDLE =
    base::ScopedGeneric<HANDLE, internal::ScopedKernelHANDLECloseTraits>;
using ScopedSearchHANDLE =
    base::ScopedGeneric<HANDLE, internal::ScopedSearchHANDLECloseTraits>;

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_SCOPED_HANDLE_H_
