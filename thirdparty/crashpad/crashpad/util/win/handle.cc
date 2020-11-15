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

#include "util/win/handle.h"

#include <stdint.h>

#include "base/numerics/safe_conversions.h"
#include "util/misc/from_pointer_cast.h"

namespace crashpad {

// These functions use “int” for the 32-bit integer handle type because
// sign-extension needs to work correctly. INVALID_HANDLE_VALUE is defined as
// ((HANDLE)(LONG_PTR)-1), and this needs to round-trip through an integer and
// back to the same HANDLE value.

int HandleToInt(HANDLE handle) {
  return FromPointerCast<int>(handle);
}

HANDLE IntToHandle(int handle_int) {
  return reinterpret_cast<HANDLE>(static_cast<intptr_t>(handle_int));
}

}  // namespace crashpad
