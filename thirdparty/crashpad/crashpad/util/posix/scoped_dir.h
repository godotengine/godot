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

#ifndef CRASHPAD_UTIL_POSIX_SCOPED_DIR_H_
#define CRASHPAD_UTIL_POSIX_SCOPED_DIR_H_

#include <dirent.h>

#include "base/scoped_generic.h"

namespace crashpad {
namespace internal {

struct ScopedDIRCloseTraits {
  static DIR* InvalidValue() { return nullptr; }
  static void Free(DIR* dir);
};

}  // namespace internal

//! \brief Maintains a directory opened by `opendir`.
//!
//! On destruction, the directory will be closed by calling `closedir`.
using ScopedDIR = base::ScopedGeneric<DIR*, internal::ScopedDIRCloseTraits>;

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_POSIX_SCOPED_DIR_H_
