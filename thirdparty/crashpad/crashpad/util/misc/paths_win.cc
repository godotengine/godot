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

#include "util/misc/paths.h"

#include <windows.h>

#include "base/logging.h"

namespace crashpad {

// static
bool Paths::Executable(base::FilePath* path) {
  wchar_t executable_path[_MAX_PATH];
  unsigned int len =
      GetModuleFileName(nullptr, executable_path, arraysize(executable_path));
  if (len == 0) {
    PLOG(ERROR) << "GetModuleFileName";
    return false;
  } else if (len >= arraysize(executable_path)) {
    LOG(ERROR) << "GetModuleFileName";
    return false;
  }

  *path = base::FilePath(executable_path);
  return true;
}

}  // namespace crashpad
