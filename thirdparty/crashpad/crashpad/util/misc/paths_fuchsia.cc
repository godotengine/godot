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

#include "util/misc/paths.h"

#include <sys/stat.h>
#include <zircon/process.h>
#include <zircon/syscalls.h>

#include "base/logging.h"

namespace crashpad {

// static
bool Paths::Executable(base::FilePath* path) {
  // Assume the environment has been set up following
  // https://fuchsia.googlesource.com/docs/+/master/namespaces.md#typical-directory-structure
  // .
  *path = base::FilePath("/pkg/bin/app");
  return true;
}

}  // namespace crashpad
