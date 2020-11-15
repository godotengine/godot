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

#include "test/scoped_module_handle.h"

#include "base/logging.h"

namespace crashpad {
namespace test {

// static
void ScopedModuleHandle::Impl::Close(ModuleHandle handle) {
#if defined(OS_POSIX)
  if (dlclose(handle) != 0) {
    LOG(ERROR) << "dlclose: " << dlerror();
  }
#elif defined(OS_WIN)
  if (!FreeLibrary(handle)) {
    PLOG(ERROR) << "FreeLibrary";
  }
#else
#error Port
#endif
}

ScopedModuleHandle::ScopedModuleHandle(ModuleHandle handle) : handle_(handle) {}

ScopedModuleHandle::ScopedModuleHandle(ScopedModuleHandle&& other)
    : handle_(other.handle_) {
  other.handle_ = nullptr;
}

ScopedModuleHandle::~ScopedModuleHandle() {
  if (valid()) {
    Impl::Close(handle_);
  }
}

}  // namespace test
}  // namespace crashpad
