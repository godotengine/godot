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

#ifndef CRASHPAD_TEST_SCOPED_MODULE_HANDLE_H_
#define CRASHPAD_TEST_SCOPED_MODULE_HANDLE_H_

#include "base/macros.h"
#include "build/build_config.h"

#if defined(OS_POSIX)
#include <dlfcn.h>
#elif defined(OS_WIN)
#include <windows.h>
#endif

namespace crashpad {
namespace test {

//! \brief Maintains ownership of a loadable module handle, releasing it as
//!     appropriate on destruction.
class ScopedModuleHandle {
 private:
  class Impl {
   public:
#if defined(OS_POSIX)
    using ModuleHandle = void*;

    static void* LookUpSymbol(ModuleHandle handle, const char* symbol_name) {
      return dlsym(handle, symbol_name);
    }
#elif defined(OS_WIN)
    using ModuleHandle = HMODULE;

    static void* LookUpSymbol(ModuleHandle handle, const char* symbol_name) {
      return reinterpret_cast<void*>(GetProcAddress(handle, symbol_name));
    }
#endif

    static void Close(ModuleHandle handle);

   private:
    DISALLOW_IMPLICIT_CONSTRUCTORS(Impl);
  };

 public:
  using ModuleHandle = Impl::ModuleHandle;

  explicit ScopedModuleHandle(ModuleHandle handle);
  ScopedModuleHandle(ScopedModuleHandle&& handle);
  ~ScopedModuleHandle();

  //! \return The module handle being managed.
  ModuleHandle get() const { return handle_; }

  //! \return `true` if this object manages a valid loadable module handle.
  bool valid() const { return handle_ != nullptr; }

  //! \return The value of the symbol named by \a symbol_name, or `nullptr` on
  //!     failure.
  template <typename T>
  T LookUpSymbol(const char* symbol_name) const {
    return reinterpret_cast<T>(Impl::LookUpSymbol(handle_, symbol_name));
  }

 private:
  ModuleHandle handle_;

  DISALLOW_COPY_AND_ASSIGN(ScopedModuleHandle);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_SCOPED_MODULE_HANDLE_H_
