// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "library.h"
#include "sysinfo.h"
#include "filename.h"

////////////////////////////////////////////////////////////////////////////////
/// Windows Platform
////////////////////////////////////////////////////////////////////////////////

#if defined(__WIN32__)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace embree
{
  /* opens a shared library */
  lib_t openLibrary(const std::string& file)
  {
    std::string fullName = file+".dll";
    FileName executable = getExecutableFileName();
    HANDLE handle = LoadLibrary((executable.path() + fullName).c_str());
    return lib_t(handle);
  }

  /* returns address of a symbol from the library */
  void* getSymbol(lib_t lib, const std::string& sym) {
    return (void*)GetProcAddress(HMODULE(lib),sym.c_str());
  }

  /* closes the shared library */
  void closeLibrary(lib_t lib) {
    FreeLibrary(HMODULE(lib));
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Unix Platform
////////////////////////////////////////////////////////////////////////////////

#if defined(__UNIX__)

#include <dlfcn.h>

namespace embree
{
  /* opens a shared library */
  lib_t openLibrary(const std::string& file)
  {
#if defined(__MACOSX__)
    std::string fullName = "lib"+file+".dylib";
#else
    std::string fullName = "lib"+file+".so";
#endif
    void* lib = dlopen(fullName.c_str(), RTLD_NOW);
    if (lib) return lib_t(lib);
    FileName executable = getExecutableFileName();
    lib = dlopen((executable.path() + fullName).c_str(),RTLD_NOW);
    if (lib == nullptr) {
      const char* error = dlerror();
      if (error) { 
        THROW_RUNTIME_ERROR(error);
      } else {
        THROW_RUNTIME_ERROR("could not load library "+executable.str());
      }
    }
    return lib_t(lib);
  }

  /* returns address of a symbol from the library */
  void* getSymbol(lib_t lib, const std::string& sym) {
    return dlsym(lib,sym.c_str());
  }

  /* closes the shared library */
  void closeLibrary(lib_t lib) {
    dlclose(lib);
  }
}
#endif
