// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "platform.h"

namespace oidn {

  // ----------------------------------------------------------------------------
  // Common functions
  // ----------------------------------------------------------------------------

  void* alignedMalloc(size_t size, size_t alignment)
  {
    if (size == 0)
      return nullptr;

    assert((alignment & (alignment-1)) == 0);
    void* ptr = _mm_malloc(size, alignment);

    if (ptr == nullptr)
      throw std::bad_alloc();

    return ptr;
  }

  void alignedFree(void* ptr)
  {
    if (ptr)
      _mm_free(ptr);
  }

  // ----------------------------------------------------------------------------
  // System information
  // ----------------------------------------------------------------------------

  std::string getPlatformName()
  {
    std::string name;

  #if defined(__linux__)
    name = "Linux";
  #elif defined(__FreeBSD__)
    name = "FreeBSD";
  #elif defined(__CYGWIN__)
    name = "Cygwin";
  #elif defined(_WIN32)
    name = "Windows";
  #elif defined(__APPLE__)
    name = "macOS";
  #elif defined(__unix__)
    name = "Unix";
  #else
    return "Unknown";
  #endif

  #if defined(__x86_64__) || defined(_M_X64) || defined(__ia64__) || defined(__aarch64__)
    name += " (64-bit)";
  #else
    name += " (32-bit)";
  #endif

    return name;
  }

  std::string getCompilerName()
  {
  #if defined(__INTEL_COMPILER)
    int mayor = __INTEL_COMPILER / 100 % 100;
    int minor = __INTEL_COMPILER % 100;
    std::string version = "Intel Compiler ";
    version += toString(mayor);
    version += "." + toString(minor);
  #if defined(__INTEL_COMPILER_UPDATE)
    version += "." + toString(__INTEL_COMPILER_UPDATE);
  #endif
    return version;
  #elif defined(__clang__)
    return "Clang " __clang_version__;
  #elif defined(__GNUC__)
    return "GCC " __VERSION__;
  #elif defined(_MSC_VER)
    std::string version = toString(_MSC_FULL_VER);
    version.insert(4, ".");
    version.insert(9, ".");
    version.insert(2, ".");
    return "Visual C++ Compiler " + version;
  #else
    return "Unknown";
  #endif
  }

  std::string getBuildName()
  {
  #if defined(NDEBUG)
    return "Release";
  #else
    return "Debug";
  #endif
  }

} // namespace oidn
