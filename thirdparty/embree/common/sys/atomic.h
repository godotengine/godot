// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
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

#pragma once

#include <atomic>
#include "intrinsics.h"

namespace embree
{
/* compiler memory barriers */
#if defined(__INTEL_COMPILER)
//#define __memory_barrier() __memory_barrier()
#elif defined(__GNUC__) || defined(__clang__)
#  define __memory_barrier() asm volatile("" ::: "memory")
#elif  defined(_MSC_VER)
#  define __memory_barrier() _ReadWriteBarrier()
#endif

  template <typename T>
    struct atomic : public std::atomic<T>
  {
    atomic () {}
      
    atomic (const T& a)
      : std::atomic<T>(a) {}

    atomic (const atomic<T>& a) {
      this->store(a.load());
    }

    atomic& operator=(const atomic<T>& other) {
      this->store(other.load());
      return *this;
    }
  };

  template<typename T>
    __forceinline void atomic_min(std::atomic<T>& aref, const T& bref)
  {
    const T b = bref.load();
    while (true) {
      T a = aref.load();
      if (a <= b) break;
      if (aref.compare_exchange_strong(a,b)) break;
    }
  }

  template<typename T>
    __forceinline void atomic_max(std::atomic<T>& aref, const T& bref)
  {
    const T b = bref.load();
    while (true) {
      T a = aref.load();
      if (a >= b) break;
      if (aref.compare_exchange_strong(a,b)) break;
    }
  }
}
