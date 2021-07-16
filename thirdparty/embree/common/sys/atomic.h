// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
