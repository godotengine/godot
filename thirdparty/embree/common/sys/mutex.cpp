// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mutex.h"
#include "regression.h"

#if defined(__WIN32__) && !defined(PTHREADS_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace embree
{
  MutexSys::MutexSys() { mutex = new CRITICAL_SECTION; InitializeCriticalSection((CRITICAL_SECTION*)mutex); }
  MutexSys::~MutexSys() { DeleteCriticalSection((CRITICAL_SECTION*)mutex); delete (CRITICAL_SECTION*)mutex; }
  void MutexSys::lock() { EnterCriticalSection((CRITICAL_SECTION*)mutex); }
  bool MutexSys::try_lock() { return TryEnterCriticalSection((CRITICAL_SECTION*)mutex) != 0; }
  void MutexSys::unlock() { LeaveCriticalSection((CRITICAL_SECTION*)mutex); }
}
#endif

#if defined(__UNIX__) || defined(PTHREADS_WIN32)
#include <pthread.h>
namespace embree
{
  /*! system mutex using pthreads */
  MutexSys::MutexSys() 
  { 
    mutex = new pthread_mutex_t; 
    if (pthread_mutex_init((pthread_mutex_t*)mutex, nullptr) != 0)
      THROW_RUNTIME_ERROR("pthread_mutex_init failed");
  }
  
  MutexSys::~MutexSys() 
  { 
    MAYBE_UNUSED bool ok = pthread_mutex_destroy((pthread_mutex_t*)mutex) == 0;
    assert(ok);
    delete (pthread_mutex_t*)mutex; 
    mutex = nullptr;
  }
  
  void MutexSys::lock() 
  { 
    if (pthread_mutex_lock((pthread_mutex_t*)mutex) != 0) 
      THROW_RUNTIME_ERROR("pthread_mutex_lock failed");
  }
  
  bool MutexSys::try_lock() { 
    return pthread_mutex_trylock((pthread_mutex_t*)mutex) == 0;
  }
  
  void MutexSys::unlock() 
  { 
    if (pthread_mutex_unlock((pthread_mutex_t*)mutex) != 0)
      THROW_RUNTIME_ERROR("pthread_mutex_unlock failed");
  }
};
#endif
