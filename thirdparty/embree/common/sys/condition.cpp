// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "condition.h"

#if defined(__WIN32__) && !defined(PTHREADS_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace embree
{
  struct ConditionImplementation
  {
    __forceinline ConditionImplementation () {
      InitializeConditionVariable(&cond);
    }

    __forceinline ~ConditionImplementation () {
    }

    __forceinline void wait(MutexSys& mutex_in) {
      SleepConditionVariableCS(&cond, (LPCRITICAL_SECTION)mutex_in.mutex, INFINITE);
    }

    __forceinline void notify_all() {
      WakeAllConditionVariable(&cond);
    }

  public:
    CONDITION_VARIABLE cond;
  };
}
#endif

#if defined(__UNIX__) || defined(PTHREADS_WIN32)
#include <pthread.h>
namespace embree
{
  struct ConditionImplementation
  {
    __forceinline ConditionImplementation () { 
      if (pthread_cond_init(&cond,nullptr) != 0)
        THROW_RUNTIME_ERROR("pthread_cond_init failed");
    }
    
    __forceinline ~ConditionImplementation() { 
      MAYBE_UNUSED bool ok = pthread_cond_destroy(&cond) == 0;
      assert(ok);
    }
    
    __forceinline void wait(MutexSys& mutex) { 
      if (pthread_cond_wait(&cond, (pthread_mutex_t*)mutex.mutex) != 0)
        THROW_RUNTIME_ERROR("pthread_cond_wait failed");
    }
    
    __forceinline void notify_all() { 
      if (pthread_cond_broadcast(&cond) != 0)
        THROW_RUNTIME_ERROR("pthread_cond_broadcast failed");
    }
    
  public:
    pthread_cond_t cond;
  };
}
#endif

namespace embree 
{
  ConditionSys::ConditionSys () { 
    cond = new ConditionImplementation; 
  }

  ConditionSys::~ConditionSys() { 
    delete (ConditionImplementation*) cond;
  }

  void ConditionSys::wait(MutexSys& mutex) { 
    ((ConditionImplementation*) cond)->wait(mutex);
  }

  void ConditionSys::notify_all() { 
    ((ConditionImplementation*) cond)->notify_all();
  }
}
