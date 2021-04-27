// Copyright 2009-2020 Intel Corporation
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
      pthread_cond_init(&cond,nullptr); 
    }
    
    __forceinline ~ConditionImplementation() { 
      pthread_cond_destroy(&cond);
    } 
    
    __forceinline void wait(MutexSys& mutex) { 
      pthread_cond_wait(&cond, (pthread_mutex_t*)mutex.mutex); 
    }
    
    __forceinline void notify_all() { 
      pthread_cond_broadcast(&cond); 
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
