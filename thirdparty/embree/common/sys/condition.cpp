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

#include "condition.h"

#if defined(__WIN32__) && !defined(PTHREADS_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace embree
{
#if (_WIN32_WINNT >= 0x0600)

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

#else

#pragma message ("WARNING: This condition variable implementation has known performance issues!")

  struct ConditionImplementation
  {
    __forceinline ConditionImplementation () : count(0) {
      event = CreateEvent(nullptr,TRUE,FALSE,nullptr);
    }

    __forceinline ~ConditionImplementation () {
      CloseHandle(event);
    }

    __forceinline void wait(MutexSys& mutex_in)
    {
      /* atomically increment thread count */
      ssize_t cnt0 = atomic_add(&count,+1);
      mutex_in.unlock();

      /* all threads except the last one are wait in the barrier */
      if (WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0)
        THROW_RUNTIME_ERROR("WaitForSingleObject failed");

      /* atomically decrement thread count */
      mutex_in.lock();
      ssize_t cnt1 = atomic_add(&count,-1);

      /* the last thread that left the barrier resets the event again */
      if (cnt1 == 1) {
        if (ResetEvent(event) == 0)
          THROW_RUNTIME_ERROR("ResetEvent failed");
      }
    }

    __forceinline void notify_all() 
    {
      /* we support only one broadcast at a given time */
      bool hasWaiters = count > 0;

      /* if threads are waiting, let them continue */
      if (hasWaiters) 
        if (SetEvent(event) == 0)
          THROW_RUNTIME_ERROR("SetEvent failed");
    }

  public:
    HANDLE event;
    volatile atomic_t count;
  };
#endif
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
