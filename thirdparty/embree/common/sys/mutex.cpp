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

#include "mutex.h"
#include "regression.h"
#include "../algorithms/parallel_for.h"

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

#if 0
namespace embree
{
  template<typename Mutex>
    struct mutex_regression_test : public RegressionTest
  {
    Mutex mutex;
    static const size_t N = 100;
    static const size_t M = 10000;

    mutex_regression_test(const char* name) : RegressionTest(name) {
      registerRegressionTest(this);
    }
    
    bool run ()
    {
      size_t counter = 0;
      parallel_for(N, [&] (const size_t i) {
          for (size_t i=0; i<M; i++) 
          {
            mutex.lock();
            counter++;
            mutex.unlock();
          }
        });
      
      return counter == N*M;
    }
  };

  mutex_regression_test<MutexSys> mutex_sys_regression("sys_mutex_regression_test");
  mutex_regression_test<SpinLock> mutex_atomic_regression("atomic_mutex_regression_test");
}
#endif
