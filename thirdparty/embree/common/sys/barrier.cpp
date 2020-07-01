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

#include "barrier.h"
#include "condition.h"
#include "regression.h"
#include "thread.h"

#if defined (__WIN32__)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace embree
{
  struct BarrierSysImplementation
  {
    __forceinline BarrierSysImplementation (size_t N) 
      : i(0), enterCount(0), exitCount(0), barrierSize(0) 
    {
      events[0] = CreateEvent(nullptr, TRUE, FALSE, nullptr);
      events[1] = CreateEvent(nullptr, TRUE, FALSE, nullptr);
      init(N);
    }
    
    __forceinline ~BarrierSysImplementation ()
    {
      CloseHandle(events[0]);
      CloseHandle(events[1]);
    }
    
    __forceinline void init(size_t N) 
    {
      barrierSize = N;
      enterCount.store(N);
      exitCount.store(N);
    }

    __forceinline void wait()
    {
      /* every thread entering the barrier decrements this count */
      size_t i0 = i;
      size_t cnt0 = enterCount--;

      /* all threads except the last one are wait in the barrier */
      if (cnt0 > 1) 
      {
        if (WaitForSingleObject(events[i0], INFINITE) != WAIT_OBJECT_0)
          THROW_RUNTIME_ERROR("WaitForSingleObjects failed");
      }
      
      /* the last thread starts all threads waiting at the barrier */
      else 
      {
        i = 1-i;
        enterCount.store(barrierSize);
        if (SetEvent(events[i0]) == 0)
          THROW_RUNTIME_ERROR("SetEvent failed");
      }

      /* every thread leaving the barrier decrements this count */
      size_t cnt1 = exitCount--;

      /* the last thread that left the barrier resets the event again */
      if (cnt1 == 1) 
      {
        exitCount.store(barrierSize);
        if (ResetEvent(events[i0]) == 0)
          THROW_RUNTIME_ERROR("ResetEvent failed");
      }
    }

  public:
    HANDLE events[2];
    atomic<size_t> i;
    atomic<size_t> enterCount;
    atomic<size_t> exitCount;
    size_t barrierSize;
  };
}

#else

namespace embree
{
  struct BarrierSysImplementation
  {
    __forceinline BarrierSysImplementation (size_t N) 
      : count(0), barrierSize(0) 
    {
      init(N);
    }
    
    __forceinline void init(size_t N) 
    {
      assert(count == 0);
      count = 0;
      barrierSize = N;
    }

    __forceinline void wait()
    {
      mutex.lock();
      count++;
      
      if (count == barrierSize) {
        count = 0;
        cond.notify_all();
        mutex.unlock();
        return;
      }
      
      cond.wait(mutex);
      mutex.unlock();
      return;
    }

  public:
    MutexSys mutex;
    ConditionSys cond;
    volatile size_t count;
    volatile size_t barrierSize;
  };
}

#endif

namespace embree
{
  BarrierSys::BarrierSys (size_t N) {
    opaque = new BarrierSysImplementation(N);
  }

  BarrierSys::~BarrierSys () {
    delete (BarrierSysImplementation*) opaque;
  }

  void BarrierSys::init(size_t count) {
    ((BarrierSysImplementation*) opaque)->init(count);
  }

  void BarrierSys::wait() {
    ((BarrierSysImplementation*) opaque)->wait();
  }

  LinearBarrierActive::LinearBarrierActive (size_t N) 
    : count0(nullptr), count1(nullptr), mode(0), flag0(0), flag1(0), threadCount(0)
  { 
    if (N == 0) N = getNumberOfLogicalThreads();
    init(N);
  }

  LinearBarrierActive::~LinearBarrierActive() 
  {
    delete[] count0;
    delete[] count1;
  }

  void LinearBarrierActive::init(size_t N) 
  {
    if (threadCount != N) {
      threadCount = N;
      if (count0) delete[] count0; count0 = new unsigned char[N];
      if (count1) delete[] count1; count1 = new unsigned char[N];
    }
    mode      = 0;
    flag0     = 0;
    flag1     = 0;
    for (size_t i=0; i<N; i++) count0[i] = 0;
    for (size_t i=0; i<N; i++) count1[i] = 0;
  }

  void LinearBarrierActive::wait (const size_t threadIndex)
  {
    if (mode == 0)
    {			
      if (threadIndex == 0)
      {	
        for (size_t i=0; i<threadCount; i++)
          count1[i] = 0;
        
        for (size_t i=1; i<threadCount; i++)
        {
          while (likely(count0[i] == 0)) 
            pause_cpu();
        }
        mode  = 1;
        flag1 = 0;
        __memory_barrier();
        flag0 = 1;
      }			
      else
      {					
        count0[threadIndex] = 1;
        {
          while (likely(flag0 == 0))
            pause_cpu();
        }
        
      }		
    }					
    else						
    {
      if (threadIndex == 0)
      {	
        for (size_t i=0; i<threadCount; i++)
          count0[i] = 0;
        
        for (size_t i=1; i<threadCount; i++)
        {		
          while (likely(count1[i] == 0))
            pause_cpu();
        }
        
        mode  = 0;
        flag0 = 0;
        __memory_barrier();
        flag1 = 1;
      }			
      else
      {					
        count1[threadIndex] = 1;
        {
          while (likely(flag1 == 0))
            pause_cpu();
        }
      }		
    }					
  }

  struct barrier_sys_regression_test : public RegressionTest
  {
    BarrierSys barrier;
    std::atomic<size_t> threadID;
    std::atomic<size_t> numFailed;
    std::vector<size_t> threadResults;

    barrier_sys_regression_test() 
      : RegressionTest("barrier_sys_regression_test"), threadID(0), numFailed(0)
    {
      registerRegressionTest(this);
    }

    static void thread_alloc(barrier_sys_regression_test* This)
    {
      size_t tid = This->threadID++;
      for (size_t j=0; j<1000; j++)
      {
        This->barrier.wait();
        This->threadResults[tid] = tid;
        This->barrier.wait();
      }
    }
    
    bool run ()
    {
      threadID.store(0);
      numFailed.store(0);

      size_t numThreads = getNumberOfLogicalThreads();
      threadResults.resize(numThreads);
      barrier.init(numThreads+1);

      /* create threads */
      std::vector<thread_t> threads;
      for (size_t i=0; i<numThreads; i++)
        threads.push_back(createThread((thread_func)thread_alloc,this));

      /* run test */ 
      for (size_t i=0; i<1000; i++)
      {
        for (size_t i=0; i<numThreads; i++) threadResults[i] = 0;
        barrier.wait();
        barrier.wait();
        for (size_t i=0; i<numThreads; i++) numFailed += threadResults[i] != i;
      }

      /* destroy threads */
      for (size_t i=0; i<numThreads; i++)
        join(threads[i]);

      return numFailed == 0;
    }
  };

  barrier_sys_regression_test barrier_sys_regression_test;
}


