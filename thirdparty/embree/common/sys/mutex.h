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

#include "platform.h"
#include "intrinsics.h"
#include "atomic.h"

namespace embree
{
  /*! system mutex */
  class MutexSys {
    friend struct ConditionImplementation;
  public:
    MutexSys();
    ~MutexSys();

  private:
    MutexSys (const MutexSys& other) DELETED; // do not implement
    MutexSys& operator= (const MutexSys& other) DELETED; // do not implement

  public:
    void lock();
    bool try_lock();
    void unlock();

  protected:
    void* mutex;
  };

  /*! spinning mutex */
  class SpinLock
  {
  public:
 
    SpinLock ()
      : flag(false) {}

    __forceinline bool isLocked() {
      return flag.load();
    }

    __forceinline void lock()
    {
      while (true) 
      {
        while (flag.load()) 
        {
          _mm_pause(); 
          _mm_pause();
        }
        
        bool expected = false;
        if (flag.compare_exchange_strong(expected,true,std::memory_order_acquire))
          break;
      }
    }
    
    __forceinline bool try_lock()
    {
      bool expected = false;
      if (flag.load() != expected) {
        return false;
      }
      return flag.compare_exchange_strong(expected,true,std::memory_order_acquire);
    }

    __forceinline void unlock() {
      flag.store(false,std::memory_order_release);
    }
    
    __forceinline void wait_until_unlocked() 
    {
      while(flag.load())
      {
        _mm_pause(); 
        _mm_pause();
      }
    }

  public:
    atomic<bool> flag;
  };

  /*! safe mutex lock and unlock helper */
  template<typename Mutex> class Lock {
  public:
    Lock (Mutex& mutex) : mutex(mutex), locked(true) { mutex.lock(); }
    Lock (Mutex& mutex, bool locked) : mutex(mutex), locked(locked) {}
    ~Lock() { if (locked) mutex.unlock(); }
    __forceinline void lock() { assert(!locked); locked = true; mutex.lock(); }
    __forceinline bool isLocked() const { return locked; }
  protected:
    Mutex& mutex;
    bool locked;
  };
}
