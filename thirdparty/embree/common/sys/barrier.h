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

#include "intrinsics.h"
#include "sysinfo.h"
#include "atomic.h"

namespace embree
{
  /*! system barrier using operating system */
  class BarrierSys
  {
  public:

    /*! construction / destruction */
    BarrierSys (size_t N = 0);
    ~BarrierSys ();

  private:
    /*! class in non-copyable */
    BarrierSys (const BarrierSys& other) DELETED; // do not implement
    BarrierSys& operator= (const BarrierSys& other) DELETED; // do not implement

  public:
    /*! intializes the barrier with some number of threads */
    void init(size_t count);

    /*! lets calling thread wait in barrier */
    void wait();

  private:
    void* opaque;
  };

  /*! fast active barrier using atomitc counter */
  struct BarrierActive 
  {
  public:
    BarrierActive () 
      : cntr(0) {}
    
    void reset() {
      cntr.store(0);
    }

    void wait (size_t numThreads) 
    {
      cntr++;
      while (cntr.load() != numThreads) 
        pause_cpu();
    }

  private:
    std::atomic<size_t> cntr;
  };

  /*! fast active barrier that does not require initialization to some number of threads */
  struct BarrierActiveAutoReset
  {
  public:
    BarrierActiveAutoReset () 
      : cntr0(0), cntr1(0) {}

    void wait (size_t threadCount) 
    {
      cntr0.fetch_add(1);
      while (cntr0 != threadCount) pause_cpu();
      cntr1.fetch_add(1);
      while (cntr1 != threadCount) pause_cpu();
      cntr0.fetch_add(-1);
      while (cntr0 != 0) pause_cpu();
      cntr1.fetch_add(-1);
      while (cntr1 != 0) pause_cpu();
    }

  private:
    std::atomic<size_t> cntr0;
    std::atomic<size_t> cntr1;
  };

  class LinearBarrierActive
  {
  public:

    /*! construction and destruction */
    LinearBarrierActive (size_t threadCount = 0);
    ~LinearBarrierActive();
    
  private:
    /*! class in non-copyable */
    LinearBarrierActive (const LinearBarrierActive& other) DELETED; // do not implement
    LinearBarrierActive& operator= (const LinearBarrierActive& other) DELETED; // do not implement

  public:
    /*! intializes the barrier with some number of threads */
    void init(size_t threadCount);
    
    /*! thread with threadIndex waits in the barrier */
    void wait (const size_t threadIndex);
    
  private:
    volatile unsigned char* count0;
    volatile unsigned char* count1; 
    volatile unsigned int mode;
    volatile unsigned int flag0;
    volatile unsigned int flag1;
    volatile size_t threadCount;
  };
}

