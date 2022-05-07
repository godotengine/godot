// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
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

#if !defined(_WIN32)
  #include <pthread.h>
  #include <sched.h>
  #if defined(__APPLE__)
    #include <mach/thread_policy.h>
  #endif
#endif

#include <vector>
#include <mutex>

namespace oidn {

  // --------------------------------------------------------------------------
  // ThreadLocal
  // --------------------------------------------------------------------------

  // Wrapper which makes any variable thread-local
  template<typename T>
  class ThreadLocal : public Verbose
  {
  private:
  #if defined(_WIN32)
    DWORD key;
  #else
    pthread_key_t key;
  #endif

    std::vector<T*> instances;
    std::mutex mutex;

  public:
    ThreadLocal(int verbose = 0)
      : Verbose(verbose)
    {
    #if defined(_WIN32)
      key = TlsAlloc();
      if (key == TLS_OUT_OF_INDEXES)
        OIDN_FATAL("TlsAlloc failed");
    #else
      if (pthread_key_create(&key, nullptr) != 0)
        OIDN_FATAL("pthread_key_create failed");
    #endif
    }

    ~ThreadLocal()
    {
      std::lock_guard<std::mutex> lock(mutex);
      for (T* ptr : instances)
        delete ptr;

    #if defined(_WIN32)
      if (!TlsFree(key))
        OIDN_WARNING("TlsFree failed");
    #else
      if (pthread_key_delete(key) != 0)
        OIDN_WARNING("pthread_key_delete failed");
    #endif
    }

    T& get()
    {
    #if defined(_WIN32)
      T* ptr = (T*)TlsGetValue(key);
    #else
      T* ptr = (T*)pthread_getspecific(key);
    #endif

      if (ptr)
        return *ptr;

      ptr = new T;
      std::lock_guard<std::mutex> lock(mutex);
      instances.push_back(ptr);

    #if defined(_WIN32)
      if (!TlsSetValue(key, ptr))
        OIDN_FATAL("TlsSetValue failed");
    #else
      if (pthread_setspecific(key, ptr) != 0)
        OIDN_FATAL("pthread_setspecific failed");
    #endif

      return *ptr;
    }
  };

#if defined(_WIN32)

  // --------------------------------------------------------------------------
  // ThreadAffinity - Windows
  // --------------------------------------------------------------------------

  class ThreadAffinity : public Verbose
  {
  private:
    typedef BOOL (WINAPI *GetLogicalProcessorInformationExFunc)(LOGICAL_PROCESSOR_RELATIONSHIP,
                                                                PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
                                                                PDWORD);

    typedef BOOL (WINAPI *SetThreadGroupAffinityFunc)(HANDLE,
                                                      CONST GROUP_AFFINITY*,
                                                      PGROUP_AFFINITY);

    GetLogicalProcessorInformationExFunc pGetLogicalProcessorInformationEx = nullptr;
    SetThreadGroupAffinityFunc pSetThreadGroupAffinity = nullptr;

    std::vector<GROUP_AFFINITY> affinities;    // thread affinities
    std::vector<GROUP_AFFINITY> oldAffinities; // original thread affinities

  public:
    ThreadAffinity(int numThreadsPerCore = INT_MAX, int verbose = 0);

    int getNumThreads() const
    {
      return (int)affinities.size();
    }

    // Sets the affinity (0..numThreads-1) of the thread after saving the current affinity
    void set(int threadIndex);

    // Restores the affinity of the thread
    void restore(int threadIndex);
  };

#elif defined(__linux__)

  // --------------------------------------------------------------------------
  // ThreadAffinity - Linux
  // --------------------------------------------------------------------------

  class ThreadAffinity : public Verbose
  {
  private:
    std::vector<cpu_set_t> affinities;    // thread affinities
    std::vector<cpu_set_t> oldAffinities; // original thread affinities

  public:
    ThreadAffinity(int numThreadsPerCore = INT_MAX, int verbose = 0);

    int getNumThreads() const
    {
      return (int)affinities.size();
    }

    // Sets the affinity (0..numThreads-1) of the thread after saving the current affinity
    void set(int threadIndex);

    // Restores the affinity of the thread
    void restore(int threadIndex);
  };

#elif defined(__APPLE__)

  // --------------------------------------------------------------------------
  // ThreadAffinity - macOS
  // --------------------------------------------------------------------------

  class ThreadAffinity : public Verbose
  {
  private:
    std::vector<thread_affinity_policy> affinities;    // thread affinities
    std::vector<thread_affinity_policy> oldAffinities; // original thread affinities

  public:
    ThreadAffinity(int numThreadsPerCore = INT_MAX, int verbose = 0);

    int getNumThreads() const
    {
      return (int)affinities.size();
    }

    // Sets the affinity (0..numThreads-1) of the thread after saving the current affinity
    void set(int threadIndex);

    // Restores the affinity of the thread
    void restore(int threadIndex);
  };

#endif

} // namespace oidn
