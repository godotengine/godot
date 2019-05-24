// Copyright 2015 The Shaderc Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LIBSHADERC_UTIL_INC_MUTEX_H
#define LIBSHADERC_UTIL_INC_MUTEX_H

// shaderc_util::mutex will be defined and specialized
// depending on the platform that is being compiled.
// It is more or less conformant to the C++11 specification of std::mutex.
// However it does not implement try_lock.

#ifdef _WIN32
// windows.h #defines min and max if we don't define this.
// this means things like std::min and std::max break
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
namespace shaderc_util {

// As the name suggests, this mutex class is for running on windows.
// It conforms to the c++11 mutex implementation, and should be a
// drop in replacement.
class windows_mutex {
 public:
  using native_handle_type = HANDLE;

  windows_mutex() { mutex_ = CreateMutex(nullptr, false, nullptr); }

  ~windows_mutex() {
    if (mutex_ != INVALID_HANDLE_VALUE) {
      CloseHandle(mutex_);
    }
  }

  windows_mutex(const windows_mutex&) = delete;
  windows_mutex& operator=(const windows_mutex&) = delete;

  // Locks this mutex, waiting until the mutex is unlocked if it is not already.
  // It is not valid to lock a mutex that has already been locked.
  void lock() { WaitForSingleObject(mutex_, INFINITE); }

  // Unlocks this mutex. It is invalid to unlock a mutex that this thread
  // has not already locked.
  void unlock() { ReleaseMutex(mutex_); }

  // Returns the native handle for this mutex. In this case a HANDLE object.
  native_handle_type native_handle() { return mutex_; }

 private:
  HANDLE mutex_;
};

using mutex = windows_mutex;
}

#else
#include <pthread.h>
#include <memory>
namespace shaderc_util {

// As the name suggests, this mutex class is for running with pthreads.
// It conforms to the c++11 mutex implementation, and should be a
// drop in replacement.
class posix_mutex {
 public:
  using native_handle_type = pthread_mutex_t*;

  posix_mutex() { pthread_mutex_init(&mutex_, nullptr); }

  ~posix_mutex() { pthread_mutex_destroy(&mutex_); }

  posix_mutex(const posix_mutex&) = delete;
  posix_mutex& operator=(const posix_mutex&) = delete;

  // Locks this mutex, waiting until the mutex is unlocked if it is not already.
  // It is not valid to lock a mutex that has already been locked.
  void lock() { pthread_mutex_lock(&mutex_); }

  // Unlocks this mutex. It is invalid to unlock a mutex that this thread
  // has not already locked.
  void unlock() { pthread_mutex_unlock(&mutex_); }

  // Returns the native handle for this mutex. In this case a pthread_mutex_t*.
  native_handle_type native_handle() { return &mutex_; }

 private:
  pthread_mutex_t mutex_;
};

using mutex = posix_mutex;
}
#endif

#endif  // LIBSHADERC_UTIL_INC_MUTEX_H
