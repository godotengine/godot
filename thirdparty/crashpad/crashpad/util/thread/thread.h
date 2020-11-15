// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_THREAD_THREAD_H_
#define CRASHPAD_UTIL_THREAD_THREAD_H_

#include "base/macros.h"
#include "build/build_config.h"

#if defined(OS_POSIX)
#include <pthread.h>
#elif defined(OS_WIN)
#include <windows.h>
#endif  // OS_POSIX

namespace crashpad {

//! \brief Basic thread abstraction. Users should derive from this
//!     class and implement ThreadMain().
class Thread {
 public:
  Thread();
  virtual ~Thread();

  //! \brief Create a platform thread, and run ThreadMain() on that thread. Must
  //!     be paired with a call to Join().
  void Start();

  //! \brief Block until ThreadMain() exits. This may be called from any thread.
  //!     Must paired with a call to Start().
  void Join();

 private:
  //! \brief The thread entry point to be implemented by the subclass.
  virtual void ThreadMain() = 0;

  static
#if defined(OS_POSIX)
      void*
#elif defined(OS_WIN)
      DWORD WINAPI
#endif  // OS_POSIX
      ThreadEntryThunk(void* argument);

#if defined(OS_POSIX)
  pthread_t platform_thread_;
#elif defined(OS_WIN)
  HANDLE platform_thread_;
#endif

  DISALLOW_COPY_AND_ASSIGN(Thread);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_THREAD_THREAD_H_
