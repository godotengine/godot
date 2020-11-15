// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_SYNCHRONIZATION_SEMAPHORE_H_
#define CRASHPAD_UTIL_SYNCHRONIZATION_SEMAPHORE_H_

#include <limits>

#include "build/build_config.h"

#if defined(OS_MACOSX)
#include <dispatch/dispatch.h>
#elif defined(OS_WIN)
#include <windows.h>
#elif defined(OS_ANDROID)
#include <condition_variable>
#include <mutex>
#else
#include <semaphore.h>
#endif

namespace crashpad {

//! \brief An anonymous in-process counting sempahore.
class Semaphore {
 public:
  //! \brief A TimedWait() argument that causes an indefinite wait.
  static constexpr double kIndefiniteWait =
      std::numeric_limits<double>::infinity();

  //! \brief Initializes the semaphore.
  //!
  //! \param[in] value The initial value of the semaphore.
  //!
  //! If the semaphore cannot be created, execution is terminated.
  explicit Semaphore(int value);

  ~Semaphore();

  //! \brief Performs the wait (or “procure”) operation on the semaphore.
  //!
  //! Atomically decrements the value of the semaphore by 1. If the new value is
  //! negative, this function blocks and will not return until the semaphore’s
  //! value is incremented to 0 by Signal().
  //!
  //! \sa TimedWait()
  void Wait();

  //! \brief Performs a timed wait (or “procure”) operation on the semaphore.
  //!
  //! \param[in] seconds The maximum number of seconds to wait for the operation
  //!     to complete. If \a seconds is #kIndefiniteWait, this method behaves as
  //!     Wait(), and will not time out.
  //!
  //! \return `false` if the wait timed out, `true` otherwise.
  //!
  //! This method is simlar to Wait(), except that the amount of time that it
  //! blocks is limited.
  bool TimedWait(double seconds);

  //! \brief Performs the signal (or “post”) operation on the semaphore.
  //!
  //! Atomically increments the value of the semaphore by 1. If the new value is
  //! 0, a caller blocked in Wait() will be awakened.
  void Signal();

 private:
#if defined(OS_MACOSX)
  dispatch_semaphore_t semaphore_;
#elif defined(OS_WIN)
  HANDLE semaphore_;
#elif defined(OS_ANDROID)
  std::condition_variable cv_;
  std::mutex mutex_;
  int value_;
#else
  sem_t semaphore_;
#endif
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_SYNCHRONIZATION_SEMAPHORE_H_
