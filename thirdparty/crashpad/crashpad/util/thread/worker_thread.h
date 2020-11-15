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

#ifndef CRASHPAD_UTIL_THREAD_WORKER_THREAD_H_
#define CRASHPAD_UTIL_THREAD_WORKER_THREAD_H_

#include <memory>

#include "base/macros.h"
#include "util/synchronization/semaphore.h"

namespace crashpad {

namespace internal {
class WorkerThreadImpl;
}  // namespace internal

//! \brief A WorkerThread executes its Delegate's DoWork method repeatedly on a
//!     dedicated thread at a set time interval.
class WorkerThread {
 public:
  //! \brief An interface for doing work on a WorkerThread.
  class Delegate {
   public:
    //! \brief The work function executed by the WorkerThread every work
    //!     interval.
    virtual void DoWork(const WorkerThread* thread) = 0;

   protected:
    virtual ~Delegate() {}
  };

  //! \brief A delay or interval argument that causes an indefinite wait.
  static constexpr double kIndefiniteWait = Semaphore::kIndefiniteWait;

  //! \brief Creates a new WorkerThread that is not yet running.
  //!
  //! \param[in] work_interval The time interval in seconds at which the \a
  //!     delegate runs. The interval counts from the completion of
  //!     Delegate::DoWork() to the next invocation. This can be
  //!     #kIndefiniteWait if work should only be done when DoWorkNow() is
  //!     called.
  //! \param[in] delegate The work delegate to invoke every interval.
  WorkerThread(double work_interval, Delegate* delegate);
  ~WorkerThread();

  //! \brief Starts the worker thread.
  //!
  //! This may not be called if the thread is_running().
  //!
  //! \param[in] initial_work_delay The amount of time in seconds to wait
  //!     before invoking the \a delegate for the first time. Pass `0` for
  //!     no delay. This can be #kIndefiniteWait if work should not be done
  //!     until DoWorkNow() is called.
  void Start(double initial_work_delay);

  //! \brief Stops the worker thread from running.
  //!
  //! This may only be called if the thread is_running().
  //!
  //! If the work function is currently executing, this will not interrupt it.
  //! This method stops any future work from occurring. This method is safe
  //! to call from any thread with the exception of the worker thread itself,
  //! as this joins the thread.
  void Stop();

  //! \brief Interrupts a \a work_interval to execute the work function
  //!     immediately. This invokes Delegate::DoWork() on the thread, without
  //!     waiting for the current \a work_interval to expire. After the
  //!     delegate is invoked, the WorkerThread will start waiting for a new
  //!     \a work_interval.
  void DoWorkNow();

  //! \return `true` if the thread is running, `false` if it is not.
  bool is_running() const { return running_; }

 private:
  friend class internal::WorkerThreadImpl;

  double work_interval_;
  Delegate* delegate_;  // weak
  std::unique_ptr<internal::WorkerThreadImpl> impl_;
  bool running_;

  DISALLOW_COPY_AND_ASSIGN(WorkerThread);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_THREAD_WORKER_THREAD_H_
