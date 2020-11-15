// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_HANDLER_PRUNE_CRASH_REPORTS_THREAD_H_
#define CRASHPAD_HANDLER_PRUNE_CRASH_REPORTS_THREAD_H_

#include <memory>

#include "base/macros.h"
#include "util/thread/stoppable.h"
#include "util/thread/worker_thread.h"

namespace crashpad {

class CrashReportDatabase;
class PruneCondition;

//! \brief A thread that periodically prunes crash reports from the database
//!     using the specified condition.
//!
//! After the thread is started, the database is pruned using the condition
//! every 24 hours. Upon calling Start(), the thread waits 10 minutes before
//! performing the initial prune operation.
class PruneCrashReportThread : public WorkerThread::Delegate, public Stoppable {
 public:
  //! \brief Constructs a new object.
  //!
  //! \param[in] database The database to prune crash reports from.
  //! \param[in] condition The condition used to evaluate crash reports for
  //!     pruning.
  PruneCrashReportThread(CrashReportDatabase* database,
                         std::unique_ptr<PruneCondition> condition);
  ~PruneCrashReportThread();

  // Stoppable:

  //! \brief Starts a dedicated pruning thread.
  //!
  //! The thread waits before running the initial prune, so as to not interfere
  //! with any startup-related IO performed by the client.
  //!
  //! This method may only be be called on a newly-constructed object or after
  //! a call to Stop().
  void Start() override;

  //! \brief Stops the pruning thread.
  //!
  //! This method must only be called after Start(). If Start() has been called,
  //! this method must be called before destroying an object of this class.
  //!
  //! This method may be called from any thread other than the pruning thread.
  //! It is expected to only be called from the same thread that called Start().
  void Stop() override;

 private:
  // WorkerThread::Delegate:
  void DoWork(const WorkerThread* thread) override;

  WorkerThread thread_;
  std::unique_ptr<PruneCondition> condition_;
  CrashReportDatabase* database_;  // weak

  DISALLOW_COPY_AND_ASSIGN(PruneCrashReportThread);
};

}  // namespace crashpad

#endif  // CRASHPAD_HANDLER_PRUNE_CRASH_REPORTS_THREAD_H_
