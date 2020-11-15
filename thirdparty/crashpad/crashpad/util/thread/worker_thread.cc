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

#include "util/thread/worker_thread.h"

#include "base/logging.h"
#include "util/thread/thread.h"

namespace crashpad {

namespace internal {

class WorkerThreadImpl final : public Thread {
 public:
  WorkerThreadImpl(WorkerThread* self, double initial_work_delay)
      : semaphore_(0),
        initial_work_delay_(initial_work_delay),
        self_(self) {}
  ~WorkerThreadImpl() {}

  void ThreadMain() override {
    if (initial_work_delay_ > 0)
      semaphore_.TimedWait(initial_work_delay_);

    while (self_->running_) {
      self_->delegate_->DoWork(self_);
      semaphore_.TimedWait(self_->work_interval_);
    }
  }

  void SignalSemaphore() {
    semaphore_.Signal();
  }

 private:
  // TODO(mark): Use a condition variable instead?
  Semaphore semaphore_;
  double initial_work_delay_;
  WorkerThread* self_;  // Weak, owns this.
};

}  // namespace internal

WorkerThread::WorkerThread(double work_interval,
                           WorkerThread::Delegate* delegate)
    : work_interval_(work_interval),
      delegate_(delegate),
      impl_(),
      running_(false) {}

WorkerThread::~WorkerThread() {
  DCHECK(!running_);
}

void WorkerThread::Start(double initial_work_delay) {
  DCHECK(!impl_);
  DCHECK(!running_);

  running_ = true;
  impl_.reset(new internal::WorkerThreadImpl(this, initial_work_delay));
  impl_->Start();
}

void WorkerThread::Stop() {
  DCHECK(running_);
  DCHECK(impl_);

  if (!running_)
    return;

  running_ = false;

  impl_->SignalSemaphore();
  impl_->Join();
  impl_.reset();
}

void WorkerThread::DoWorkNow() {
  DCHECK(running_);
  impl_->SignalSemaphore();
}

}  // namespace crashpad
