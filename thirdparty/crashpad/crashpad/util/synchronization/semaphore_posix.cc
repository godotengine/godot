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

#include "util/synchronization/semaphore.h"

#include <errno.h>
#include <math.h>
#include <time.h>

#include <chrono>

#include "base/logging.h"
#include "base/posix/eintr_wrapper.h"
#include "util/misc/time.h"

namespace crashpad {

#if defined(OS_ANDROID)

Semaphore::Semaphore(int value) : cv_(), mutex_(), value_(value) {}

Semaphore::~Semaphore() = default;

void Semaphore::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return this->value_ > 0; });
  --value_;
}

bool Semaphore::TimedWait(double seconds) {
  DCHECK_GE(seconds, 0.0);

  if (isinf(seconds)) {
    Wait();
    return true;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  if (!cv_.wait_for(lock, std::chrono::duration<double>(seconds), [this] {
        return this->value_ > 0;
      })) {
    return false;
  }
  --value_;
  return true;
}

void Semaphore::Signal() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++value_;
  cv_.notify_one();
}

#elif !defined(OS_MACOSX)

Semaphore::Semaphore(int value) {
  PCHECK(sem_init(&semaphore_, 0, value) == 0) << "sem_init";
}

Semaphore::~Semaphore() {
  PCHECK(sem_destroy(&semaphore_) == 0) << "sem_destroy";
}

void Semaphore::Wait() {
  PCHECK(HANDLE_EINTR(sem_wait(&semaphore_)) == 0) << "sem_wait";
}

bool Semaphore::TimedWait(double seconds) {
  DCHECK_GE(seconds, 0.0);

  if (isinf(seconds)) {
    Wait();
    return true;
  }

  timespec current_time;
  if (clock_gettime(CLOCK_REALTIME, &current_time) != 0) {
    PLOG(ERROR) << "clock_gettime";
    return false;
  }
  timespec timeout;
  timeout.tv_sec = seconds;
  timeout.tv_nsec = (seconds - trunc(seconds)) * 1E9;
  AddTimespec(current_time, timeout, &timeout);

  int rv = HANDLE_EINTR(sem_timedwait(&semaphore_, &timeout));
  PCHECK(rv == 0 || errno == ETIMEDOUT) << "sem_timedwait";
  return rv == 0;
}

void Semaphore::Signal() {
  PCHECK(sem_post(&semaphore_) == 0) << "sem_post";
}

#endif  // OS_ANDROID

}  // namespace crashpad
