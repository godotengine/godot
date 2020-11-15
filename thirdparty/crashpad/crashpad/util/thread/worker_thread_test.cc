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

#include "gtest/gtest.h"
#include "util/misc/clock.h"
#include "util/synchronization/semaphore.h"

namespace crashpad {
namespace test {
namespace {

constexpr uint64_t kNanosecondsPerSecond = static_cast<uint64_t>(1E9);

class WorkDelegate : public WorkerThread::Delegate {
 public:
  WorkDelegate() {}
  ~WorkDelegate() {}

  void DoWork(const WorkerThread* thread) override {
    if (work_count_ < waiting_for_count_) {
      if (++work_count_ == waiting_for_count_) {
        semaphore_.Signal();
      }
    }
  }

  void SetDesiredWorkCount(int times) {
    waiting_for_count_ = times;
  }

  //! \brief Suspends the calling thread until the DoWork() has been called
  //!     the number of times specified by SetDesiredWorkCount().
  void WaitForWorkCount() {
    semaphore_.Wait();
  }

  int work_count() const { return work_count_; }

 private:
  Semaphore semaphore_{0};
  int work_count_ = 0;
  int waiting_for_count_ = -1;

  DISALLOW_COPY_AND_ASSIGN(WorkDelegate);
};

TEST(WorkerThread, DoWork) {
  WorkDelegate delegate;
  WorkerThread thread(0.05, &delegate);

  uint64_t start = ClockMonotonicNanoseconds();

  delegate.SetDesiredWorkCount(2);
  thread.Start(0);
  EXPECT_TRUE(thread.is_running());

  delegate.WaitForWorkCount();
  thread.Stop();
  EXPECT_FALSE(thread.is_running());

// Fuchsia's scheduler is very antagonistic. The assumption that the two work
// items complete in some particular amount of time is strictly incorrect, but
// also somewhat useful. The expected time "should" be ~40-50ms with a work
// interval of 0.05s, but on Fuchsia, 1200ms was observed. So, on Fuchsia, use a
// much larger timeout. See https://crashpad.chromium.org/bug/231.
#if defined(OS_FUCHSIA)
  constexpr uint64_t kUpperBoundTime = 10;
#else
  constexpr uint64_t kUpperBoundTime = 1;
#endif
  EXPECT_GE(kUpperBoundTime * kNanosecondsPerSecond,
            ClockMonotonicNanoseconds() - start);
}

TEST(WorkerThread, StopBeforeDoWork) {
  WorkDelegate delegate;
  WorkerThread thread(1, &delegate);

  thread.Start(15);
  thread.Stop();

  EXPECT_EQ(delegate.work_count(), 0);
}

TEST(WorkerThread, Restart) {
  WorkDelegate delegate;
  WorkerThread thread(0.05, &delegate);

  delegate.SetDesiredWorkCount(1);
  thread.Start(0);
  EXPECT_TRUE(thread.is_running());

  delegate.WaitForWorkCount();
  thread.Stop();
  ASSERT_FALSE(thread.is_running());

  delegate.SetDesiredWorkCount(2);
  thread.Start(0);
  delegate.WaitForWorkCount();
  thread.Stop();
  ASSERT_FALSE(thread.is_running());
}

TEST(WorkerThread, DoWorkNow) {
  WorkDelegate delegate;
  WorkerThread thread(100, &delegate);

  uint64_t start = ClockMonotonicNanoseconds();

  delegate.SetDesiredWorkCount(1);
  thread.Start(0);
  EXPECT_TRUE(thread.is_running());

  delegate.WaitForWorkCount();
  EXPECT_EQ(delegate.work_count(), 1);

  delegate.SetDesiredWorkCount(2);
  thread.DoWorkNow();
  delegate.WaitForWorkCount();
  thread.Stop();
  EXPECT_EQ(delegate.work_count(), 2);

  EXPECT_GE(100 * kNanosecondsPerSecond, ClockMonotonicNanoseconds() - start);
}

TEST(WorkerThread, DoWorkNowAtStart) {
  WorkDelegate delegate;
  WorkerThread thread(100, &delegate);

  uint64_t start = ClockMonotonicNanoseconds();

  delegate.SetDesiredWorkCount(1);
  thread.Start(100);
  EXPECT_TRUE(thread.is_running());

  thread.DoWorkNow();
  delegate.WaitForWorkCount();
  EXPECT_EQ(delegate.work_count(), 1);

  EXPECT_GE(100 * kNanosecondsPerSecond, ClockMonotonicNanoseconds() - start);

  thread.Stop();
  EXPECT_FALSE(thread.is_running());
}

}  // namespace
}  // namespace test
}  // namespace crashpad
