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

#include "util/thread/thread.h"

#include "base/macros.h"
#include "gtest/gtest.h"
#include "util/synchronization/semaphore.h"

namespace crashpad {
namespace test {
namespace {

class NoopThread : public Thread {
 public:
  NoopThread() {}
  ~NoopThread() override {}

 private:
  void ThreadMain() override {}

  DISALLOW_COPY_AND_ASSIGN(NoopThread);
};

class WaitThread : public Thread {
 public:
  explicit WaitThread(Semaphore* semaphore) : semaphore_(semaphore) {}
  ~WaitThread() override {}

 private:
  void ThreadMain() override { semaphore_->Wait(); }

  Semaphore* semaphore_;

  DISALLOW_COPY_AND_ASSIGN(WaitThread);
};

class JoinAndSignalThread : public Thread {
 public:
  JoinAndSignalThread(Thread* thread, Semaphore* semaphore)
      : thread_(thread), semaphore_(semaphore) {}
  ~JoinAndSignalThread() override {}

 private:
  void ThreadMain() override {
    thread_->Join();
    semaphore_->Signal();
  }

  Thread* thread_;
  Semaphore* semaphore_;

  DISALLOW_COPY_AND_ASSIGN(JoinAndSignalThread);
};

TEST(ThreadTest, NoStart) {
  NoopThread thread;
}

TEST(ThreadTest, Start) {
  NoopThread thread;
  thread.Start();
  thread.Join();
}

TEST(ThreadTest, JoinBlocks) {
  Semaphore unblock_wait_thread_semaphore(0);
  Semaphore join_completed_semaphore(0);
  WaitThread wait_thread(&unblock_wait_thread_semaphore);
  wait_thread.Start();
  JoinAndSignalThread join_and_signal_thread(&wait_thread,
                                             &join_completed_semaphore);
  join_and_signal_thread.Start();
  // join_completed_semaphore will be signaled when wait_thread.Join() returns
  // (in JoinAndSignalThread::ThreadMain). Since wait_thread is blocking on
  // unblock_wait_thread_semaphore, we don't expect the Join to return yet. We
  // wait up to 100ms to give a broken implementation of Thread::Join a chance
  // to return.
  ASSERT_FALSE(join_completed_semaphore.TimedWait(.1));
  unblock_wait_thread_semaphore.Signal();
  join_completed_semaphore.Wait();
  join_and_signal_thread.Join();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
