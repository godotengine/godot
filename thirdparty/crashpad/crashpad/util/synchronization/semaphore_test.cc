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

#include <sys/types.h>

#include "base/macros.h"
#include "gtest/gtest.h"

#if defined(OS_POSIX)
#include <pthread.h>
#endif  // OS_POSIX

namespace crashpad {
namespace test {
namespace {

TEST(Semaphore, Simple) {
  Semaphore semaphore(1);
  semaphore.Wait();
  semaphore.Signal();
}

TEST(Semaphore, TimedWait) {
  Semaphore semaphore(0);
  semaphore.Signal();
  EXPECT_TRUE(semaphore.TimedWait(0.01));  // 10ms
}

TEST(Semaphore, TimedWaitTimeout) {
  Semaphore semaphore(0);
  semaphore.Signal();
  constexpr double kTenMs = 0.01;
  EXPECT_TRUE(semaphore.TimedWait(kTenMs));
  EXPECT_FALSE(semaphore.TimedWait(kTenMs));
}

TEST(Semaphore, TimedWaitInfinite_0) {
  Semaphore semaphore(0);
  semaphore.Signal();
  EXPECT_TRUE(semaphore.TimedWait(std::numeric_limits<double>::infinity()));
}

TEST(Semaphore, TimedWaitInfinite_1) {
  Semaphore semaphore(1);
  EXPECT_TRUE(semaphore.TimedWait(std::numeric_limits<double>::infinity()));
  semaphore.Signal();
}

struct ThreadMainInfo {
#if defined(OS_POSIX)
  pthread_t pthread;
#elif defined(OS_WIN)
  HANDLE thread;
#endif
  Semaphore* semaphore;
  size_t iterations;
};

#if defined(OS_POSIX)
void*
#elif defined(OS_WIN)
DWORD WINAPI
#endif  // OS_POSIX
ThreadMain(void* argument) {
  ThreadMainInfo* info = reinterpret_cast<ThreadMainInfo*>(argument);
  for (size_t iteration = 0; iteration < info->iterations; ++iteration) {
    info->semaphore->Wait();
  }
#if defined(OS_POSIX)
  return nullptr;
#elif defined(OS_WIN)
  return 0;
#endif  // OS_POSIX
}

void StartThread(ThreadMainInfo* info) {
#if defined(OS_POSIX)
  int rv = pthread_create(&info->pthread, nullptr, ThreadMain, info);
  ASSERT_EQ(rv, 0) << "pthread_create";
#elif defined(OS_WIN)
  info->thread = CreateThread(nullptr, 0, ThreadMain, info, 0, nullptr);
  ASSERT_NE(info->thread, nullptr) << "CreateThread";
#endif  // OS_POSIX
}

void JoinThread(ThreadMainInfo* info) {
#if defined(OS_POSIX)
  int rv = pthread_join(info->pthread, nullptr);
  EXPECT_EQ(rv, 0) << "pthread_join";
#elif defined(OS_WIN)
  DWORD result = WaitForSingleObject(info->thread, INFINITE);
  EXPECT_EQ(result, WAIT_OBJECT_0) << "WaitForSingleObject";
#endif  // OS_POSIX
}

TEST(Semaphore, Threaded) {
  Semaphore semaphore(0);
  ThreadMainInfo info;
  info.semaphore = &semaphore;
  info.iterations = 1;

  ASSERT_NO_FATAL_FAILURE(StartThread(&info));

  semaphore.Signal();

  JoinThread(&info);
}

TEST(Semaphore, TenThreaded) {
  // This test has a smaller initial value (5) than threads contending for these
  // resources (10), and the threads each try to obtain the resource a different
  // number of times.
  Semaphore semaphore(5);
  ThreadMainInfo info[10];
  size_t iterations = 0;
  for (size_t index = 0; index < arraysize(info); ++index) {
    info[index].semaphore = &semaphore;
    info[index].iterations = index;
    iterations += info[index].iterations;

    ASSERT_NO_FATAL_FAILURE(StartThread(&info[index]));
  }

  for (size_t iteration = 0; iteration < iterations; ++iteration) {
    semaphore.Signal();
  }

  for (size_t index = 0; index < arraysize(info); ++index) {
    JoinThread(&info[index]);
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
