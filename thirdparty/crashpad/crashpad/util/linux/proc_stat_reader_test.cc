// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "util/linux/proc_stat_reader.h"

#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "base/logging.h"
#include "gtest/gtest.h"
#include "test/linux/fake_ptrace_connection.h"
#include "util/thread/thread.h"

namespace crashpad {
namespace test {
namespace {

TEST(ProcStatReader, Basic) {
  FakePtraceConnection connection;
  ASSERT_TRUE(connection.Initialize(getpid()));

  ProcStatReader stat;
  ASSERT_TRUE(stat.Initialize(&connection, getpid()));

  timeval start_time;
  ASSERT_TRUE(stat.StartTime(&start_time));

  time_t now;
  time(&now);
  EXPECT_LE(start_time.tv_sec, now);

  time_t elapsed_sec = now - start_time.tv_sec;

  timeval user_time;
  ASSERT_TRUE(stat.UserCPUTime(&user_time));
  EXPECT_LE(user_time.tv_sec, elapsed_sec);

  timeval system_time;
  ASSERT_TRUE(stat.SystemCPUTime(&system_time));
  EXPECT_LE(system_time.tv_sec, elapsed_sec);
}

pid_t gettid() {
  return syscall(SYS_gettid);
}

void GetStartTime(timeval* start_time) {
  FakePtraceConnection connection;
  ASSERT_TRUE(connection.Initialize(getpid()));

  ProcStatReader stat;
  ASSERT_TRUE(stat.Initialize(&connection, gettid()));
  ASSERT_TRUE(stat.StartTime(start_time));
}

class StatTimeThread : public Thread {
 public:
  StatTimeThread(timeval* start_time) : start_time_(start_time) {}

 private:
  void ThreadMain() override { GetStartTime(start_time_); }
  timeval* start_time_;
};

TEST(ProcStatReader, Threads) {
  timeval main_time;
  ASSERT_NO_FATAL_FAILURE(GetStartTime(&main_time));

  timeval thread_time;
  StatTimeThread thread(&thread_time);
  thread.Start();
  ASSERT_NO_FATAL_FAILURE(thread.Join());

  EXPECT_PRED4(
      [](time_t main_sec,
         suseconds_t main_usec,
         time_t thread_sec,
         suseconds_t thread_usec) {
        return (thread_sec > main_sec) ||
               (thread_sec == main_sec && thread_usec > main_usec);
      },
      main_time.tv_sec,
      main_time.tv_usec,
      thread_time.tv_sec,
      thread_time.tv_usec);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
