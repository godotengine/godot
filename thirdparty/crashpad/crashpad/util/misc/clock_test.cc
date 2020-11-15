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

#include "util/misc/clock.h"

#include <sys/types.h>

#include <algorithm>

#include "base/format_macros.h"
#include "base/logging.h"
#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(Clock, ClockMonotonicNanoseconds) {
  uint64_t start = ClockMonotonicNanoseconds();
  EXPECT_GT(start, 0u);

  uint64_t now = start;
  for (size_t iteration = 0; iteration < 10; ++iteration) {
    uint64_t last = now;
    now = ClockMonotonicNanoseconds();

    // Use EXPECT_GE instead of EXPECT_GT, because there are no guarantees about
    // the clock’s resolution.
    EXPECT_GE(now, last);
  }

#if !defined(OS_WIN)  // No SleepNanoseconds implemented on Windows.
  // SleepNanoseconds() should sleep for at least the value of the clock’s
  // resolution, so the clock’s value should definitely increase after a sleep.
  // EXPECT_GT can be used instead of EXPECT_GE after the sleep.
  SleepNanoseconds(1);
  now = ClockMonotonicNanoseconds();
  EXPECT_GT(now, start);
#endif  // OS_WIN
}

#if !defined(OS_WIN)  // No SleepNanoseconds implemented on Windows.

void TestSleepNanoseconds(uint64_t nanoseconds) {
  uint64_t start = ClockMonotonicNanoseconds();

  SleepNanoseconds(nanoseconds);

  uint64_t end = ClockMonotonicNanoseconds();
  uint64_t diff = end - start;

  // |nanoseconds| is the lower bound for the actual amount of time spent
  // sleeping.
  EXPECT_GE(diff, nanoseconds);

  // It’s difficult to set an upper bound for the time spent sleeping, and
  // attempting to do so results in a flaky test.
}

TEST(Clock, SleepNanoseconds) {
  static constexpr uint64_t kTestData[] = {
      0,
      1,
      static_cast<uint64_t>(1E3),  // 1 microsecond
      static_cast<uint64_t>(1E4),  // 10 microseconds
      static_cast<uint64_t>(1E5),  // 100 microseconds
      static_cast<uint64_t>(1E6),  // 1 millisecond
      static_cast<uint64_t>(1E7),  // 10 milliseconds
      static_cast<uint64_t>(2E7),  // 20 milliseconds
      static_cast<uint64_t>(5E7),  // 50 milliseconds
  };

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    const uint64_t nanoseconds = kTestData[index];
    SCOPED_TRACE(base::StringPrintf(
        "index %zu, nanoseconds %" PRIu64, index, nanoseconds));

    TestSleepNanoseconds(nanoseconds);
  }
}

#endif  // OS_WIN

}  // namespace
}  // namespace test
}  // namespace crashpad
