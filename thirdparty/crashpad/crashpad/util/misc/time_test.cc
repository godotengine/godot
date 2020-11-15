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

#include "util/misc/time.h"

#include <limits>

#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(Time, TimespecArithmetic) {
  timespec ts1, ts2, result;
  ts1.tv_sec = ts2.tv_sec = 1;
  ts1.tv_nsec = ts2.tv_nsec = kNanosecondsPerSecond / 2;
  AddTimespec(ts1, ts2, &result);
  EXPECT_EQ(result.tv_sec, 3);
  EXPECT_EQ(result.tv_nsec, 0);

  ts1.tv_sec = 2;
  ts1.tv_nsec = 0;
  ts2.tv_sec = 1;
  ts2.tv_nsec = 1;
  SubtractTimespec(ts1, ts2, &result);
  EXPECT_EQ(result.tv_sec, 0);
  EXPECT_EQ(result.tv_nsec, long{kNanosecondsPerSecond - 1});
}

TEST(Time, TimeConversions) {
  // On July 30th, 2014 at 9:15 PM GMT+0, the Crashpad git repository was born.
  // (nanoseconds are approximate)
  constexpr timespec kCrashpadBirthdate = {
    /* .tv_sec= */ 1406754914,
    /* .tv_nsec= */ 32487
  };

  timeval timeval_birthdate;
  ASSERT_TRUE(TimespecToTimeval(kCrashpadBirthdate, &timeval_birthdate));
  EXPECT_EQ(timeval_birthdate.tv_sec, kCrashpadBirthdate.tv_sec);
  EXPECT_EQ(timeval_birthdate.tv_usec, kCrashpadBirthdate.tv_nsec / 1000);

  timespec timespec_birthdate;
  TimevalToTimespec(timeval_birthdate, &timespec_birthdate);
  EXPECT_EQ(timespec_birthdate.tv_sec, kCrashpadBirthdate.tv_sec);
  EXPECT_EQ(timespec_birthdate.tv_nsec,
            kCrashpadBirthdate.tv_nsec - (kCrashpadBirthdate.tv_nsec % 1000));

  constexpr timespec kEndOfTime = {
    /* .tv_sec= */ std::numeric_limits<decltype(timespec::tv_sec)>::max(),
    /* .tv_nsec= */ 0
  };

  timeval end_of_timeval;
  if (std::numeric_limits<decltype(timespec::tv_sec)>::max() >
      std::numeric_limits<decltype(timeval::tv_sec)>::max()) {
    EXPECT_FALSE(TimespecToTimeval(kEndOfTime, &end_of_timeval));
  } else {
    EXPECT_TRUE(TimespecToTimeval(kEndOfTime, &end_of_timeval));
  }

#if defined(OS_WIN)
  constexpr uint64_t kBirthdateFiletimeIntervals = 130512285140000324;
  FILETIME filetime_birthdate;
  filetime_birthdate.dwLowDateTime = 0xffffffff & kBirthdateFiletimeIntervals;
  filetime_birthdate.dwHighDateTime = kBirthdateFiletimeIntervals >> 32;

  FILETIME filetime = TimespecToFiletimeEpoch(kCrashpadBirthdate);
  EXPECT_EQ(filetime.dwLowDateTime, filetime_birthdate.dwLowDateTime);
  EXPECT_EQ(filetime.dwHighDateTime, filetime_birthdate.dwHighDateTime);

  timespec_birthdate = FiletimeToTimespecEpoch(filetime_birthdate);
  EXPECT_EQ(timespec_birthdate.tv_sec, kCrashpadBirthdate.tv_sec);
  EXPECT_EQ(timespec_birthdate.tv_nsec,
            kCrashpadBirthdate.tv_nsec - kCrashpadBirthdate.tv_nsec % 100);

  timeval_birthdate = FiletimeToTimevalEpoch(filetime_birthdate);
  EXPECT_EQ(timeval_birthdate.tv_sec, kCrashpadBirthdate.tv_sec);
  EXPECT_EQ(timeval_birthdate.tv_usec, kCrashpadBirthdate.tv_nsec / 1000);

  FILETIME elapsed_filetime;
  elapsed_filetime.dwLowDateTime = 0;
  elapsed_filetime.dwHighDateTime = 0;
  timeval elapsed_timeval = FiletimeToTimevalInterval(elapsed_filetime);
  EXPECT_EQ(elapsed_timeval.tv_sec, 0);
  EXPECT_EQ(elapsed_timeval.tv_usec, 0);

  elapsed_filetime.dwLowDateTime = 9;
  elapsed_timeval = FiletimeToTimevalInterval(elapsed_filetime);
  EXPECT_EQ(elapsed_timeval.tv_sec, 0);
  EXPECT_EQ(elapsed_timeval.tv_usec, 0);

  elapsed_filetime.dwLowDateTime = 10;
  elapsed_timeval = FiletimeToTimevalInterval(elapsed_filetime);
  EXPECT_EQ(elapsed_timeval.tv_sec, 0);
  EXPECT_EQ(elapsed_timeval.tv_usec, 1);

  elapsed_filetime.dwHighDateTime = 1;
  elapsed_filetime.dwLowDateTime = 0;
  elapsed_timeval = FiletimeToTimevalInterval(elapsed_filetime);
  EXPECT_EQ(elapsed_timeval.tv_sec, 429);
  EXPECT_EQ(elapsed_timeval.tv_usec, 496729);
#endif  // OS_WIN
}

#if defined(OS_WIN)

TEST(Time, GetTimeOfDay) {
  timeval t;
  GetTimeOfDay(&t);
  time_t approx_now = time(nullptr);
  EXPECT_GE(approx_now, t.tv_sec);
  EXPECT_LT(approx_now - 100, t.tv_sec);
}

#endif  // OS_WIN

}  // namespace
}  // namespace test
}  // namespace crashpad
