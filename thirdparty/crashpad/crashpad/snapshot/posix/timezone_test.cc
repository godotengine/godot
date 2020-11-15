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

#include "snapshot/posix/timezone.h"

#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include <string>

#include "base/logging.h"
#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "test/errors.h"

namespace crashpad {
namespace test {
namespace {

class ScopedSetTZ {
 public:
  ScopedSetTZ(const std::string& tz) {
    const char* old_tz = getenv(kTZ);
    old_tz_set_ = old_tz;
    if (old_tz_set_) {
      old_tz_.assign(old_tz);
    }

    EXPECT_EQ(setenv(kTZ, tz.c_str(), 1), 0) << ErrnoMessage("setenv");
    tzset();
  }

  ~ScopedSetTZ() {
    if (old_tz_set_) {
      EXPECT_EQ(setenv(kTZ, old_tz_.c_str(), 1), 0) << ErrnoMessage("setenv");
    } else {
      EXPECT_EQ(unsetenv(kTZ), 0) << ErrnoMessage("unsetenv");
    }
    tzset();
  }

 private:
  std::string old_tz_;
  bool old_tz_set_;

  static constexpr char kTZ[] = "TZ";

  DISALLOW_COPY_AND_ASSIGN(ScopedSetTZ);
};

constexpr char ScopedSetTZ::kTZ[];

TEST(TimeZone, Basic) {
  SystemSnapshot::DaylightSavingTimeStatus dst_status;
  int standard_offset_seconds;
  int daylight_offset_seconds;
  std::string standard_name;
  std::string daylight_name;

  timeval snapshot_time;
  ASSERT_EQ(gettimeofday(&snapshot_time, nullptr), 0);

  internal::TimeZone(snapshot_time,
                     &dst_status,
                     &standard_offset_seconds,
                     &daylight_offset_seconds,
                     &standard_name,
                     &daylight_name);

  // |standard_offset_seconds| gives seconds east of UTC, and |timezone| gives
  // seconds west of UTC.
  EXPECT_EQ(standard_offset_seconds, -timezone);

  // In contemporary usage, most time zones have an integer hour offset from
  // UTC, although several are at a half-hour offset, and two are at 15-minute
  // offsets. Throughout history, other variations existed. See
  // https://www.timeanddate.com/time/time-zones-interesting.html.
  EXPECT_EQ(standard_offset_seconds % (15 * 60), 0)
      << "standard_offset_seconds " << standard_offset_seconds;

  if (dst_status == SystemSnapshot::kDoesNotObserveDaylightSavingTime) {
    EXPECT_EQ(daylight_offset_seconds, standard_offset_seconds);
    EXPECT_EQ(daylight_name, standard_name);
  } else {
    EXPECT_EQ(daylight_offset_seconds % (15 * 60), 0)
        << "daylight_offset_seconds " << daylight_offset_seconds;

    // In contemporary usage, dst_delta_seconds will almost always be one hour,
    // except for Lord Howe Island, Australia, which uses a 30-minute delta.
    // Throughout history, other variations existed. See
    // https://www.timeanddate.com/time/dst/.
    int dst_delta_seconds = daylight_offset_seconds - standard_offset_seconds;
    if (dst_delta_seconds != 60 * 60 && dst_delta_seconds != 30 * 60) {
      FAIL() << "dst_delta_seconds " << dst_delta_seconds;
    }

    EXPECT_NE(standard_name, daylight_name);
  }

  // Test a variety of time zones. Some of these observe daylight saving time,
  // some don’t. Some used to but no longer do. Some have uncommon UTC offsets.
  // standard_name and daylight_name can be nullptr where no name exists to
  // verify, as may happen when some versions of the timezone database carry
  // invented names and others do not.
  static constexpr struct {
    const char* tz;
    bool observes_dst;
    float standard_offset_hours;
    float daylight_offset_hours;
    const char* standard_name;
    const char* daylight_name;
  } kTestTimeZones[] = {
      {"America/Anchorage", true, -9, -8, "AKST", "AKDT"},
      {"America/Chicago", true, -6, -5, "CST", "CDT"},
      {"America/Denver", true, -7, -6, "MST", "MDT"},
      {"America/Halifax", true, -4, -3, "AST", "ADT"},
      {"America/Los_Angeles", true, -8, -7, "PST", "PDT"},
      {"America/New_York", true, -5, -4, "EST", "EDT"},
      {"America/Phoenix", false, -7, -7, "MST", "MST"},
      {"Asia/Karachi", false, 5, 5, "PKT", "PKT"},
      {"Asia/Kolkata", false, 5.5, 5.5, "IST", "IST"},
      {"Asia/Shanghai", false, 8, 8, "CST", "CST"},
      {"Asia/Tokyo", false, 9, 9, "JST", "JST"},

      // Australian timezone names have an optional "A" prefix, which is
      // present for glibc and macOS, but missing on Android.
      {"Australia/Adelaide", true, 9.5, 10.5, nullptr, nullptr},
      {"Australia/Brisbane", false, 10, 10, nullptr, nullptr},
      {"Australia/Darwin", false, 9.5, 9.5, nullptr, nullptr},
      {"Australia/Eucla", false, 8.75, 8.75, nullptr, nullptr},
      {"Australia/Lord_Howe", true, 10.5, 11, nullptr, nullptr},
      {"Australia/Perth", false, 8, 8, nullptr, nullptr},
      {"Australia/Sydney", true, 10, 11, nullptr, nullptr},

      {"Europe/Bucharest", true, 2, 3, "EET", "EEST"},
      {"Europe/London", true, 0, 1, "GMT", "BST"},
      {"Europe/Paris", true, 1, 2, "CET", "CEST"},
      {"Europe/Reykjavik", false, 0, 0, nullptr, nullptr},
      {"Pacific/Auckland", true, 12, 13, "NZST", "NZDT"},
      {"Pacific/Honolulu", false, -10, -10, "HST", "HST"},
      {"UTC", false, 0, 0, "UTC", "UTC"},
  };

  for (size_t index = 0; index < arraysize(kTestTimeZones); ++index) {
    const auto& test_time_zone = kTestTimeZones[index];
    const char* tz = test_time_zone.tz;
    SCOPED_TRACE(base::StringPrintf("index %zu, tz %s", index, tz));

    {
      ScopedSetTZ set_tz(tz);
      internal::TimeZone(snapshot_time,
                         &dst_status,
                         &standard_offset_seconds,
                         &daylight_offset_seconds,
                         &standard_name,
                         &daylight_name);
    }

    EXPECT_PRED2(
        [](SystemSnapshot::DaylightSavingTimeStatus dst, bool observes) {
          return (dst != SystemSnapshot::kDoesNotObserveDaylightSavingTime) ==
                 observes;
        },
        dst_status,
        test_time_zone.observes_dst);

    EXPECT_EQ(standard_offset_seconds,
              test_time_zone.standard_offset_hours * 60 * 60);
    EXPECT_EQ(daylight_offset_seconds,
              test_time_zone.daylight_offset_hours * 60 * 60);
    if (test_time_zone.standard_name) {
      EXPECT_EQ(standard_name, test_time_zone.standard_name);
    }
    if (test_time_zone.daylight_name) {
      EXPECT_EQ(daylight_name, test_time_zone.daylight_name);
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
