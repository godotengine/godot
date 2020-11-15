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

#include <stddef.h>
#include <time.h>

#include "base/logging.h"
#include "build/build_config.h"

namespace crashpad {
namespace internal {

void TimeZone(const timeval& snapshot_time,
              SystemSnapshot::DaylightSavingTimeStatus* dst_status,
              int* standard_offset_seconds,
              int* daylight_offset_seconds,
              std::string* standard_name,
              std::string* daylight_name) {
  tzset();

  tm local;
  PCHECK(localtime_r(&snapshot_time.tv_sec, &local)) << "localtime_r";

  *standard_name = tzname[0];

  bool found_transition = false;
  long probe_gmtoff = local.tm_gmtoff;
#if defined(OS_ANDROID)
  // Some versions of the timezone database on Android have incorrect
  // information (e.g. Asia/Kolkata and Pacific/Honolulu). These timezones set
  // daylight to a non-zero value and return incorrect, >= 0 values for tm_isdst
  // in the probes below. If tzname[1] is set to a bogus value, assume the
  // timezone does not actually use daylight saving time.
  if (daylight && strncmp(tzname[1], "_TZif", 5) != 0) {
#else
  if (daylight) {
#endif
    // Scan forward and backward, one month at a time, looking for an instance
    // when the observance of daylight saving time is different than it is in
    // |local|. It’s possible that no such instance will be found even with
    // |daylight| set. This can happen in locations where daylight saving time
    // was once observed or is expected to be observed in the future, but where
    // no transitions to or from daylight saving time occurred or will occur
    // within a year of the current date. Arizona, which last observed daylight
    // saving time in 1967, is an example.
    static constexpr int kMonthDeltas[] =
        {0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6,
         7, -7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12};
    for (size_t index = 0;
         index < arraysize(kMonthDeltas) && !found_transition;
         ++index) {
      // Look at a day of each month at local noon. Set tm_isdst to -1 to avoid
      // giving mktime() any hints about whether to consider daylight saving
      // time in effect. mktime() accepts values of tm_mon that are outside of
      // its normal range and behaves as expected: if tm_mon is -1, it
      // references December of the preceding year, and if it is 12, it
      // references January of the following year.
      tm probe_tm = {};
      probe_tm.tm_hour = 12;
      probe_tm.tm_mday = std::min(local.tm_mday, 28);
      probe_tm.tm_mon = local.tm_mon + kMonthDeltas[index];
      probe_tm.tm_year = local.tm_year;
      probe_tm.tm_isdst = -1;
      if (mktime(&probe_tm) == -1) {
        PLOG(WARNING) << "mktime";
        continue;
      }
      if (probe_tm.tm_isdst < 0 || local.tm_isdst < 0) {
        LOG(WARNING) << "dst status not available";
        continue;
      }
      if (probe_tm.tm_isdst != local.tm_isdst) {
        found_transition = true;
        probe_gmtoff = probe_tm.tm_gmtoff;
      }
    }
  }

  if (found_transition) {
    *daylight_name = tzname[1];
    if (!local.tm_isdst) {
      *dst_status = SystemSnapshot::kObservingStandardTime;
      *standard_offset_seconds = local.tm_gmtoff;
      *daylight_offset_seconds = probe_gmtoff;
    } else {
      *dst_status = SystemSnapshot::kObservingDaylightSavingTime;
      *standard_offset_seconds = probe_gmtoff;
      *daylight_offset_seconds = local.tm_gmtoff;
    }
  } else {
    *daylight_name = tzname[0];
    *dst_status = SystemSnapshot::kDoesNotObserveDaylightSavingTime;
#if defined(OS_ANDROID)
    // timezone is more reliably set correctly on Android.
    *standard_offset_seconds = -timezone;
    *daylight_offset_seconds = -timezone;
#else
    *standard_offset_seconds = local.tm_gmtoff;
    *daylight_offset_seconds = local.tm_gmtoff;
#endif  // OS_ANDROID
  }
}

}  // namespace internal
}  // namespace crashpad
