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

#ifndef CRASHPAD_UTIL_MISC_TIME_H_
#define CRASHPAD_UTIL_MISC_TIME_H_

#include <stdint.h>
#include <sys/time.h>
#include <time.h>

#include "build/build_config.h"

#if defined(OS_WIN)
#include <windows.h>
#endif

namespace crashpad {

constexpr uint64_t kNanosecondsPerSecond = static_cast<uint64_t>(1E9);

//! \brief Add `timespec` \a ts1 and \a ts2 and return the result in \a result.
void AddTimespec(const timespec& ts1, const timespec& ts2, timespec* result);

//! \brief Subtract `timespec` \a ts2 from \a ts1 and return the result in \a
//!     result.
void SubtractTimespec(const timespec& ts1,
                      const timespec& ts2,
                      timespec* result);

//! \brief Convert the timespec \a ts to a timeval \a tv.
//! \return `true` if the assignment is possible without truncation.
bool TimespecToTimeval(const timespec& ts, timeval* tv);

//! \brief Convert the timeval \a tv to a timespec \a ts.
void TimevalToTimespec(const timeval& tv, timespec* ts);

#if defined(OS_WIN) || DOXYGEN

//! \brief Convert a `timespec` to a Windows `FILETIME`, converting from POSIX
//!     epoch to Windows epoch.
FILETIME TimespecToFiletimeEpoch(const timespec& ts);

//! \brief Convert a Windows `FILETIME` to `timespec`, converting from Windows
//!     epoch to POSIX epoch.
timespec FiletimeToTimespecEpoch(const FILETIME& filetime);

//! \brief Convert Windows `FILETIME` to `timeval`, converting from Windows
//!     epoch to POSIX epoch.
timeval FiletimeToTimevalEpoch(const FILETIME& filetime);

//! \brief Convert Windows `FILETIME` to `timeval`, treating the values as
//!     an interval of elapsed time.
timeval FiletimeToTimevalInterval(const FILETIME& filetime);

//! \brief Similar to POSIX `gettimeofday()`, gets the current system time in
//!     UTC.
void GetTimeOfDay(timeval* tv);

#endif  // OS_WIN

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_TIME_H_
