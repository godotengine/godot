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

#ifndef CRASHPAD_COMPAT_NON_WIN_TIMEZONEAPI_H_
#define CRASHPAD_COMPAT_NON_WIN_TIMEZONEAPI_H_

#include <stdint.h>

#include "base/strings/string16.h"
#include "compat/non_win/minwinbase.h"

//! \brief Information about a time zone and its daylight saving rules.
struct TIME_ZONE_INFORMATION {
  //! \brief The number of minutes west of UTC.
  int32_t Bias;

  //! \brief The UTF-16-encoded name of the time zone when observing standard
  //!     time.
  base::char16 StandardName[32];

  //! \brief The date and time to switch from daylight saving time to standard
  //!     time.
  //!
  //! This can be a specific time, or with SYSTEMTIME::wYear set to `0`, it can
  //! reflect an annual recurring transition. In that case, SYSTEMTIME::wDay in
  //! the range `1` to `5` is interpreted as the given occurrence of
  //! SYSTEMTIME::wDayOfWeek within the month, `1` being the first occurrence
  //! and `5` being the last (even if there are fewer than 5).
  SYSTEMTIME StandardDate;

  //! \brief The bias relative to #Bias to be applied when observing standard
  //!     time.
  int32_t StandardBias;

  //! \brief The UTF-16-encoded name of the time zone when observing daylight
  //!     saving time.
  base::char16 DaylightName[32];

  //! \brief The date and time to switch from standard time to daylight saving
  //!     time.
  //!
  //! This field is specified in the same manner as #StandardDate.
  SYSTEMTIME DaylightDate;

  //! \brief The bias relative to #Bias to be applied when observing daylight
  //!     saving time.
  int32_t DaylightBias;
};

#endif  // CRASHPAD_COMPAT_NON_WIN_TIMEZONEAPI_H_
