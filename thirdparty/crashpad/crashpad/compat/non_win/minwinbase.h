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

#ifndef CRASHPAD_COMPAT_NON_WIN_MINWINBASE_H_
#define CRASHPAD_COMPAT_NON_WIN_MINWINBASE_H_

#include <stdint.h>

//! \brief Represents a date and time.
struct SYSTEMTIME {
  //! \brief The year, represented fully.
  //!
  //! The year 2014 would be represented in this field as `2014`.
  uint16_t wYear;

  //! \brief The month of the year, `1` for January and `12` for December.
  uint16_t wMonth;

  //! \brief The day of the week, `0` for Sunday and `6` for Saturday.
  uint16_t wDayOfWeek;

  //! \brief The day of the month, `1` through `31`.
  uint16_t wDay;

  //! \brief The hour of the day, `0` through `23`.
  uint16_t wHour;

  //! \brief The minute of the hour, `0` through `59`.
  uint16_t wMinute;

  //! \brief The second of the minute, `0` through `60`.
  uint16_t wSecond;

  //! \brief The millisecond of the second, `0` through `999`.
  uint16_t wMilliseconds;
};

#endif  // CRASHPAD_COMPAT_NON_WIN_MINWINBASE_H_
