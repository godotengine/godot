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

#ifndef CRASHPAD_SNAPSHOT_POSIX_TIMEZONE_H_
#define CRASHPAD_SNAPSHOT_POSIX_TIMEZONE_H_

#include <sys/time.h>

#include <string>

#include "snapshot/system_snapshot.h"

namespace crashpad {
namespace internal {

//! \brief Returns time zone information from the snapshot system, based on
//!     its locale configuration and \a snapshot_time.
//!
//! \param[in] snapshot_time The time to use collect daylight saving time status
//!     for, given in time since Epoch.
//! \param[out] dst_status Whether the location observes daylight saving time,
//!     and if so, whether it or standard time is currently being observed.
//! \param[out] standard_offset_seconds The number of seconds that the
//!     location’s time zone is east (ahead) of UTC during standard time.
//! \param[out] daylight_offset_seconds The number of seconds that the
//!     location’s time zone is east (ahead) of UTC during daylight saving.
//!     time.
//! \param[out] standard_name The name of the time zone while standard time is
//!     being observed.
//! \param[out] daylight_name The name of the time zone while daylight saving
//!     time is being observed.
//!
//! \sa SystemSnapshot::TimeZone
void TimeZone(const timeval& snapshot_time,
              SystemSnapshot::DaylightSavingTimeStatus* dst_status,
              int* standard_offset_seconds,
              int* daylight_offset_seconds,
              std::string* standard_name,
              std::string* daylight_name);

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_POSIX_TIMEZONE_H_
