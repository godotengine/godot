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

#ifndef CRASHPAD_UTIL_MISC_CLOCK_H_
#define CRASHPAD_UTIL_MISC_CLOCK_H_

#include <stdint.h>

#include "build/build_config.h"

namespace crashpad {

//! \brief Returns the value of the system’s monotonic clock.
//!
//! The monotonic clock is a tick counter whose epoch is unspecified. It is a
//! monotonically-increasing clock that cannot be set, and never jumps backwards
//! on a running system. The monotonic clock may stop while the system is
//! sleeping, and it may be reset when the system starts up. This clock is
//! suitable for computing durations of events. Subject to the underlying
//! clock’s resolution, successive calls to this function will result in a
//! series of increasing values.
//!
//! \return The value of the system’s monotonic clock, in nanoseconds.
uint64_t ClockMonotonicNanoseconds();

//! \brief Sleeps for the specified duration.
//!
//! \param[in] nanoseconds The number of nanoseconds to sleep. The actual sleep
//!     may be slightly longer due to latencies and timer resolution.
//!
//! On POSIX, this function is resilient against the underlying `nanosleep()`
//! system call being interrupted by a signal.
void SleepNanoseconds(uint64_t nanoseconds);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_CLOCK_H_
