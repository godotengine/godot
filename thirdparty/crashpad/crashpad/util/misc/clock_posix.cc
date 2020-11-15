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

#include <time.h>

#include "base/logging.h"
#include "base/posix/eintr_wrapper.h"
#include "build/build_config.h"

namespace {

constexpr uint64_t kNanosecondsPerSecond = 1E9;

}  // namespace

namespace crashpad {

#if !defined(OS_MACOSX)

uint64_t ClockMonotonicNanoseconds() {
  timespec now;
  int rv = clock_gettime(CLOCK_MONOTONIC, &now);
  DPCHECK(rv == 0) << "clock_gettime";

  return now.tv_sec * kNanosecondsPerSecond + now.tv_nsec;
}

#endif

void SleepNanoseconds(uint64_t nanoseconds) {
  timespec sleep_time;
  sleep_time.tv_sec = nanoseconds / kNanosecondsPerSecond;
  sleep_time.tv_nsec = nanoseconds % kNanosecondsPerSecond;
  int rv = HANDLE_EINTR(nanosleep(&sleep_time, &sleep_time));
  DPCHECK(rv == 0) << "nanosleep";
}

}  // namespace crashpad
