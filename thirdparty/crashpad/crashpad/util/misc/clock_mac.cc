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

#include <mach/mach_time.h>

#include "base/mac/mach_logging.h"

namespace {

mach_timebase_info_data_t* TimebaseInternal() {
  mach_timebase_info_data_t* timebase_info = new mach_timebase_info_data_t;
  kern_return_t kr = mach_timebase_info(timebase_info);
  MACH_CHECK(kr == KERN_SUCCESS, kr) << "mach_timebase_info";
  return timebase_info;
}

mach_timebase_info_data_t* Timebase() {
  static mach_timebase_info_data_t* timebase_info = TimebaseInternal();
  return timebase_info;
}

}  // namespace

namespace crashpad {

uint64_t ClockMonotonicNanoseconds() {
  uint64_t absolute_time = mach_absolute_time();
  mach_timebase_info_data_t* timebase_info = Timebase();
  return absolute_time * timebase_info->numer / timebase_info->denom;
}

}  // namespace crashpad
