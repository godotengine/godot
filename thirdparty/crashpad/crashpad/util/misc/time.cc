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

#include "util/misc/time.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {

void AddTimespec(const timespec& ts1, const timespec& ts2, timespec* result) {
  result->tv_sec = ts1.tv_sec + ts2.tv_sec;
  result->tv_nsec = ts1.tv_nsec + ts2.tv_nsec;
  if (result->tv_nsec >= long{kNanosecondsPerSecond}) {
    ++result->tv_sec;
    result->tv_nsec -= kNanosecondsPerSecond;
  }
}

void SubtractTimespec(const timespec& t1,
                      const timespec& t2,
                      timespec* result) {
  result->tv_sec = t1.tv_sec - t2.tv_sec;
  result->tv_nsec = t1.tv_nsec - t2.tv_nsec;
  if (result->tv_nsec < 0) {
    result->tv_sec -= 1;
    result->tv_nsec += kNanosecondsPerSecond;
  }
}

bool TimespecToTimeval(const timespec& ts, timeval* tv) {
  tv->tv_usec = ts.tv_nsec / 1000;

  // timespec::tv_sec and timeval::tv_sec should generally both be of type
  // time_t, however, on Windows, timeval::tv_sec is declared as a long, which
  // may be smaller than a time_t.
  return AssignIfInRange(&tv->tv_sec, ts.tv_sec);
}

void TimevalToTimespec(const timeval& tv, timespec* ts) {
  ts->tv_sec = tv.tv_sec;
  ts->tv_nsec = tv.tv_usec * 1000;
}

}  // namespace crashpad
