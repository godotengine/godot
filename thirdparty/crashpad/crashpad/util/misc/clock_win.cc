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

#include <windows.h>
#include <sys/types.h>

namespace {

LARGE_INTEGER QpcFrequencyInternal() {
  LARGE_INTEGER frequency;
  QueryPerformanceFrequency(&frequency);
  return frequency;
}

int64_t QpcFrequency() {
  static LARGE_INTEGER frequency = QpcFrequencyInternal();
  return frequency.QuadPart;
}

}  // namespace

namespace crashpad {

uint64_t ClockMonotonicNanoseconds() {
  LARGE_INTEGER time;
  QueryPerformanceCounter(&time);
  int64_t frequency = QpcFrequency();
  int64_t whole_seconds = time.QuadPart / frequency;
  int64_t leftover_ticks = time.QuadPart % frequency;
  constexpr int64_t kNanosecondsPerSecond = static_cast<const int64_t>(1E9);
  return (whole_seconds * kNanosecondsPerSecond) +
         ((leftover_ticks * kNanosecondsPerSecond) / frequency);
}

void SleepNanoseconds(uint64_t nanoseconds) {
  // This is both inaccurate (will be way too long for short sleeps) and
  // incorrect (can sleep for less than requested). But it's what's available
  // without implementing a busy loop.
  constexpr uint64_t kNanosecondsPerMillisecond = static_cast<uint64_t>(1E6);
  Sleep(static_cast<DWORD>(nanoseconds / kNanosecondsPerMillisecond));
}

}  // namespace crashpad
