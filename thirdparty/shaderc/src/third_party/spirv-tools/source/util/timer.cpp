// Copyright (c) 2018 Google LLC.
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

#if defined(SPIRV_TIMER_ENABLED)

#include "source/util/timer.h"

#include <sys/resource.h>
#include <sys/time.h>
#include <iomanip>
#include <iostream>
#include <string>

namespace spvtools {
namespace utils {

void PrintTimerDescription(std::ostream* out, bool measure_mem_usage) {
  if (out) {
    *out << std::setw(30) << "PASS name" << std::setw(12) << "CPU time"
         << std::setw(12) << "WALL time" << std::setw(12) << "USR time"
         << std::setw(12) << "SYS time";
    if (measure_mem_usage) {
      *out << std::setw(12) << "RSS delta" << std::setw(16) << "PGFault delta";
    }
    *out << std::endl;
  }
}

// Do not change the order of invoking system calls. We want to make CPU/Wall
// time correct as much as possible. Calling functions to get CPU/Wall time must
// closely surround the target code of measuring.
void Timer::Start() {
  if (report_stream_) {
    if (getrusage(RUSAGE_SELF, &usage_before_) == -1)
      usage_status_ |= kGetrusageFailed;
    if (clock_gettime(CLOCK_MONOTONIC, &wall_before_) == -1)
      usage_status_ |= kClockGettimeWalltimeFailed;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_before_) == -1)
      usage_status_ |= kClockGettimeCPUtimeFailed;
  }
}

// The order of invoking system calls is important with the same reason as
// Timer::Start().
void Timer::Stop() {
  if (report_stream_ && usage_status_ == kSucceeded) {
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_after_) == -1)
      usage_status_ |= kClockGettimeCPUtimeFailed;
    if (clock_gettime(CLOCK_MONOTONIC, &wall_after_) == -1)
      usage_status_ |= kClockGettimeWalltimeFailed;
    if (getrusage(RUSAGE_SELF, &usage_after_) == -1)
      usage_status_ = kGetrusageFailed;
  }
}

void Timer::Report(const char* tag) {
  if (!report_stream_) return;

  report_stream_->precision(2);
  *report_stream_ << std::fixed << std::setw(30) << tag;

  if (usage_status_ & kClockGettimeCPUtimeFailed)
    *report_stream_ << std::setw(12) << "Failed";
  else
    *report_stream_ << std::setw(12) << CPUTime();

  if (usage_status_ & kClockGettimeWalltimeFailed)
    *report_stream_ << std::setw(12) << "Failed";
  else
    *report_stream_ << std::setw(12) << WallTime();

  if (usage_status_ & kGetrusageFailed) {
    *report_stream_ << std::setw(12) << "Failed" << std::setw(12) << "Failed";
    if (measure_mem_usage_) {
      *report_stream_ << std::setw(12) << "Failed" << std::setw(12) << "Failed";
    }
  } else {
    *report_stream_ << std::setw(12) << UserTime() << std::setw(12)
                    << SystemTime();
    if (measure_mem_usage_) {
      *report_stream_ << std::fixed << std::setw(12) << RSS() << std::setw(16)
                      << PageFault();
    }
  }
  *report_stream_ << std::endl;
}

}  // namespace utils
}  // namespace spvtools

#endif  // defined(SPIRV_TIMER_ENABLED)
