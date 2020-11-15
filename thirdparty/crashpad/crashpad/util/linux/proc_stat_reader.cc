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

#include "util/linux/proc_stat_reader.h"

#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "base/files/file_path.h"
#include "base/logging.h"
#include "util/file/file_io.h"
#include "util/misc/lexing.h"
#include "util/misc/time.h"

namespace crashpad {

namespace {

long GetClockTicksPerSecond() {
  long clock_ticks_per_s = sysconf(_SC_CLK_TCK);
  if (clock_ticks_per_s <= 0) {
    PLOG(ERROR) << "sysconf";
  }
  return clock_ticks_per_s;
}

}  // namespace

ProcStatReader::ProcStatReader()
    : contents_(), third_column_position_(0), initialized_() {}

ProcStatReader::~ProcStatReader() {}

bool ProcStatReader::Initialize(PtraceConnection* connection, pid_t tid) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  char path[32];
  snprintf(path, arraysize(path), "/proc/%d/stat", tid);
  if (!connection->ReadFileContents(base::FilePath(path), &contents_)) {
    return false;
  }

  // The first column is process ID and the second column is the executable name
  // in parentheses. This class only cares about columns after the second, so
  // find the start of the third here and save it for later.
  // The executable name may have parentheses itself, so find the end of the
  // second column by working backwards to find the last closing parens.
  size_t stat_pos = contents_.rfind(')');
  if (stat_pos == std::string::npos) {
    LOG(ERROR) << "format error";
    return false;
  }

  third_column_position_ = contents_.find(' ', stat_pos);
  if (third_column_position_ == std::string::npos ||
      ++third_column_position_ >= contents_.size()) {
    LOG(ERROR) << "format error";
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool ProcStatReader::UserCPUTime(timeval* user_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return ReadTimeAtIndex(13, user_time);
}

bool ProcStatReader::SystemCPUTime(timeval* system_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return ReadTimeAtIndex(14, system_time);
}

bool ProcStatReader::StartTime(timeval* start_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  timeval time_after_boot;
  if (!ReadTimeAtIndex(21, &time_after_boot)) {
    return false;
  }

  timespec uptime;
  if (clock_gettime(CLOCK_BOOTTIME, &uptime) != 0) {
    PLOG(ERROR) << "clock_gettime";
    return false;
  }

  timespec current_time;
  if (clock_gettime(CLOCK_REALTIME, &current_time) != 0) {
    PLOG(ERROR) << "clock_gettime";
    return false;
  }

  timespec boot_time_ts;
  SubtractTimespec(current_time, uptime, &boot_time_ts);
  timeval boot_time_tv;
  TimespecToTimeval(boot_time_ts, &boot_time_tv);
  timeradd(&boot_time_tv, &time_after_boot, start_time);

  return true;
}

bool ProcStatReader::FindColumn(int col_index, const char** column) const {
  size_t position = third_column_position_;
  for (int index = 2; index < col_index; ++index) {
    position = contents_.find(' ', position);
    if (position == std::string::npos) {
      break;
    }
    ++position;
  }
  if (position >= contents_.size()) {
    LOG(ERROR) << "format error";
    return false;
  }
  *column = &contents_[position];
  return true;
}

bool ProcStatReader::ReadTimeAtIndex(int index, timeval* time_val) const {
  const char* ticks_ptr;
  if (!FindColumn(index, &ticks_ptr)) {
    return false;
  }

  uint64_t ticks;
  if (!AdvancePastNumber<uint64_t>(&ticks_ptr, &ticks)) {
    LOG(ERROR) << "format error";
    return false;
  }

  static long clock_ticks_per_s = GetClockTicksPerSecond();
  if (clock_ticks_per_s <= 0) {
    return false;
  }

  time_val->tv_sec = ticks / clock_ticks_per_s;
  time_val->tv_usec = (ticks % clock_ticks_per_s) *
                      (static_cast<long>(1E6) / clock_ticks_per_s);
  return true;
}

}  // namespace crashpad
