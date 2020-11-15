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

#ifndef CRASHPAD_UTIL_LINUX_PROC_STAT_READER_H_
#define CRASHPAD_UTIL_LINUX_PROC_STAT_READER_H_

#include <stddef.h>
#include <sys/time.h>
#include <sys/types.h>

#include <string>

#include "base/macros.h"
#include "util/linux/ptrace_connection.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

//! \brief Reads the /proc/[pid]/stat file for a thread.
class ProcStatReader {
 public:
  ProcStatReader();
  ~ProcStatReader();

  //! \brief Initializes the reader.
  //!
  //! This method must be successfully called before calling any other.
  //!
  //! \param[in] connection A connection to the process to which the target
  //!     thread belongs.
  //! \param[in] tid The thread ID to read the stat file for.
  bool Initialize(PtraceConnection* connection, pid_t tid);

  //! \brief Determines the time the thread has spent executing in user mode.
  //!
  //! \param[out] user_time The time spent executing in user mode.
  //!
  //! \return `true` on success, with \a user_time set. Otherwise, `false` with
  //!     a message logged.
  bool UserCPUTime(timeval* user_time) const;

  //! \brief Determines the time the thread has spent executing in system mode.
  //!
  //! \param[out] system_time The time spent executing in system mode.
  //!
  //! \return `true` on success, with \a system_time set. Otherwise, `false`
  //!     with a message logged.
  bool SystemCPUTime(timeval* system_time) const;

  //! \brief Determines the target thread’s start time.
  //!
  //! \param[out] start_time The time that the thread started.
  //!
  //! \return `true` on success, with \a start_time set. Otherwise, `false` with
  //!     a message logged.
  bool StartTime(timeval* start_time) const;

 private:
  bool FindColumn(int index, const char** column) const;
  bool ReadTimeAtIndex(int index, timeval* time_val) const;

  std::string contents_;
  size_t third_column_position_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcStatReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_LINUX_PROC_STAT_READER_H_
