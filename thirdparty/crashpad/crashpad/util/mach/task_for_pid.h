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

#ifndef CRASHPAD_UTIL_MACH_TASK_FOR_PID_H_
#define CRASHPAD_UTIL_MACH_TASK_FOR_PID_H_

#include <mach/mach.h>
#include <sys/types.h>

namespace crashpad {

//! \brief Wraps `task_for_pid()`.
//!
//! This function exists to support `task_for_pid()` access checks in a setuid
//! environment. Normally, `task_for_pid()` can only return an arbitrary task’s
//! port when running as root or when taskgated(8) approves. When not running as
//! root, a series of access checks are perfomed to ensure that the running
//! process has permission to obtain the other process’ task port.
//!
//! It is possible to make an executable setuid root to give it broader
//! `task_for_pid()` access by bypassing taskgated(8) checks, but this also has
//! the effect of bypassing the access checks, allowing any process’ task port
//! to be obtained. In most situations, these access checks are desirable to
//! prevent security and privacy breaches.
//!
//! When running as setuid root, this function wraps `task_for_pid()`,
//! reimplementing those access checks. A process whose effective user ID is 0
//! and whose real user ID is nonzero is understood to be running setuid root.
//! In this case, the requested task’s real, effective, and saved set-user IDs
//! must all equal the running process’ real user ID, the requested task must
//! not have changed privileges, and the requested task’s set of all group IDs
//! (including its real, effective, and saved set-group IDs and supplementary
//! group list) must be a subset of the running process’ set of all group IDs.
//! These access checks mimic those that the kernel performs.
//!
//! When not running as setuid root, `task_for_pid()` is called directly,
//! without imposing any additional checks beyond what the kernel does.
//!
//! \param[in] pid The process ID of the task whose task port is desired.
//!
//! \return A send right to the task port if it could be obtained, or
//!     `TASK_NULL` otherwise, with an error message logged. If a send right is
//!     returned, the caller takes ownership of it.
task_t TaskForPID(pid_t pid);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_TASK_FOR_PID_H_
