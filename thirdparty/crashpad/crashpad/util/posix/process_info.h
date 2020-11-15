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

#ifndef CRASHPAD_UTIL_POSIX_PROCESS_INFO_H_
#define CRASHPAD_UTIL_POSIX_PROCESS_INFO_H_

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <set>
#include <string>
#include <vector>

#include "base/macros.h"
#include "build/build_config.h"
#include "util/misc/initialization_state.h"
#include "util/misc/initialization_state_dcheck.h"

#if defined(OS_MACOSX)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

#if defined(OS_LINUX) || defined(OS_ANDROID)
#include "util/linux/ptrace_connection.h"
#endif

namespace crashpad {

class ProcessInfo {
 public:
  ProcessInfo();
  ~ProcessInfo();

#if defined(OS_LINUX) || defined(OS_ANDROID) || DOXYGEN
  //! \brief Initializes this object with information about the process whose ID
  //!     is \a pid using a PtraceConnection \a connection.
  //!
  //! This method must be called successfully prior to calling any other method
  //! in this class. This method may only be called once.
  //!
  //! It is unspecified whether the information that an object of this class
  //! returns is loaded at the time Initialize() is called or subsequently, and
  //! whether this information is cached in the object or not.
  //!
  //! \param[in] connection A connection to the remote process.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  bool InitializeWithPtrace(PtraceConnection* connection);
#endif  // OS_LINUX || OS_ANDROID || DOXYGEN

#if defined(OS_MACOSX) || DOXYGEN
  //! \brief Initializes this object with information about the process whose ID
  //!     is \a pid.
  //!
  //! This method must be called successfully prior to calling any other method
  //! in this class. This method may only be called once.
  //!
  //! It is unspecified whether the information that an object of this class
  //! returns is loaded at the time Initialize() is called or subsequently, and
  //! whether this information is cached in the object or not.
  //!
  //! \param[in] pid The process ID to obtain information for.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  bool InitializeWithPid(pid_t pid);

  //! \brief Initializes this object with information about a process based on
  //!     its Mach task.
  //!
  //! This method serves as a stand-in for InitializeWithPid() and may be called
  //! in its place with the same restrictions and considerations.
  //!
  //! \param[in] task The Mach task to obtain information for.
  //!
  //! \return `true` on success, `false` on failure with an message logged.
  bool InitializeWithTask(task_t task);
#endif

  //! \return The target task’s process ID.
  pid_t ProcessID() const;

  //! \return The target task’s parent process ID.
  pid_t ParentProcessID() const;

  //! \return The target process’ real user ID as would be returned to it by
  //!     `getuid()`.
  uid_t RealUserID() const;

  //! \return The target process’ effective user ID as would be returned to it
  //!     by `geteuid()`.
  uid_t EffectiveUserID() const;

  //! \return The target process’ saved set-user ID.
  uid_t SavedUserID() const;

  //! \return the target process’ real group ID as would be returned to it by
  //!     `getgid()`.
  gid_t RealGroupID() const;

  //! \return the target process’ effective group ID as would be returned to it
  //!     by `getegid()`.
  gid_t EffectiveGroupID() const;

  //! \return The target process’ saved set-group ID.
  gid_t SavedGroupID() const;

  //! \return the target process’ supplementary group list as would be returned
  //!     to it by `getgroups()`.
  std::set<gid_t> SupplementaryGroups() const;

  //! \return All groups that the target process claims membership in, including
  //!     RealGroupID(), EffectiveGroupID(), SavedGroupID(), and
  //!     SupplementaryGroups().
  std::set<gid_t> AllGroups() const;

  //! \brief Determines whether the target process has changed privileges.
  //!
  //! A process is considered to have changed privileges if it has changed its
  //! real, effective, or saved set-user or group IDs with the `setuid()`,
  //! `seteuid()`, `setreuid()`, `setgid()`, `setegid()`, or `setregid()` system
  //! calls since its most recent `execve()`, or if its privileges changed at
  //! `execve()` as a result of executing a setuid or setgid executable.
  bool DidChangePrivileges() const;

  //! \brief Determines the target process’ bitness.
  //!
  //! \return `true` if the target task is a 64-bit process.
  bool Is64Bit() const;

  //! \brief Determines the target process’ start time.
  //!
  //! \param[out] start_time The time that the process started.
  //!
  //! \return `true` on success, with \a start_time set. Otherwise, `false` with
  //!     a message logged.
  bool StartTime(timeval* start_time) const;

  //! \brief Obtains the arguments used to launch a process.
  //!
  //! Whether it is possible to obtain this information for a process with
  //! different privileges than the running program is system-dependent.
  //!
  //! \param[out] argv The process’ arguments as passed to its `main()` function
  //!     as the \a argv parameter, possibly modified by the process.
  //!
  //! \return `true` on success, with \a argv populated appropriately.
  //!     Otherwise, `false` with a message logged.
  //!
  //! \note This function may spuriously return `false` when used to examine a
  //!     process that it is calling `exec()`. If examining such a process, call
  //!     this function in a retry loop with a small (100ns) delay to avoid an
  //!     erroneous assumption that \a pid is not running.
  bool Arguments(std::vector<std::string>* argv) const;

 private:
#if defined(OS_MACOSX)
  kinfo_proc kern_proc_info_;
#elif defined(OS_LINUX) || defined(OS_ANDROID)
  // Some members are marked mutable so that they can be lazily initialized by
  // const methods. These are always InitializationState-protected so that
  // multiple successive calls will always produce the same return value and out
  // parameters. This is necessary for intergration with the Snapshot interface.
  // See https://crashpad.chromium.org/bug/9.
  PtraceConnection* connection_;
  std::set<gid_t> supplementary_groups_;
  mutable timeval start_time_;
  pid_t pid_;
  pid_t ppid_;
  uid_t uid_;
  uid_t euid_;
  uid_t suid_;
  gid_t gid_;
  gid_t egid_;
  gid_t sgid_;
  bool is_64_bit_;
  mutable InitializationState start_time_initialized_;
#endif
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcessInfo);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_POSIX_PROCESS_INFO_H_
