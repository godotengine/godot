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

#include "util/mach/task_for_pid.h"

#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <iterator>
#include <set>

#include "base/mac/mach_logging.h"
#include "base/mac/scoped_mach_port.h"
#include "util/posix/process_info.h"

namespace crashpad {

namespace {

//! \brief Determines whether the groups that \a process_reader belongs to are
//!     a subset of the groups that the current process belongs to.
//!
//! This function is similar to 10.9.5
//! `xnu-2422.115.4/bsd/kern/kern_credential.c` `kauth_cred_gid_subset()`.
bool TaskForPIDGroupCheck(const ProcessInfo& process_info) {
  std::set<gid_t> groups = process_info.AllGroups();

  ProcessInfo process_info_self;
  if (!process_info_self.InitializeWithPid(getpid())) {
    return false;
  }

  std::set<gid_t> groups_self = process_info_self.AllGroups();

  // difference will only contain elements of groups not present in groups_self.
  // It will not contain elements of groups_self not present in groups. (That
  // would be std::set_symmetric_difference.)
  std::set<gid_t> difference;
  std::set_difference(groups.begin(),
                      groups.end(),
                      groups_self.begin(),
                      groups_self.end(),
                      std::inserter(difference, difference.begin()));
  if (!difference.empty()) {
    LOG(ERROR) << "permission denied (gid)";
    return false;
  }

  return true;
}

//! \brief Determines whether the current process should have permission to
//!     access the specified task port.
//!
//! This function is similar to 10.9.5
//! `xnu-2422.115.4/bsd/vm/vm_unix.c` `task_for_pid_posix_check()`.
//!
//! This function accepts a `task_t` argument instead of a `pid_t` argument,
//! implying that the task send right must be retrieved before it can be
//! checked. This is done because a `pid_t` argument may refer to a different
//! task in between the time that access is checked and its corresponding
//! `task_t` is obtained by `task_for_pid()`. When `task_for_pid()` is called
//! first, any operations requiring the process ID will call `pid_for_task()`
//! and be guaranteed to use the process ID corresponding to the correct task,
//! or to fail if that task is no longer running. If the task dies and the PID
//! is recycled, it is still possible to look up the wrong PID, but falsely
//! granting task access based on the new process’ characteristics is harmless
//! because the task will be a dead name at that point.
bool TaskForPIDCheck(task_t task) {
  // If the effective user ID is not 0, then this code is not running as root at
  // all, and the kernel’s own checks are sufficient to determine access. The
  // point of this function is to simulate the kernel’s own checks when the
  // effective user ID is 0 but the real user ID is anything else.
  if (geteuid() != 0) {
    return true;
  }

  // If the real user ID is 0, then this code is not running setuid root, it’s
  // genuinely running as root, and it should be allowed maximum access.
  uid_t uid = getuid();
  if (uid == 0) {
    return true;
  }

  // task_for_pid_posix_check() would permit access to the running process’ own
  // task here, and would then check the kern.tfp.policy sysctl. If set to
  // KERN_TFP_POLICY_DENY, it would deny access.
  //
  // This behavior is not duplicated here because the point of this function is
  // to permit task_for_pid() access for setuid root programs. It is assumed
  // that a setuid root program ought to be able to overcome any policy set in
  // kern.tfp.policy.
  //
  // Access to the running process’ own task is not granted outright and is
  // instead subjected to the same user/group ID checks as any other process.
  // This has the effect of denying access to the running process’ own task when
  // it is setuid root. This is intentional, because it prevents the same sort
  // of cross-privilege disclosure discussed below at the DidChangePriveleges()
  // check. The running process can still access its own task port via
  // mach_task_self(), but a non-root user cannot coerce a setuid root tool to
  // operate on itself by specifying its own process ID to this TaskForPID()
  // interface.

  ProcessInfo process_info;
  if (!process_info.InitializeWithTask(task)) {
    return false;
  }

  // The target process’ real user ID, effective user ID, and saved set-user ID
  // must match this process’ own real user ID. task_for_pid_posix_check()
  // checks against the current process’ effective user ID, but for the purposes
  // of this function, when running setuid root, the real user ID is the correct
  // choice.
  if (process_info.RealUserID() != uid ||
      process_info.EffectiveUserID() != uid ||
      process_info.SavedUserID() != uid) {
    LOG(ERROR) << "permission denied (uid)";
    return false;
  }

  // The target process must not have changed privileges. The rationale for this
  // check is explained in 10.9.5 xnu-2422.115.4/bsd/kern/kern_prot.c
  // issetugid(): processes that have changed privileges may have loaded data
  // using different credentials than they are currently operating with, and
  // allowing other processes access to this data based solely on a check of the
  // current credentials could violate confidentiality.
  if (process_info.DidChangePrivileges()) {
    LOG(ERROR) << "permission denied (P_SUGID)";
    return false;
  }

  return TaskForPIDGroupCheck(process_info);
}

}  // namespace

task_t TaskForPID(pid_t pid) {
  task_t task;
  kern_return_t kr = task_for_pid(mach_task_self(), pid, &task);
  if (kr != KERN_SUCCESS) {
    MACH_LOG(ERROR, kr) << "task_for_pid";
    return TASK_NULL;
  }

  base::mac::ScopedMachSendRight task_owner(task);

  if (!TaskForPIDCheck(task)) {
    return TASK_NULL;
  }

  return task_owner.release();
}

}  // namespace crashpad
