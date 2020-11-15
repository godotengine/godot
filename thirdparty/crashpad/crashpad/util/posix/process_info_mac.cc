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

#include "util/posix/process_info.h"

#include <string.h>

#include "base/logging.h"
#include "base/mac/mach_logging.h"

namespace crashpad {

ProcessInfo::ProcessInfo() : kern_proc_info_(), initialized_() {
}

ProcessInfo::~ProcessInfo() {
}

bool ProcessInfo::InitializeWithPid(pid_t pid) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  int mib[] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
  size_t len = sizeof(kern_proc_info_);
  if (sysctl(mib, arraysize(mib), &kern_proc_info_, &len, nullptr, 0) != 0) {
    PLOG(ERROR) << "sysctl for pid " << pid;
    return false;
  }

  // This sysctl does not return an error if the pid was not found. 10.9.5
  // xnu-2422.115.4/bsd/kern/kern_sysctl.c sysctl_prochandle() calls
  // xnu-2422.115.4/bsd/kern/kern_proc.c proc_iterate(), which provides no
  // indication of whether anything was done. To catch this, check that the PID
  // has changed from the 0 value it was given when initialized by the
  // constructor.
  if (kern_proc_info_.kp_proc.p_pid == 0) {
    LOG(WARNING) << "pid " << pid << " not found";
    return false;
  }

  DCHECK_EQ(kern_proc_info_.kp_proc.p_pid, pid);

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool ProcessInfo::InitializeWithTask(task_t task) {
  pid_t pid;
  kern_return_t kr = pid_for_task(task, &pid);
  if (kr != KERN_SUCCESS) {
    MACH_LOG(ERROR, kr) << "pid_for_task";
    return false;
  }

  return InitializeWithPid(pid);
}

pid_t ProcessInfo::ProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return kern_proc_info_.kp_proc.p_pid;
}

pid_t ProcessInfo::ParentProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return kern_proc_info_.kp_eproc.e_ppid;
}

uid_t ProcessInfo::RealUserID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return kern_proc_info_.kp_eproc.e_pcred.p_ruid;
}

uid_t ProcessInfo::EffectiveUserID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return kern_proc_info_.kp_eproc.e_ucred.cr_uid;
}

uid_t ProcessInfo::SavedUserID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return kern_proc_info_.kp_eproc.e_pcred.p_svuid;
}

gid_t ProcessInfo::RealGroupID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return kern_proc_info_.kp_eproc.e_pcred.p_rgid;
}

gid_t ProcessInfo::EffectiveGroupID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return kern_proc_info_.kp_eproc.e_ucred.cr_gid;
}

gid_t ProcessInfo::SavedGroupID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return kern_proc_info_.kp_eproc.e_pcred.p_svgid;
}

std::set<gid_t> ProcessInfo::SupplementaryGroups() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  const short ngroups = kern_proc_info_.kp_eproc.e_ucred.cr_ngroups;
  DCHECK_GE(ngroups, 0);
  DCHECK_LE(static_cast<size_t>(ngroups),
            arraysize(kern_proc_info_.kp_eproc.e_ucred.cr_groups));

  const gid_t* groups = kern_proc_info_.kp_eproc.e_ucred.cr_groups;
  return std::set<gid_t>(&groups[0], &groups[ngroups]);
}

std::set<gid_t> ProcessInfo::AllGroups() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  std::set<gid_t> all_groups = SupplementaryGroups();
  all_groups.insert(RealGroupID());
  all_groups.insert(EffectiveGroupID());
  all_groups.insert(SavedGroupID());
  return all_groups;
}

bool ProcessInfo::DidChangePrivileges() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return kern_proc_info_.kp_proc.p_flag & P_SUGID;
}

bool ProcessInfo::Is64Bit() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return kern_proc_info_.kp_proc.p_flag & P_LP64;
}

bool ProcessInfo::StartTime(timeval* start_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *start_time = kern_proc_info_.kp_proc.p_starttime;
  return true;
}

bool ProcessInfo::Arguments(std::vector<std::string>* argv) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  // The format of KERN_PROCARGS2 is explained in 10.9.2 adv_cmds-153/ps/print.c
  // getproclline(). It is an int (argc) followed by the executable’s string
  // area. The string area consists of NUL-terminated strings, beginning with
  // the executable path, and then starting on an aligned boundary, all of the
  // elements of argv, envp, and applev.

  // It is possible for a process to exec() in between the two sysctl() calls
  // below. If that happens, and the string area of the new program is larger
  // than that of the old one, args_size_estimate will be too small. To detect
  // this situation, the second sysctl() attempts to fetch args_size_estimate +
  // 1 bytes, expecting to only receive args_size_estimate. If it gets the extra
  // byte, it indicates that the string area has grown, and the sysctl() pair
  // will be retried a limited number of times.

  size_t args_size_estimate;
  size_t args_size;
  std::string args;
  int tries = 3;
  const pid_t pid = ProcessID();
  do {
    int mib[] = {CTL_KERN, KERN_PROCARGS2, pid};
    int rv =
        sysctl(mib, arraysize(mib), nullptr, &args_size_estimate, nullptr, 0);
    if (rv != 0) {
      PLOG(ERROR) << "sysctl (size) for pid " << pid;
      return false;
    }

    args_size = args_size_estimate + 1;
    args.resize(args_size);
    rv = sysctl(mib, arraysize(mib), &args[0], &args_size, nullptr, 0);
    if (rv != 0) {
      PLOG(ERROR) << "sysctl (data) for pid " << pid;
      return false;
    }
  } while (args_size == args_size_estimate + 1 && tries--);

  if (args_size == args_size_estimate + 1) {
    LOG(ERROR) << "unexpected args_size";
    return false;
  }

  // KERN_PROCARGS2 needs to at least contain argc.
  if (args_size < sizeof(int)) {
    LOG(ERROR) << "tiny args_size";
    return false;
  }
  args.resize(args_size);

  // Get argc.
  int argc;
  memcpy(&argc, &args[0], sizeof(argc));

  // Find the end of the executable path.
  size_t start_pos = sizeof(argc);
  size_t nul_pos = args.find('\0', start_pos);
  if (nul_pos == std::string::npos) {
    LOG(ERROR) << "unterminated executable path";
    return false;
  }

  // Find the beginning of the string area.
  start_pos = args.find_first_not_of('\0', nul_pos);
  if (start_pos == std::string::npos) {
    LOG(ERROR) << "no string area";
    return false;
  }

  std::vector<std::string> local_argv;
  while (argc-- && nul_pos != std::string::npos) {
    nul_pos = args.find('\0', start_pos);
    local_argv.push_back(args.substr(start_pos, nul_pos - start_pos));
    start_pos = nul_pos + 1;
  }

  if (argc >= 0) {
    // Not every argument was recovered.
    LOG(ERROR) << "did not recover all arguments";
    return false;
  }

  argv->swap(local_argv);
  return true;
}

}  // namespace crashpad
