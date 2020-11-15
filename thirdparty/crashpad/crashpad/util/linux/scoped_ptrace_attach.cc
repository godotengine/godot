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

#include "util/linux/scoped_ptrace_attach.h"

#include <sys/ptrace.h>
#include <sys/wait.h>

#include "base/logging.h"
#include "base/posix/eintr_wrapper.h"

namespace crashpad {

ScopedPtraceAttach::ScopedPtraceAttach()
    : pid_(-1) {}

ScopedPtraceAttach::~ScopedPtraceAttach() {
  Reset();
}

bool ScopedPtraceAttach::Reset() {
  if (pid_ >= 0 && ptrace(PTRACE_DETACH, pid_, nullptr, nullptr) != 0) {
    PLOG(ERROR) << "ptrace";
    return false;
  }
  pid_ = -1;
  return true;
}

bool ScopedPtraceAttach::ResetAttach(pid_t pid) {
  Reset();

  if (ptrace(PTRACE_ATTACH, pid, nullptr, nullptr) != 0) {
    PLOG(ERROR) << "ptrace";
    return false;
  }
  pid_ = pid;

  int status;
  if (HANDLE_EINTR(waitpid(pid_, &status, __WALL)) < 0) {
    PLOG(ERROR) << "waitpid";
    return false;
  }
  if (!WIFSTOPPED(status)) {
    LOG(ERROR) << "process not stopped";
    return false;
  }
  return true;
}

}  // namespace crashpad
