// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#include "util/win/scoped_process_suspend.h"

#include <stddef.h>
#include <winternl.h>

#include "util/win/nt_internals.h"
#include "util/win/ntstatus_logging.h"

namespace crashpad {

ScopedProcessSuspend::ScopedProcessSuspend(HANDLE process) {
  NTSTATUS status = NtSuspendProcess(process);
  if (NT_SUCCESS(status)) {
    process_ = process;
  } else {
    process_ = nullptr;
    NTSTATUS_LOG(ERROR, status) << "NtSuspendProcess";
  }
}

ScopedProcessSuspend::~ScopedProcessSuspend() {
  if (process_) {
    NTSTATUS status = NtResumeProcess(process_);
    if (!NT_SUCCESS(status) &&
        (!tolerate_termination_ || status != STATUS_PROCESS_IS_TERMINATING)) {
      NTSTATUS_LOG(ERROR, status) << "NtResumeProcess";
    }
  }
}

void ScopedProcessSuspend::TolerateTermination() {
  tolerate_termination_ = true;
}

}  // namespace crashpad
