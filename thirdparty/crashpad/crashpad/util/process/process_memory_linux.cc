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

#include "util/process/process_memory_linux.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <limits>

#include "base/logging.h"
#include "base/posix/eintr_wrapper.h"

namespace crashpad {

ProcessMemoryLinux::ProcessMemoryLinux()
    : ProcessMemory(), mem_fd_(), pid_(-1), initialized_() {}

ProcessMemoryLinux::~ProcessMemoryLinux() {}

bool ProcessMemoryLinux::Initialize(pid_t pid) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  pid_ = pid;
  char path[32];
  snprintf(path, sizeof(path), "/proc/%d/mem", pid_);
  mem_fd_.reset(HANDLE_EINTR(open(path, O_RDONLY | O_NOCTTY | O_CLOEXEC)));
  if (!mem_fd_.is_valid()) {
    PLOG(ERROR) << "open";
    return false;
  }
  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

ssize_t ProcessMemoryLinux::ReadUpTo(VMAddress address,
                                     size_t size,
                                     void* buffer) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  DCHECK(mem_fd_.is_valid());
  DCHECK_LE(size, size_t{std::numeric_limits<ssize_t>::max()});

  ssize_t bytes_read =
      HANDLE_EINTR(pread64(mem_fd_.get(), buffer, size, address));
  if (bytes_read < 0) {
    PLOG(ERROR) << "pread64";
  }
  return bytes_read;
}

}  // namespace crashpad
