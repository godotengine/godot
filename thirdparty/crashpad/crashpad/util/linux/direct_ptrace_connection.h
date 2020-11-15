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

#ifndef CRASHPAD_UTIL_LINUX_DIRECT_PTRACE_CONNECTION_H_
#define CRASHPAD_UTIL_LINUX_DIRECT_PTRACE_CONNECTION_H_

#include <sys/types.h>

#include <memory>
#include <vector>

#include "base/macros.h"
#include "util/linux/ptrace_connection.h"
#include "util/linux/ptracer.h"
#include "util/linux/scoped_ptrace_attach.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/process/process_memory_linux.h"

namespace crashpad {

//! \brief Manages a direct `ptrace` connection to a process.
//!
//! This class is used when the current process has `ptrace` capabilities for
//! the target process.
class DirectPtraceConnection : public PtraceConnection {
 public:
  DirectPtraceConnection();
  ~DirectPtraceConnection();

  //! \brief Initializes this connection for the process whose process ID is
  //!     \a pid.
  //!
  //! The main thread of the process is automatically attached by this call.
  //!
  //! \param[in] pid The process ID of the process to connect to.
  //! \return `true` on success. `false` on failure with a message logged.
  bool Initialize(pid_t pid);

  // PtraceConnection:

  pid_t GetProcessID() override;
  bool Attach(pid_t tid) override;
  bool Is64Bit() override;
  bool GetThreadInfo(pid_t tid, ThreadInfo* info) override;
  bool ReadFileContents(const base::FilePath& path,
                        std::string* contents) override;
  ProcessMemory* Memory() override;
  bool Threads(std::vector<pid_t>* threads) override;

 private:
  std::vector<std::unique_ptr<ScopedPtraceAttach>> attachments_;
  ProcessMemoryLinux memory_;
  pid_t pid_;
  Ptracer ptracer_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(DirectPtraceConnection);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_LINUX_DIRECT_PTRACE_CONNECTION_H_
