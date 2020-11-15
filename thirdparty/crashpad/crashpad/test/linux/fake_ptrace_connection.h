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

#ifndef CRASHPAD_TEST_LINUX_FAKE_PTRACE_CONNECTION_H_
#define CRASHPAD_TEST_LINUX_FAKE_PTRACE_CONNECTION_H_

#include <sys/types.h>

#include <memory>
#include <set>

#include "base/macros.h"
#include "util/linux/ptrace_connection.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/process/process_memory_linux.h"

namespace crashpad {
namespace test {

//! \brief Stands-in where real PtraceConnections aren't available.
//!
//! This class performs basic EXPECTs that it is used correctly, but does not
//! execute any real `ptrace` calls or attachments.
class FakePtraceConnection : public PtraceConnection {
 public:
  FakePtraceConnection();
  ~FakePtraceConnection();

  //! \brief Initializes this connection for the process whose process ID is
  //!     \a pid.
  //!
  //! \param[in] pid The process ID of the process to connect to.
  //! \return `true` on success. `false` on failure with a message logged.
  bool Initialize(pid_t pid);

  // PtraceConnection:

  pid_t GetProcessID() override;
  bool Attach(pid_t tid) override;

  //! \brief Returns `true` if the current process is 64-bit.
  bool Is64Bit() override;

  //! \brief Does not modify \a info.
  bool GetThreadInfo(pid_t tid, ThreadInfo* info) override;

  bool ReadFileContents(const base::FilePath& path,
                        std::string* contents) override;

  //! \brief Attempts to create a ProcessMemory when called, calling
  //!     ADD_FAILURE() and returning `nullptr` on failure.
  ProcessMemory* Memory() override;

  //! \todo Not yet implemented.
  bool Threads(std::vector<pid_t>* threads) override;

 private:
  std::set<pid_t> attachments_;
  std::unique_ptr<ProcessMemoryLinux> memory_;
  pid_t pid_;
  bool is_64_bit_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(FakePtraceConnection);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_LINUX_FAKE_PTRACE_CONNECTION_H_
