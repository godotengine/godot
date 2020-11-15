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

#ifndef CRASHPAD_SNAPSHOT_LINUX_EXCEPTION_SNAPSHOT_LINUX_H_
#define CRASHPAD_SNAPSHOT_LINUX_EXCEPTION_SNAPSHOT_LINUX_H_

#include <stdint.h>
#include <sys/types.h>

#include <vector>

#include "base/macros.h"
#include "build/build_config.h"
#include "snapshot/cpu_context.h"
#include "snapshot/exception_snapshot.h"
#include "snapshot/linux/process_reader_linux.h"
#include "snapshot/memory_snapshot.h"
#include "util/linux/address_types.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {
namespace internal {

//! \brief An ExceptionSnapshot of an signal received by a running (or crashed)
//!     process on a Linux system.
class ExceptionSnapshotLinux final : public ExceptionSnapshot {
 public:
  ExceptionSnapshotLinux();
  ~ExceptionSnapshotLinux() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] process_reader A ProcessReaderLinux for the process that
  //! received
  //!     the signal.
  //! \param[in] siginfo_address The address in the target process' address
  //!     space of the siginfo_t passed to the signal handler.
  //! \param[in] context_address The address in the target process' address
  //!     space of the ucontext_t passed to the signal handler.
  //! \param[in] thread_id The thread ID of the thread that received the signal.
  //!
  //! \return `true` if the snapshot could be created, `false` otherwise with
  //!     an appropriate message logged.
  bool Initialize(ProcessReaderLinux* process_reader,
                  LinuxVMAddress siginfo_address,
                  LinuxVMAddress context_address,
                  pid_t thread_id);

  // ExceptionSnapshot:

  const CPUContext* Context() const override;
  uint64_t ThreadID() const override;
  uint32_t Exception() const override;
  uint32_t ExceptionInfo() const override;
  uint64_t ExceptionAddress() const override;
  const std::vector<uint64_t>& Codes() const override;
  virtual std::vector<const MemorySnapshot*> ExtraMemory() const override;

 private:
  template <typename Traits>
  bool ReadSiginfo(ProcessReaderLinux* reader, LinuxVMAddress siginfo_address);

  template <typename Traits>
  bool ReadContext(ProcessReaderLinux* reader, LinuxVMAddress context_address);

  union {
#if defined(ARCH_CPU_X86_FAMILY)
    CPUContextX86 x86;
    CPUContextX86_64 x86_64;
#elif defined(ARCH_CPU_ARM_FAMILY)
    CPUContextARM arm;
    CPUContextARM64 arm64;
#elif defined(ARCH_CPU_MIPS_FAMILY)
    CPUContextMIPS mipsel;
    CPUContextMIPS64 mips64;
#endif
  } context_union_;
  CPUContext context_;
  std::vector<uint64_t> codes_;
  uint64_t thread_id_;
  uint64_t exception_address_;
  uint32_t signal_number_;
  uint32_t signal_code_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ExceptionSnapshotLinux);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_LINUX_EXCEPTION_SNAPSHOT_LINUX_H_
