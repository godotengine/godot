// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_SNAPSHOT_FUCHSIA_EXCEPTION_SNAPSHOT_FUCHSIA_H_
#define CRASHPAD_SNAPSHOT_FUCHSIA_EXCEPTION_SNAPSHOT_FUCHSIA_H_

#include <stdint.h>
#include <zircon/syscalls/exception.h>
#include <zircon/types.h>

#include "build/build_config.h"
#include "snapshot/cpu_context.h"
#include "snapshot/exception_snapshot.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

class ProcessReaderFuchsia;

namespace internal {

//! \brief An ExceptionSnapshot of an exception sustained by a process on a
//!     Fuchsia system.
class ExceptionSnapshotFuchsia final : public ExceptionSnapshot {
 public:
  ExceptionSnapshotFuchsia();
  ~ExceptionSnapshotFuchsia() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] process_reader A ProcessReaderFuchsia for the process that
  //!     sustained the exception.
  //! \param[in] thread_id The koid of the thread that sustained the exception.
  //! \param[in] exception_report The `zx_exception_report_t` retrieved from the
  //!     thread in the exception state, corresponding to \a thread_id.
  void Initialize(ProcessReaderFuchsia* process_reader,
                  zx_koid_t thread_id,
                  const zx_exception_report_t& exception_report);

  // ExceptionSnapshot:
  const CPUContext* Context() const override;
  uint64_t ThreadID() const override;
  uint32_t Exception() const override;
  uint32_t ExceptionInfo() const override;
  uint64_t ExceptionAddress() const override;
  const std::vector<uint64_t>& Codes() const override;
  std::vector<const MemorySnapshot*> ExtraMemory() const override;

 private:
#if defined(ARCH_CPU_X86_64)
  CPUContextX86_64 context_arch_;
#elif defined(ARCH_CPU_ARM64)
  CPUContextARM64 context_arch_;
#endif
  CPUContext context_;
  std::vector<uint64_t> codes_;
  zx_koid_t thread_id_;
  zx_vaddr_t exception_address_;
  uint32_t exception_;
  uint32_t exception_info_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ExceptionSnapshotFuchsia);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_FUCHSIA_EXCEPTION_SNAPSHOT_FUCHSIA_H_
