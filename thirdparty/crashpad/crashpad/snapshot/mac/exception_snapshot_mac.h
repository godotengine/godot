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

#ifndef CRASHPAD_SNAPSHOT_MAC_EXCEPTION_SNAPSHOT_MAC_H_
#define CRASHPAD_SNAPSHOT_MAC_EXCEPTION_SNAPSHOT_MAC_H_

#include <mach/mach.h>
#include <stdint.h>

#include <vector>

#include "base/macros.h"
#include "build/build_config.h"
#include "snapshot/cpu_context.h"
#include "snapshot/exception_snapshot.h"
#include "util/mach/mach_extensions.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

class ProcessReaderMac;

namespace internal {

//! \brief An ExceptionSnapshot of an exception sustained by a running (or
//!     crashed) process on a macOS system.
class ExceptionSnapshotMac final : public ExceptionSnapshot {
 public:
  ExceptionSnapshotMac();
  ~ExceptionSnapshotMac() override;

  //! \brief Initializes the object.
  //!
  //! Other than \a process_reader, the parameters may be passed directly
  //! through from a Mach exception handler.
  //!
  //! \param[in] process_reader A ProcessReaderMac for the task that sustained
  //!     the exception.
  //! \param[in] behavior
  //! \param[in] exception_thread
  //! \param[in] exception
  //! \param[in] code
  //! \param[in] code_count
  //! \param[in,out] flavor
  //! \param[in] state
  //! \param[in] state_count
  //!
  //! \return `true` if the snapshot could be created, `false` otherwise with
  //!     an appropriate message logged.
  bool Initialize(ProcessReaderMac* process_reader,
                  exception_behavior_t behavior,
                  thread_t exception_thread,
                  exception_type_t exception,
                  const mach_exception_data_type_t* code,
                  mach_msg_type_number_t code_count,
                  thread_state_flavor_t flavor,
                  ConstThreadState state,
                  mach_msg_type_number_t state_count);

  // ExceptionSnapshot:

  const CPUContext* Context() const override;
  uint64_t ThreadID() const override;
  uint32_t Exception() const override;
  uint32_t ExceptionInfo() const override;
  uint64_t ExceptionAddress() const override;
  const std::vector<uint64_t>& Codes() const override;
  virtual std::vector<const MemorySnapshot*> ExtraMemory() const override;

 private:
#if defined(ARCH_CPU_X86_FAMILY)
  union {
    CPUContextX86 x86;
    CPUContextX86_64 x86_64;
  } context_union_;
#endif
  CPUContext context_;
  std::vector<uint64_t> codes_;
  uint64_t thread_id_;
  uint64_t exception_address_;
  exception_type_t exception_;
  uint32_t exception_code_0_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ExceptionSnapshotMac);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MAC_EXCEPTION_SNAPSHOT_MAC_H_
