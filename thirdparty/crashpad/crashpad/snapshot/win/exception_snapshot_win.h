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

#ifndef CRASHPAD_SNAPSHOT_WIN_EXCEPTION_SNAPSHOT_WIN_H_
#define CRASHPAD_SNAPSHOT_WIN_EXCEPTION_SNAPSHOT_WIN_H_

#include <windows.h>
#include <stdint.h>

#include <memory>
#include <vector>

#include "base/macros.h"
#include "build/build_config.h"
#include "snapshot/cpu_context.h"
#include "snapshot/exception_snapshot.h"
#include "snapshot/win/thread_snapshot_win.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/win/address_types.h"
#include "util/win/process_structs.h"

namespace crashpad {

class ProcessReaderWin;

namespace internal {

class MemorySnapshotWin;

#if defined(ARCH_CPU_X86_FAMILY)
union CPUContextUnion {
  CPUContextX86 x86;
  CPUContextX86_64 x86_64;
};
#endif

class ExceptionSnapshotWin final : public ExceptionSnapshot {
 public:
  ExceptionSnapshotWin();
  ~ExceptionSnapshotWin() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] process_reader A ProcessReaderWin for the process that
  //!     sustained the exception.
  //! \param[in] thread_id The thread ID in which the exception occurred.
  //! \param[in] exception_pointers The address of an `EXCEPTION_POINTERS`
  //!     record in the target process, passed through from the exception
  //!     handler.
  //!
  //! \note If the exception was triggered by
  //!     CrashpadClient::DumpAndCrashTargetProcess(), this has the side-effect
  //!     of correcting the thread suspend counts for \a process_reader.
  //!
  //! \return `true` if the snapshot could be created, `false` otherwise with
  //!     an appropriate message logged.
  bool Initialize(ProcessReaderWin* process_reader,
                  DWORD thread_id,
                  WinVMAddress exception_pointers);

  // ExceptionSnapshot:

  const CPUContext* Context() const override;
  uint64_t ThreadID() const override;
  uint32_t Exception() const override;
  uint32_t ExceptionInfo() const override;
  uint64_t ExceptionAddress() const override;
  const std::vector<uint64_t>& Codes() const override;
  std::vector<const MemorySnapshot*> ExtraMemory() const override;

 private:
  template <class ExceptionRecordType,
            class ExceptionPointersType,
            class ContextType>
  bool InitializeFromExceptionPointers(
      ProcessReaderWin* process_reader,
      WinVMAddress exception_pointers_address,
      DWORD exception_thread_id,
      void (*native_to_cpu_context)(const ContextType& context_record,
                                    CPUContext* context,
                                    CPUContextUnion* context_union));

#if defined(ARCH_CPU_X86_FAMILY)
  CPUContextUnion context_union_;
#endif
  CPUContext context_;
  std::vector<uint64_t> codes_;
  std::vector<std::unique_ptr<internal::MemorySnapshotWin>> extra_memory_;
  uint64_t thread_id_;
  uint64_t exception_address_;
  uint32_t exception_flags_;
  DWORD exception_code_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ExceptionSnapshotWin);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_WIN_EXCEPTION_SNAPSHOT_WIN_H_
