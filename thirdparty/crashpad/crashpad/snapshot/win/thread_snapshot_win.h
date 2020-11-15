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

#ifndef CRASHPAD_SNAPSHOT_WIN_THREAD_SNAPSHOT_WIN_H_
#define CRASHPAD_SNAPSHOT_WIN_THREAD_SNAPSHOT_WIN_H_

#include <stdint.h>

#include <memory>
#include <vector>

#include "base/macros.h"
#include "build/build_config.h"
#include "snapshot/cpu_context.h"
#include "snapshot/memory_snapshot.h"
#include "snapshot/thread_snapshot.h"
#include "snapshot/win/memory_snapshot_win.h"
#include "snapshot/win/process_reader_win.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

class ProcessReaderWin;

namespace internal {

//! \brief A ThreadSnapshot of a thread in a running (or crashed) process on a
//!     Windows system.
class ThreadSnapshotWin final : public ThreadSnapshot {
 public:
  ThreadSnapshotWin();
  ~ThreadSnapshotWin() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] process_reader A ProcessReaderWin for the process containing
  //!     the thread.
  //! \param[in] process_reader_thread The thread within the ProcessReaderWin
  //!     for which the snapshot should be created.
  //! \param[in,out] gather_indirectly_referenced_memory_bytes_remaining If
  //!     non-null, add extra memory regions to the snapshot pointed to by the
  //!     thread's stack. The size of the regions added is subtracted from the
  //!     count, and when it's `0`, no more regions will be added.
  //!
  //! \return `true` if the snapshot could be created, `false` otherwise with
  //!     an appropriate message logged.
  bool Initialize(
      ProcessReaderWin* process_reader,
      const ProcessReaderWin::Thread& process_reader_thread,
      uint32_t* gather_indirectly_referenced_memory_bytes_remaining);

  // ThreadSnapshot:

  const CPUContext* Context() const override;
  const MemorySnapshot* Stack() const override;
  uint64_t ThreadID() const override;
  int SuspendCount() const override;
  int Priority() const override;
  uint64_t ThreadSpecificDataAddress() const override;
  std::vector<const MemorySnapshot*> ExtraMemory() const override;

 private:
#if defined(ARCH_CPU_X86_FAMILY)
  union {
    CPUContextX86 x86;
    CPUContextX86_64 x86_64;
  } context_union_;
#endif
  CPUContext context_;
  MemorySnapshotWin stack_;
  MemorySnapshotWin teb_;
  ProcessReaderWin::Thread thread_;
  InitializationStateDcheck initialized_;
  std::vector<std::unique_ptr<MemorySnapshotWin>> pointed_to_memory_;

  DISALLOW_COPY_AND_ASSIGN(ThreadSnapshotWin);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_WIN_THREAD_SNAPSHOT_WIN_H_
