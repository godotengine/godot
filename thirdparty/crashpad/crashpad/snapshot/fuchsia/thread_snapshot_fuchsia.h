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

#ifndef CRASHPAD_SNAPSHOT_FUCHSIA_THREAD_SNAPSHOT_FUCHSIA_H_
#define CRASHPAD_SNAPSHOT_FUCHSIA_THREAD_SNAPSHOT_FUCHSIA_H_

#include <stdint.h>
#include <zircon/types.h>

#include "base/macros.h"
#include "build/build_config.h"
#include "snapshot/cpu_context.h"
#include "snapshot/fuchsia/process_reader_fuchsia.h"
#include "snapshot/memory_snapshot.h"
#include "snapshot/memory_snapshot_generic.h"
#include "snapshot/thread_snapshot.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {
namespace internal {

//! \brief A ThreadSnapshot of a thread on a Fuchsia system.
class ThreadSnapshotFuchsia final : public ThreadSnapshot {
 public:
  ThreadSnapshotFuchsia();
  ~ThreadSnapshotFuchsia() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] process_reader A ProcessReaderFuchsia for the process
  //!     containing the thread.
  //! \param[in] thread The thread within the ProcessReaderFuchsia for
  //!     which the snapshot should be created.
  //!
  //! \return `true` if the snapshot could be created, `false` otherwise with
  //!     a message logged.
  bool Initialize(ProcessReaderFuchsia* process_reader,
                  const ProcessReaderFuchsia::Thread& thread);

  // ThreadSnapshot:

  const CPUContext* Context() const override;
  const MemorySnapshot* Stack() const override;
  uint64_t ThreadID() const override;
  int SuspendCount() const override;
  int Priority() const override;
  uint64_t ThreadSpecificDataAddress() const override;
  std::vector<const MemorySnapshot*> ExtraMemory() const override;

 private:
#if defined(ARCH_CPU_X86_64)
  CPUContextX86_64 context_arch_;
#elif defined(ARCH_CPU_ARM64)
  CPUContextARM64 context_arch_;
#else
#error Port.
#endif
  CPUContext context_;
  MemorySnapshotGeneric<ProcessReaderFuchsia> stack_;
  zx_koid_t thread_id_;
  zx_vaddr_t thread_specific_data_address_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ThreadSnapshotFuchsia);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_FUCHSIA_THREAD_SNAPSHOT_FUCHSIA_H_
