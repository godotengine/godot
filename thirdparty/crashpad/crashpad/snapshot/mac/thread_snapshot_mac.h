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

#ifndef CRASHPAD_SNAPSHOT_MAC_THREAD_SNAPSHOT_MAC_H_
#define CRASHPAD_SNAPSHOT_MAC_THREAD_SNAPSHOT_MAC_H_

#include <mach/mach.h>
#include <stdint.h>

#include "base/macros.h"
#include "build/build_config.h"
#include "snapshot/cpu_context.h"
#include "snapshot/mac/process_reader_mac.h"
#include "snapshot/memory_snapshot.h"
#include "snapshot/memory_snapshot_generic.h"
#include "snapshot/thread_snapshot.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

class ProcessReaderMac;

namespace internal {

//! \brief A ThreadSnapshot of a thread in a running (or crashed) process on a
//!     macOS system.
class ThreadSnapshotMac final : public ThreadSnapshot {
 public:
  ThreadSnapshotMac();
  ~ThreadSnapshotMac() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] process_reader A ProcessReaderMac for the task containing the
  //!     thread.
  //! \param[in] process_reader_thread The thread within the ProcessReaderMac
  //!     for which the snapshot should be created.
  //!
  //! \return `true` if the snapshot could be created, `false` otherwise with
  //!     an appropriate message logged.
  bool Initialize(ProcessReaderMac* process_reader,
                  const ProcessReaderMac::Thread& process_reader_thread);

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
  MemorySnapshotGeneric<ProcessReaderMac> stack_;
  uint64_t thread_id_;
  uint64_t thread_specific_data_address_;
  thread_t thread_;
  int suspend_count_;
  int priority_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ThreadSnapshotMac);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MAC_THREAD_SNAPSHOT_MAC_H_
