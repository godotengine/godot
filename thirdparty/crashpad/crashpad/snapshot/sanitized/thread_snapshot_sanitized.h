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

#ifndef CRASHPAD_SNAPSHOT_SANITIZED_THREAD_SNAPSHOT_SANITIZED_H_
#define CRASHPAD_SNAPSHOT_SANITIZED_THREAD_SNAPSHOT_SANITIZED_H_

#include "snapshot/thread_snapshot.h"

#include "snapshot/sanitized/memory_snapshot_sanitized.h"
#include "util/misc/range_set.h"

namespace crashpad {
namespace internal {

//! \brief A ThreadSnapshot which wraps and filters sensitive information from
//!     another ThreadSnapshot.
class ThreadSnapshotSanitized final : public ThreadSnapshot {
 public:
  //! \brief Constructs this object.
  //!
  //! \param[in] snapshot The ThreadSnapshot to sanitize.
  //! \param[in] ranges A set of address ranges with which to sanitize this
  //!     thread's stacks. \see internal::MemorySnapshotSanitized.
  ThreadSnapshotSanitized(const ThreadSnapshot* snapshot, RangeSet* ranges);

  ~ThreadSnapshotSanitized() override;

  // ThreadSnapshot:

  const CPUContext* Context() const override;
  const MemorySnapshot* Stack() const override;
  uint64_t ThreadID() const override;
  int SuspendCount() const override;
  int Priority() const override;
  uint64_t ThreadSpecificDataAddress() const override;
  std::vector<const MemorySnapshot*> ExtraMemory() const override;

 private:
  const ThreadSnapshot* snapshot_;
  MemorySnapshotSanitized stack_;

  DISALLOW_COPY_AND_ASSIGN(ThreadSnapshotSanitized);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_SANITIZED_THREAD_SNAPSHOT_SANITIZED_H_
