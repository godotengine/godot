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

#include "snapshot/sanitized/thread_snapshot_sanitized.h"

#include "snapshot/cpu_context.h"

namespace crashpad {
namespace internal {

ThreadSnapshotSanitized::ThreadSnapshotSanitized(const ThreadSnapshot* snapshot,
                                                 RangeSet* ranges)
    : ThreadSnapshot(),
      snapshot_(snapshot),
      stack_(snapshot_->Stack(), ranges, snapshot_->Context()->Is64Bit()) {}

ThreadSnapshotSanitized::~ThreadSnapshotSanitized() = default;

const CPUContext* ThreadSnapshotSanitized::Context() const {
  return snapshot_->Context();
}

const MemorySnapshot* ThreadSnapshotSanitized::Stack() const {
  return &stack_;
}

uint64_t ThreadSnapshotSanitized::ThreadID() const {
  return snapshot_->ThreadID();
}

int ThreadSnapshotSanitized::SuspendCount() const {
  return snapshot_->SuspendCount();
}

int ThreadSnapshotSanitized::Priority() const {
  return snapshot_->Priority();
}

uint64_t ThreadSnapshotSanitized::ThreadSpecificDataAddress() const {
  return snapshot_->ThreadSpecificDataAddress();
}

std::vector<const MemorySnapshot*> ThreadSnapshotSanitized::ExtraMemory()
    const {
  // TODO(jperaza): If/when ExtraMemory() is used, decide whether and how it
  // should be sanitized.
  DCHECK(snapshot_->ExtraMemory().empty());
  return std::vector<const MemorySnapshot*>();
}

}  // namespace internal
}  // namespace crashpad
