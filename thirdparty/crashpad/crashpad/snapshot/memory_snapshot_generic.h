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

#ifndef CRASHPAD_SNAPSHOT_MEMORY_SNAPSHOT_GENERIC_H_
#define CRASHPAD_SNAPSHOT_MEMORY_SNAPSHOT_GENERIC_H_

#include <stdint.h>
#include <sys/types.h>

#include "base/macros.h"
#include "snapshot/memory_snapshot.h"
#include "util/misc/address_types.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/process/process_memory.h"

namespace crashpad {
namespace internal {

//! \brief A MemorySnapshot of a memory region in a process on the running
//!     system. Used on Mac, Linux, Android, and Fuchsia, templated on the
//!     platform-specific ProcessReader type.
template <class ProcessReaderType>
class MemorySnapshotGeneric final : public MemorySnapshot {
 public:
  MemorySnapshotGeneric() = default;
  ~MemorySnapshotGeneric() = default;

  //! \brief Initializes the object.
  //!
  //! Memory is read lazily. No attempt is made to read the memory snapshot data
  //! until Read() is called, and the memory snapshot data is discared when
  //! Read() returns.
  //!
  //! \param[in] process_reader A reader for the process being snapshotted.
  //! \param[in] address The base address of the memory region to snapshot, in
  //!     the snapshot process’ address space.
  //! \param[in] size The size of the memory region to snapshot.
  void Initialize(ProcessReaderType* process_reader,
                  VMAddress address,
                  VMSize size) {
    INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
    process_reader_ = process_reader;
    address_ = address;
    size_ = size;
    INITIALIZATION_STATE_SET_VALID(initialized_);
  }

  // MemorySnapshot:

  uint64_t Address() const override {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return address_;
  }

  size_t Size() const override {
    INITIALIZATION_STATE_DCHECK_VALID(initialized_);
    return size_;
  }

  bool Read(Delegate* delegate) const override {
    INITIALIZATION_STATE_DCHECK_VALID(initialized_);

    if (size_ == 0) {
      return delegate->MemorySnapshotDelegateRead(nullptr, size_);
    }

    std::unique_ptr<uint8_t[]> buffer(new uint8_t[size_]);
    if (!process_reader_->Memory()->Read(address_, size_, buffer.get())) {
      return false;
    }
    return delegate->MemorySnapshotDelegateRead(buffer.get(), size_);
  }

  const MemorySnapshot* MergeWithOtherSnapshot(
      const MemorySnapshot* other) const override {
    return MergeWithOtherSnapshotImpl(this, other);
  }

 private:
  template <class T>
  friend const MemorySnapshot* MergeWithOtherSnapshotImpl(
      const T* self,
      const MemorySnapshot* other);

  ProcessReaderType* process_reader_;  // weak
  uint64_t address_;
  uint64_t size_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(MemorySnapshotGeneric);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_GENERIC_MEMORY_SNAPSHOT_GENERIC_H_
