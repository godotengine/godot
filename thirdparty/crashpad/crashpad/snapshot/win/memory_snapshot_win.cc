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

#include <memory>

#include "snapshot/win/memory_snapshot_win.h"

namespace crashpad {
namespace internal {

MemorySnapshotWin::MemorySnapshotWin()
    : MemorySnapshot(),
      process_reader_(nullptr),
      address_(0),
      size_(0),
      initialized_() {
}

MemorySnapshotWin::~MemorySnapshotWin() {
}

void MemorySnapshotWin::Initialize(ProcessReaderWin* process_reader,
                                   uint64_t address,
                                   uint64_t size) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  process_reader_ = process_reader;
  address_ = address;
  DLOG_IF(WARNING, size >= std::numeric_limits<size_t>::max())
      << "size overflow";
  size_ = static_cast<size_t>(size);
  INITIALIZATION_STATE_SET_VALID(initialized_);
}

uint64_t MemorySnapshotWin::Address() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return address_;
}

size_t MemorySnapshotWin::Size() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return size_;
}

bool MemorySnapshotWin::Read(Delegate* delegate) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (size_ == 0) {
    return delegate->MemorySnapshotDelegateRead(nullptr, size_);
  }

  std::unique_ptr<uint8_t[]> buffer(new uint8_t[size_]);
  if (!process_reader_->ReadMemory(address_, size_, buffer.get())) {
    return false;
  }
  return delegate->MemorySnapshotDelegateRead(buffer.get(), size_);
}

const MemorySnapshot* MemorySnapshotWin::MergeWithOtherSnapshot(
    const MemorySnapshot* other) const {
  return MergeWithOtherSnapshotImpl(this, other);
}

}  // namespace internal
}  // namespace crashpad
