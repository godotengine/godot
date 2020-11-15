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

#include "snapshot/sanitized/memory_snapshot_sanitized.h"

#include <string.h>

namespace crashpad {
namespace internal {

namespace {

class MemorySanitizer : public MemorySnapshot::Delegate {
 public:
  MemorySanitizer(MemorySnapshot::Delegate* delegate,
                  RangeSet* ranges,
                  VMAddress address,
                  bool is_64_bit)
      : delegate_(delegate),
        ranges_(ranges),
        address_(address),
        is_64_bit_(is_64_bit) {}

  ~MemorySanitizer() = default;

  bool MemorySnapshotDelegateRead(void* data, size_t size) override {
    if (is_64_bit_) {
      Sanitize<uint64_t>(data, size);
    } else {
      Sanitize<uint32_t>(data, size);
    }
    return delegate_->MemorySnapshotDelegateRead(data, size);
  }

 private:
  template <typename Pointer>
  void Sanitize(void* data, size_t size) {
    const Pointer defaced =
        static_cast<Pointer>(MemorySnapshotSanitized::kDefaced);

    // Sanitize up to a word-aligned address.
    const size_t aligned_offset =
        ((address_ + sizeof(Pointer) - 1) & ~(sizeof(Pointer) - 1)) - address_;
    memcpy(data, &defaced, aligned_offset);

    // Sanitize words that aren't small and don't look like pointers.
    size_t word_count = (size - aligned_offset) / sizeof(Pointer);
    auto words =
        reinterpret_cast<Pointer*>(static_cast<char*>(data) + aligned_offset);
    for (size_t index = 0; index < word_count; ++index) {
      if (words[index] > MemorySnapshotSanitized::kSmallWordMax &&
          !ranges_->Contains(words[index])) {
        words[index] = defaced;
      }
    }

    // Sanitize trailing bytes beyond the word-sized items.
    const size_t sanitized_bytes =
        aligned_offset + word_count * sizeof(Pointer);
    memcpy(static_cast<char*>(data) + sanitized_bytes,
           &defaced,
           size - sanitized_bytes);
  }

  MemorySnapshot::Delegate* delegate_;
  RangeSet* ranges_;
  VMAddress address_;
  bool is_64_bit_;

  DISALLOW_COPY_AND_ASSIGN(MemorySanitizer);
};

}  // namespace

MemorySnapshotSanitized::MemorySnapshotSanitized(const MemorySnapshot* snapshot,
                                                 RangeSet* ranges,
                                                 bool is_64_bit)
    : snapshot_(snapshot), ranges_(ranges), is_64_bit_(is_64_bit) {}

MemorySnapshotSanitized::~MemorySnapshotSanitized() = default;

uint64_t MemorySnapshotSanitized::Address() const {
  return snapshot_->Address();
}

size_t MemorySnapshotSanitized::Size() const {
  return snapshot_->Size();
}

bool MemorySnapshotSanitized::Read(Delegate* delegate) const {
  MemorySanitizer sanitizer(delegate, ranges_, Address(), is_64_bit_);
  return snapshot_->Read(&sanitizer);
}

}  // namespace internal
}  // namespace crashpad
