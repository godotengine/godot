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

#include "minidump/minidump_memory_writer.h"

#include <algorithm>
#include <iterator>
#include <utility>

#include "base/auto_reset.h"
#include "base/logging.h"
#include "util/file/file_writer.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {

SnapshotMinidumpMemoryWriter::SnapshotMinidumpMemoryWriter(
    const MemorySnapshot* memory_snapshot)
    : internal::MinidumpWritable(),
      MemorySnapshot::Delegate(),
      memory_descriptor_(),
      registered_memory_descriptors_(),
      memory_snapshot_(memory_snapshot),
      file_writer_(nullptr) {}

SnapshotMinidumpMemoryWriter::~SnapshotMinidumpMemoryWriter() {}

bool SnapshotMinidumpMemoryWriter::MemorySnapshotDelegateRead(void* data,
                                                              size_t size) {
  DCHECK_EQ(state(), kStateWritable);
  DCHECK_EQ(size, UnderlyingSnapshot()->Size());
  return file_writer_->Write(data, size);
}

bool SnapshotMinidumpMemoryWriter::WriteObject(
    FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);
  DCHECK(!file_writer_);

  base::AutoReset<FileWriterInterface*> file_writer_reset(&file_writer_,
                                                          file_writer);

  // This will result in MemorySnapshotDelegateRead() being called.
  if (!memory_snapshot_->Read(this)) {
    // If the Read() fails (perhaps because the process' memory map has changed
    // since it the range was captured), write an empty block of memory. It
    // would be nice to instead not include this memory, but at this point in
    // the writing process, it would be difficult to amend the minidump's
    // structure. See https://crashpad.chromium.org/234 for background.
    std::vector<uint8_t> empty(memory_snapshot_->Size(), 0xfe);
    MemorySnapshotDelegateRead(empty.data(), empty.size());
  }

  return true;
}

const MINIDUMP_MEMORY_DESCRIPTOR*
SnapshotMinidumpMemoryWriter::MinidumpMemoryDescriptor() const {
  DCHECK_EQ(state(), kStateWritable);

  return &memory_descriptor_;
}

void SnapshotMinidumpMemoryWriter::RegisterMemoryDescriptor(
    MINIDUMP_MEMORY_DESCRIPTOR* memory_descriptor) {
  DCHECK_LE(state(), kStateFrozen);

  registered_memory_descriptors_.push_back(memory_descriptor);
  RegisterLocationDescriptor(&memory_descriptor->Memory);
}

bool SnapshotMinidumpMemoryWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  RegisterMemoryDescriptor(&memory_descriptor_);

  return true;
}

size_t SnapshotMinidumpMemoryWriter::Alignment() {
  DCHECK_GE(state(), kStateFrozen);

  return 16;
}

size_t SnapshotMinidumpMemoryWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return UnderlyingSnapshot()->Size();
}

bool SnapshotMinidumpMemoryWriter::WillWriteAtOffsetImpl(FileOffset offset) {
  DCHECK_EQ(state(), kStateFrozen);

  // There will always be at least one registered descriptor, the one for this
  // object’s own memory_descriptor_ field.
  DCHECK_GE(registered_memory_descriptors_.size(), 1u);

  uint64_t base_address = UnderlyingSnapshot()->Address();
  decltype(registered_memory_descriptors_[0]->StartOfMemoryRange) local_address;
  if (!AssignIfInRange(&local_address, base_address)) {
    LOG(ERROR) << "base_address " << base_address << " out of range";
    return false;
  }

  for (MINIDUMP_MEMORY_DESCRIPTOR* memory_descriptor :
           registered_memory_descriptors_) {
    memory_descriptor->StartOfMemoryRange = local_address;
  }

  return MinidumpWritable::WillWriteAtOffsetImpl(offset);
}

internal::MinidumpWritable::Phase SnapshotMinidumpMemoryWriter::WritePhase() {
  // Memory dumps are large and are unlikely to be consumed in their entirety.
  // Data accesses are expected to be sparse and sporadic, and are expected to
  // occur after all of the other structural and informational data from the
  // minidump file has been read. Put memory dumps at the end of the minidump
  // file to improve spatial locality.
  return kPhaseLate;
}

MinidumpMemoryListWriter::MinidumpMemoryListWriter()
    : MinidumpStreamWriter(),
      non_owned_memory_writers_(),
      children_(),
      snapshots_created_during_merge_(),
      all_memory_writers_(),
      memory_list_base_() {}

MinidumpMemoryListWriter::~MinidumpMemoryListWriter() {
}

void MinidumpMemoryListWriter::AddFromSnapshot(
    const std::vector<const MemorySnapshot*>& memory_snapshots) {
  DCHECK_EQ(state(), kStateMutable);

  for (const MemorySnapshot* memory_snapshot : memory_snapshots) {
    std::unique_ptr<SnapshotMinidumpMemoryWriter> memory(
        new SnapshotMinidumpMemoryWriter(memory_snapshot));
    AddMemory(std::move(memory));
  }
}

void MinidumpMemoryListWriter::AddMemory(
    std::unique_ptr<SnapshotMinidumpMemoryWriter> memory_writer) {
  DCHECK_EQ(state(), kStateMutable);

  children_.push_back(std::move(memory_writer));
}

void MinidumpMemoryListWriter::AddNonOwnedMemory(
    SnapshotMinidumpMemoryWriter* memory_writer) {
  DCHECK_EQ(state(), kStateMutable);

  non_owned_memory_writers_.push_back(memory_writer);
}

void MinidumpMemoryListWriter::CoalesceOwnedMemory() {
  if (children_.empty())
    return;

  DropRangesThatOverlapNonOwned();

  std::sort(children_.begin(),
            children_.end(),
            [](const std::unique_ptr<SnapshotMinidumpMemoryWriter>& a_ptr,
               const std::unique_ptr<SnapshotMinidumpMemoryWriter>& b_ptr) {
              const MemorySnapshot* a = a_ptr->UnderlyingSnapshot();
              const MemorySnapshot* b = b_ptr->UnderlyingSnapshot();
              if (a->Address() == b->Address()) {
                return a->Size() < b->Size();
              }
              return a->Address() < b->Address();
            });

  // Remove any empty ranges.
  children_.erase(
      std::remove_if(children_.begin(),
                     children_.end(),
                     [](const auto& snapshot) {
                       return snapshot->UnderlyingSnapshot()->Size() == 0;
                     }),
      children_.end());

  std::vector<std::unique_ptr<SnapshotMinidumpMemoryWriter>> all_merged;
  all_merged.push_back(std::move(children_.front()));
  for (size_t i = 1; i < children_.size(); ++i) {
    SnapshotMinidumpMemoryWriter* top = all_merged.back().get();
    auto& child = children_[i];
    if (!DetermineMergedRange(
            child->UnderlyingSnapshot(), top->UnderlyingSnapshot(), nullptr)) {
      // If it doesn't overlap with the current range, push it.
      all_merged.push_back(std::move(child));
    } else {
      // Otherwise, merge and update the current element.
      std::unique_ptr<const MemorySnapshot> merged(
          top->UnderlyingSnapshot()->MergeWithOtherSnapshot(
              child->UnderlyingSnapshot()));
      top->SetSnapshot(merged.get());
      snapshots_created_during_merge_.push_back(std::move(merged));
    }
  }
  std::swap(children_, all_merged);
}

bool MinidumpMemoryListWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  CoalesceOwnedMemory();

  std::copy(non_owned_memory_writers_.begin(),
            non_owned_memory_writers_.end(),
            std::back_inserter(all_memory_writers_));
  for (const auto& ptr : children_)
    all_memory_writers_.push_back(ptr.get());

  if (!MinidumpStreamWriter::Freeze()) {
    return false;
  }

  size_t memory_region_count = all_memory_writers_.size();
  CHECK_LE(children_.size(), memory_region_count);

  if (!AssignIfInRange(&memory_list_base_.NumberOfMemoryRanges,
                       memory_region_count)) {
    LOG(ERROR) << "memory_region_count " << memory_region_count
               << " out of range";
    return false;
  }

  return true;
}

size_t MinidumpMemoryListWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);
  DCHECK_LE(children_.size(), all_memory_writers_.size());

  return sizeof(memory_list_base_) +
         all_memory_writers_.size() * sizeof(MINIDUMP_MEMORY_DESCRIPTOR);
}

std::vector<internal::MinidumpWritable*> MinidumpMemoryListWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);
  DCHECK_LE(children_.size(), all_memory_writers_.size());

  std::vector<MinidumpWritable*> children;
  for (const auto& child : children_) {
    children.push_back(child.get());
  }

  return children;
}

bool MinidumpMemoryListWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  WritableIoVec iov;
  iov.iov_base = &memory_list_base_;
  iov.iov_len = sizeof(memory_list_base_);
  std::vector<WritableIoVec> iovecs(1, iov);

  for (const SnapshotMinidumpMemoryWriter* memory_writer :
       all_memory_writers_) {
    iov.iov_base = memory_writer->MinidumpMemoryDescriptor();
    iov.iov_len = sizeof(MINIDUMP_MEMORY_DESCRIPTOR);
    iovecs.push_back(iov);
  }

  return file_writer->WriteIoVec(&iovecs);
}

MinidumpStreamType MinidumpMemoryListWriter::StreamType() const {
  return kMinidumpStreamTypeMemoryList;
}

void MinidumpMemoryListWriter::DropRangesThatOverlapNonOwned() {
  std::vector<std::unique_ptr<SnapshotMinidumpMemoryWriter>> non_overlapping;
  non_overlapping.reserve(children_.size());
  for (auto& child_ptr : children_) {
    bool overlaps = false;
    for (const auto* non_owned : non_owned_memory_writers_) {
      if (DetermineMergedRange(child_ptr->UnderlyingSnapshot(),
                               non_owned->UnderlyingSnapshot(),
                               nullptr)) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps)
      non_overlapping.push_back(std::move(child_ptr));
  }
  std::swap(children_, non_overlapping);
}

}  // namespace crashpad
