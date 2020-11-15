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

#include "minidump/minidump_memory_info_writer.h"

#include "base/logging.h"
#include "snapshot/memory_map_region_snapshot.h"
#include "util/file/file_writer.h"

namespace crashpad {

MinidumpMemoryInfoListWriter::MinidumpMemoryInfoListWriter()
    : memory_info_list_base_(), items_() {
}

MinidumpMemoryInfoListWriter::~MinidumpMemoryInfoListWriter() {
}

void MinidumpMemoryInfoListWriter::InitializeFromSnapshot(
    const std::vector<const MemoryMapRegionSnapshot*>& memory_map) {
  DCHECK_EQ(state(), kStateMutable);

  DCHECK(items_.empty());
  for (const auto& region : memory_map)
    items_.push_back(region->AsMinidumpMemoryInfo());
}

bool MinidumpMemoryInfoListWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpStreamWriter::Freeze())
    return false;

  memory_info_list_base_.SizeOfHeader = sizeof(MINIDUMP_MEMORY_INFO_LIST);
  memory_info_list_base_.SizeOfEntry = sizeof(MINIDUMP_MEMORY_INFO);
  memory_info_list_base_.NumberOfEntries = items_.size();

  return true;
}

size_t MinidumpMemoryInfoListWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);
  return sizeof(memory_info_list_base_) + sizeof(items_[0]) * items_.size();
}

std::vector<internal::MinidumpWritable*>
MinidumpMemoryInfoListWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);
  return std::vector<internal::MinidumpWritable*>();
}

bool MinidumpMemoryInfoListWriter::WriteObject(
    FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  WritableIoVec iov;
  iov.iov_base = &memory_info_list_base_;
  iov.iov_len = sizeof(memory_info_list_base_);
  std::vector<WritableIoVec> iovecs(1, iov);

  for (const auto& minidump_memory_info : items_) {
    iov.iov_base = &minidump_memory_info;
    iov.iov_len = sizeof(minidump_memory_info);
    iovecs.push_back(iov);
  }

  return file_writer->WriteIoVec(&iovecs);
}

MinidumpStreamType MinidumpMemoryInfoListWriter::StreamType() const {
  return kMinidumpStreamTypeMemoryInfoList;
}

}  // namespace crashpad
