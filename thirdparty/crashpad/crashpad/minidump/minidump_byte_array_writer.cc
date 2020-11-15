// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "minidump/minidump_byte_array_writer.h"

#include "base/logging.h"
#include "util/file/file_writer.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {

MinidumpByteArrayWriter::MinidumpByteArrayWriter()
    : minidump_array_(new MinidumpByteArray()) {}

MinidumpByteArrayWriter::~MinidumpByteArrayWriter() = default;

void MinidumpByteArrayWriter::set_data(const uint8_t* data, size_t size) {
  data_.clear();
  data_.insert(data_.begin(), data, data + size);
}

bool MinidumpByteArrayWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  size_t size = data_.size();
  if (!AssignIfInRange(&minidump_array_->length, size)) {
    LOG(ERROR) << "data size " << size << " is out of range";
    return false;
  }

  return true;
}

size_t MinidumpByteArrayWriter::SizeOfObject() {
  DCHECK_EQ(state(), kStateFrozen);

  return sizeof(*minidump_array_) + data_.size();
}

bool MinidumpByteArrayWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  WritableIoVec iov;
  iov.iov_base = minidump_array_.get();
  iov.iov_len = sizeof(*minidump_array_);

  std::vector<WritableIoVec> iovecs(1, iov);

  if (!data_.empty()) {
    iov.iov_base = data_.data();
    iov.iov_len = data_.size();
    iovecs.push_back(iov);
  }

  return file_writer->WriteIoVec(&iovecs);
}

}  // namespace crashpad
