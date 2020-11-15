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

#include "util/file/string_file.h"

#include <string.h>

#include <algorithm>
#include <limits>

#include "base/logging.h"
#include "base/numerics/safe_math.h"
#include "util/misc/implicit_cast.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {

StringFile::StringFile() : string_(), offset_(0) {
}

StringFile::~StringFile() {
}

void StringFile::SetString(const std::string& string) {
  CHECK_LE(
      string.size(),
      implicit_cast<size_t>(std::numeric_limits<FileOperationResult>::max()));
  string_ = string;
  offset_ = 0;
}

void StringFile::Reset() {
  string_.clear();
  offset_ = 0;
}

FileOperationResult StringFile::Read(void* data, size_t size) {
  DCHECK(offset_.IsValid());

  const size_t offset = offset_.ValueOrDie();
  if (offset >= string_.size()) {
    return 0;
  }

  const size_t nread = std::min(size, string_.size() - offset);

  base::CheckedNumeric<FileOperationResult> new_offset = offset_;
  new_offset += nread;
  if (!new_offset.IsValid()) {
    LOG(ERROR) << "Read(): file too large";
    return -1;
  }

  memcpy(data, &string_[offset], nread);
  offset_ = new_offset;

  return nread;
}

bool StringFile::Write(const void* data, size_t size) {
  DCHECK(offset_.IsValid());

  const size_t offset = offset_.ValueOrDie();
  if (offset > string_.size()) {
    string_.resize(offset);
  }

  base::CheckedNumeric<FileOperationResult> new_offset = offset_;
  new_offset += size;
  if (!new_offset.IsValid()) {
    LOG(ERROR) << "Write(): file too large";
    return false;
  }

  string_.replace(offset, size, reinterpret_cast<const char*>(data), size);
  offset_ = new_offset;

  return true;
}

bool StringFile::WriteIoVec(std::vector<WritableIoVec>* iovecs) {
  DCHECK(offset_.IsValid());

  if (iovecs->empty()) {
    LOG(ERROR) << "WriteIoVec(): no iovecs";
    return false;
  }

  // Avoid writing anything at all if it would cause an overflow.
  base::CheckedNumeric<FileOperationResult> new_offset = offset_;
  for (const WritableIoVec& iov : *iovecs) {
    new_offset += iov.iov_len;
    if (!new_offset.IsValid()) {
      LOG(ERROR) << "WriteIoVec(): file too large";
      return false;
    }
  }

  for (const WritableIoVec& iov : *iovecs) {
    if (!Write(iov.iov_base, iov.iov_len)) {
      return false;
    }
  }

#ifndef NDEBUG
  // The interface says that |iovecs| is not sacred, so scramble it to make sure
  // that nobody depends on it.
  memset(&(*iovecs)[0], 0xa5, sizeof((*iovecs)[0]) * iovecs->size());
#endif

  return true;
}

FileOffset StringFile::Seek(FileOffset offset, int whence) {
  DCHECK(offset_.IsValid());

  size_t base_offset;

  switch (whence) {
    case SEEK_SET:
      base_offset = 0;
      break;

    case SEEK_CUR:
      base_offset = offset_.ValueOrDie();
      break;

    case SEEK_END:
      base_offset = string_.size();
      break;

    default:
      LOG(ERROR) << "Seek(): invalid whence " << whence;
      return -1;
  }

  FileOffset base_offset_fileoffset;
  if (!AssignIfInRange(&base_offset_fileoffset, base_offset)) {
    LOG(ERROR) << "Seek(): base_offset " << base_offset
               << " invalid for FileOffset";
    return -1;
  }
  base::CheckedNumeric<FileOffset> new_offset(base_offset_fileoffset);
  new_offset += offset;
  if (!new_offset.IsValid()) {
    LOG(ERROR) << "Seek(): new_offset invalid";
    return -1;
  }
  size_t new_offset_sizet;
  if (!new_offset.AssignIfValid(&new_offset_sizet)) {
    LOG(ERROR) << "Seek(): new_offset " << new_offset.ValueOrDie()
               << " invalid for size_t";
    return -1;
  }

  offset_ = new_offset_sizet;

  return base::ValueOrDieForType<FileOffset>(offset_);
}

}  // namespace crashpad
