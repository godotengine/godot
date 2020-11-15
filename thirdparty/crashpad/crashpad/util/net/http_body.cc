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

#include "util/net/http_body.h"

#include <string.h>

#include <algorithm>
#include <limits>

#include "base/logging.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {

StringHTTPBodyStream::StringHTTPBodyStream(const std::string& string)
    : HTTPBodyStream(), string_(string), bytes_read_() {
}

StringHTTPBodyStream::~StringHTTPBodyStream() {
}

FileOperationResult StringHTTPBodyStream::GetBytesBuffer(uint8_t* buffer,
                                                         size_t max_len) {
  size_t num_bytes_remaining = string_.length() - bytes_read_;
  if (num_bytes_remaining == 0) {
    return num_bytes_remaining;
  }

  size_t num_bytes_returned = std::min(
      std::min(num_bytes_remaining, max_len),
      implicit_cast<size_t>(std::numeric_limits<FileOperationResult>::max()));
  memcpy(buffer, &string_[bytes_read_], num_bytes_returned);
  bytes_read_ += num_bytes_returned;
  return num_bytes_returned;
}

FileReaderHTTPBodyStream::FileReaderHTTPBodyStream(FileReaderInterface* reader)
    : HTTPBodyStream(), reader_(reader), reached_eof_(false) {
  DCHECK(reader_);
}

FileReaderHTTPBodyStream::~FileReaderHTTPBodyStream() {}

FileOperationResult FileReaderHTTPBodyStream::GetBytesBuffer(uint8_t* buffer,
                                                             size_t max_len) {
  if (reached_eof_) {
    return 0;
  }

  FileOperationResult rv = reader_->Read(buffer, max_len);
  if (rv == 0) {
    reached_eof_ = true;
  }
  return rv;
}

CompositeHTTPBodyStream::CompositeHTTPBodyStream(
    const CompositeHTTPBodyStream::PartsList& parts)
    : HTTPBodyStream(), parts_(parts), current_part_(parts_.begin()) {
}

CompositeHTTPBodyStream::~CompositeHTTPBodyStream() {
  for (auto& item : parts_)
    delete item;
}

FileOperationResult CompositeHTTPBodyStream::GetBytesBuffer(uint8_t* buffer,
                                                            size_t buffer_len) {
  FileOperationResult max_len = std::min(
      buffer_len,
      implicit_cast<size_t>(std::numeric_limits<FileOperationResult>::max()));
  FileOperationResult bytes_copied = 0;
  while (bytes_copied < max_len && current_part_ != parts_.end()) {
    FileOperationResult this_read =
        (*current_part_)
            ->GetBytesBuffer(buffer + bytes_copied, max_len - bytes_copied);

    if (this_read == 0) {
      // If the current part has returned 0 indicating EOF, advance the current
      // part and try again.
      ++current_part_;
    } else if (this_read < 0) {
      return this_read;
    }
    bytes_copied += this_read;
  }

  return bytes_copied;
}

}  // namespace crashpad
