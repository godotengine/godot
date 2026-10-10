// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/skip_parser.h"

#include <cassert>
#include <cstdint>

#include "webm/element.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

Status SkipParser::Init(const ElementMetadata& metadata,
                        std::uint64_t max_size) {
  assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

  if (metadata.size == kUnknownElementSize) {
    return Status(Status::kInvalidElementSize);
  }

  num_bytes_remaining_ = metadata.size;

  return Status(Status::kOkCompleted);
}

Status SkipParser::Feed(Callback* callback, Reader* reader,
                        std::uint64_t* num_bytes_read) {
  assert(callback != nullptr);
  assert(reader != nullptr);
  assert(num_bytes_read != nullptr);

  *num_bytes_read = 0;

  if (num_bytes_remaining_ == 0) {
    return Status(Status::kOkCompleted);
  }

  Status status;
  do {
    std::uint64_t local_num_bytes_read = 0;
    status = reader->Skip(num_bytes_remaining_, &local_num_bytes_read);
    assert((status.completed_ok() &&
            local_num_bytes_read == num_bytes_remaining_) ||
           (status.ok() && local_num_bytes_read < num_bytes_remaining_) ||
           (!status.ok() && local_num_bytes_read == 0));
    *num_bytes_read += local_num_bytes_read;
    num_bytes_remaining_ -= local_num_bytes_read;
  } while (status.code == Status::kOkPartial);

  return status;
}

}  // namespace webm
