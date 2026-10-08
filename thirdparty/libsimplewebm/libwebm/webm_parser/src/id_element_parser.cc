// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/id_element_parser.h"

#include <cassert>
#include <cstdint>

#include "src/element_parser.h"
#include "src/parser_utils.h"
#include "webm/callback.h"
#include "webm/element.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

Status IdElementParser::Init(const ElementMetadata& metadata,
                             std::uint64_t max_size) {
  assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

  if (metadata.size == 0 || metadata.size > 4) {
    return Status(Status::kInvalidElementSize);
  }

  num_bytes_remaining_ = static_cast<int>(metadata.size);
  value_ = static_cast<Id>(0);

  return Status(Status::kOkCompleted);
}

Status IdElementParser::Feed(Callback* callback, Reader* reader,
                             std::uint64_t* num_bytes_read) {
  assert(callback != nullptr);
  assert(reader != nullptr);
  assert(num_bytes_read != nullptr);

  const Status status = AccumulateIntegerBytes(num_bytes_remaining_, reader,
                                               &value_, num_bytes_read);
  num_bytes_remaining_ -= static_cast<int>(*num_bytes_read);

  return status;
}

}  // namespace webm
