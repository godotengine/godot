// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/date_parser.h"

#include <cassert>
#include <cstdint>
#include <limits>

#include "src/parser_utils.h"
#include "webm/element.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#EBML_ex
// https://github.com/Matroska-Org/ebml-specification/blob/master/specification.markdown#ebml-element-types
DateParser::DateParser(std::int64_t default_value)
    : default_value_(default_value) {}

Status DateParser::Init(const ElementMetadata& metadata,
                        std::uint64_t max_size) {
  assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

  if (metadata.size != 0 && metadata.size != 8) {
    return Status(Status::kInvalidElementSize);
  }

  num_bytes_remaining_ = static_cast<int>(metadata.size);

  // The meaning of a 0-byte element is still being debated. EBML says the value
  // is zero; Matroska says it's the default according to whatever the document
  // spec says. Neither specifies what a 0-byte mandatory element means. I've
  // asked about this on the Matroska mailing list. I'm going to assume a 0-byte
  // mandatory element should be treated the same as a 0-byte optional element,
  // meaning that they both get their default value (which may be some value
  // other than zero). This applies to all non-master-elements (not just dates).
  // This parser is an EBML-level parser, and so will default to a value of
  // zero. The Matroska-level parser can reset this default value to something
  // else after parsing (as needed).
  // See:
  // https://github.com/Matroska-Org/ebml-specification/pull/17
  // http://lists.matroska.org/pipermail/matroska-devel/2015-October/004866.html
  if (metadata.size == 0) {
    value_ = default_value_;
  } else {
    value_ = 0;
  }

  return Status(Status::kOkCompleted);
}

Status DateParser::Feed(Callback* callback, Reader* reader,
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
