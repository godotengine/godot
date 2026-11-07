// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_BOOL_PARSER_H_
#define SRC_BOOL_PARSER_H_

#include <cassert>
#include <cstdint>

#include "src/element_parser.h"
#include "src/parser_utils.h"
#include "webm/callback.h"
#include "webm/element.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Parses a boolean from a byte stream. EBML does not have a boolean type, but
// the Matroska spec defines some unsigned integer elements that have a range of
// [0, 1]. The BoolParser parses these unsigned integer elements into
// true/false, and reports a Status::kInvalidElementValue error if the integer
// is outside of its permitted range.
class BoolParser : public ElementParser {
 public:
  explicit BoolParser(bool default_value = false)
      : default_value_(default_value) {}

  Status Init(const ElementMetadata& metadata,
              std::uint64_t max_size) override {
    assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

    // Booleans are really just unsigned integers with a range limit of 0-1.
    // Unsigned integers can't be encoded with more than 8 bytes.
    if (metadata.size > 8) {
      return Status(Status::kInvalidElementSize);
    }

    size_ = num_bytes_remaining_ = static_cast<int>(metadata.size);
    value_ = default_value_;

    return Status(Status::kOkCompleted);
  }

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override {
    assert(callback != nullptr);
    assert(reader != nullptr);
    assert(num_bytes_read != nullptr);

    std::uint64_t uint_value = 0;
    const Status status = AccumulateIntegerBytes(num_bytes_remaining_, reader,
                                                 &uint_value, num_bytes_read);
    num_bytes_remaining_ -= static_cast<int>(*num_bytes_read);

    // Only the last byte should have a value, and it should only be 0 or 1.
    if ((num_bytes_remaining_ != 0 && uint_value != 0) || uint_value > 1) {
      return Status(Status::kInvalidElementValue);
    }

    if (size_ > 0) {
      value_ = uint_value == 1;
    }

    return status;
  }

  // Gets the parsed bool. This must not be called until the parse had been
  // successfully completed.
  bool value() const {
    assert(num_bytes_remaining_ == 0);
    return value_;
  }

  // Gets the parsed bool. This must not be called until the parse had been
  // successfully completed.
  bool* mutable_value() {
    assert(num_bytes_remaining_ == 0);
    return &value_;
  }

 private:
  bool value_;
  bool default_value_;
  int num_bytes_remaining_ = -1;
  int size_;
};

}  // namespace webm

#endif  // SRC_BOOL_PARSER_H_
