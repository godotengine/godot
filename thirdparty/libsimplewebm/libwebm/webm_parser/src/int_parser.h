// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_INT_PARSER_H_
#define SRC_INT_PARSER_H_

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "src/element_parser.h"
#include "src/parser_utils.h"
#include "webm/callback.h"
#include "webm/element.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Parses an EBML signed/unsigned int from a byte stream.
// Spec reference:
// http://matroska.org/technical/specs/index.html#EBML_ex
// https://github.com/Matroska-Org/ebml-specification/blob/master/specification.markdown#element-data-size
// https://github.com/Matroska-Org/ebml-specification/blob/master/specification.markdown#ebml-element-types
template <typename T>
class IntParser : public ElementParser {
 public:
  static_assert(
      std::is_same<T, std::int64_t>::value ||
          std::is_same<T, std::uint64_t>::value ||
          (std::is_enum<T>::value && sizeof(T) == 8),
      "T must be either std::int64_t, std::uint64_t, or a 64-bit enum");

  // Constructs a new parser which will use the given default_value as the
  // value for the element if its size is zero. Defaults to the value zero (as
  // the EBML spec indicates).
  explicit IntParser(T default_value = {}) : default_value_(default_value) {}

  IntParser(IntParser&&) = default;
  IntParser& operator=(IntParser&&) = default;

  IntParser(const IntParser&) = delete;
  IntParser& operator=(const IntParser&) = delete;

  Status Init(const ElementMetadata& metadata,
              std::uint64_t max_size) override {
    assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

    // Matroska requires integers to be 0-8 bytes in size.
    if (metadata.size > 8) {
      return Status(Status::kInvalidElementSize);
    }

    size_ = num_bytes_remaining_ = static_cast<int>(metadata.size);

    if (metadata.size == 0) {
      value_ = default_value_;
    } else {
      value_ = {};
    }

    return Status(Status::kOkCompleted);
  }

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override {
    assert(callback != nullptr);
    assert(reader != nullptr);
    assert(num_bytes_read != nullptr);

    const Status status = AccumulateIntegerBytes(num_bytes_remaining_, reader,
                                                 &value_, num_bytes_read);
    num_bytes_remaining_ -= static_cast<int>(*num_bytes_read);

    // Sign extend the integer if it's a negative value. EBML allows for
    // negative integers to drop superfluous sign bytes (i.e. -1 can be encoded
    // as 0xFF instead of 0xFFFFFFFFFFFFFFFF).
    if (std::is_signed<T>::value && num_bytes_remaining_ == 0 && size_ > 0) {
      std::uint64_t sign_bits = std::numeric_limits<std::uint64_t>::max()
                                << (8 * size_ - 1);
      std::uint64_t unsigned_value = static_cast<std::uint64_t>(value_);
      if (unsigned_value & sign_bits) {
        value_ = static_cast<T>(unsigned_value | sign_bits);
      }
    }

    return status;
  }

  // Gets the parsed int. This must not be called until the parse had been
  // successfully completed.
  T value() const {
    assert(num_bytes_remaining_ == 0);
    return value_;
  }

  // Gets the parsed int. This must not be called until the parse had been
  // successfully completed.
  T* mutable_value() {
    assert(num_bytes_remaining_ == 0);
    return &value_;
  }

 private:
  T value_;
  T default_value_;
  int num_bytes_remaining_ = -1;
  int size_;
};

using SignedIntParser = IntParser<std::int64_t>;
using UnsignedIntParser = IntParser<std::uint64_t>;

}  // namespace webm

#endif  // SRC_INT_PARSER_H_
