// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_BYTE_PARSER_H_
#define SRC_BYTE_PARSER_H_

#include <cassert>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "src/element_parser.h"
#include "webm/callback.h"
#include "webm/element.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Parses an EBML string (UTF-8 and ASCII) or binary element from a byte stream.
// Spec reference for string/binary elements:
// http://matroska.org/technical/specs/index.html#EBML_ex
// https://github.com/Matroska-Org/ebml-specification/blob/master/specification.markdown#ebml-element-types
template <typename T>
class ByteParser : public ElementParser {
 public:
  static_assert(std::is_same<T, std::vector<std::uint8_t>>::value ||
                    std::is_same<T, std::string>::value,
                "T must be std::vector<std::uint8_t> or std::string");

  // Constructs a new parser which will use the given default_value as the
  // value for the element if its size is zero. Defaults to the empty string
  // or empty binary element (as the EBML spec indicates).
  explicit ByteParser(T default_value = {})
      : default_value_(std::move(default_value)) {}

  ByteParser(ByteParser&&) = default;
  ByteParser& operator=(ByteParser&&) = default;

  ByteParser(const ByteParser&) = delete;
  ByteParser& operator=(const ByteParser&) = delete;

  Status Init(const ElementMetadata& metadata,
              std::uint64_t max_size) override {
    assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

    if (metadata.size == kUnknownElementSize) {
      return Status(Status::kInvalidElementSize);
    }

    if (metadata.size > std::numeric_limits<std::size_t>::max() ||
        metadata.size > value_.max_size()) {
      return Status(Status::kNotEnoughMemory);
    }

#if WEBM_FUZZER_BYTE_ELEMENT_SIZE_LIMIT
    // AFL and ASan just kill the process if too much memory is allocated, so
    // let's cap the maximum size of the element. It's too easy for the fuzzer
    // to make an element with a ridiculously huge size, and that just creates
    // uninteresting false positives.
    if (metadata.size > WEBM_FUZZER_BYTE_ELEMENT_SIZE_LIMIT) {
      return Status(Status::kNotEnoughMemory);
    }
#endif

    if (metadata.size == 0) {
      value_ = default_value_;
      total_read_ = default_value_.size();
    } else {
      value_.resize(static_cast<std::size_t>(metadata.size));
      total_read_ = 0;
    }

    return Status(Status::kOkCompleted);
  }

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override {
    assert(callback != nullptr);
    assert(reader != nullptr);
    assert(num_bytes_read != nullptr);

    *num_bytes_read = 0;

    if (total_read_ == value_.size()) {
      return Status(Status::kOkCompleted);
    }

    Status status;
    do {
      std::uint64_t local_num_bytes_read = 0;
      std::uint8_t* buffer =
          reinterpret_cast<std::uint8_t*>(&value_.front()) + total_read_;
      std::size_t buffer_size = value_.size() - total_read_;
      status = reader->Read(buffer_size, buffer, &local_num_bytes_read);
      assert((status.completed_ok() && local_num_bytes_read == buffer_size) ||
             (status.ok() && local_num_bytes_read < buffer_size) ||
             (!status.ok() && local_num_bytes_read == 0));
      *num_bytes_read += local_num_bytes_read;
      total_read_ += static_cast<std::size_t>(local_num_bytes_read);
    } while (status.code == Status::kOkPartial);

    // UTF-8 and ASCII string elements can be padded with NUL characters at the
    // end, which should be ignored.
    if (std::is_same<T, std::string>::value && status.completed_ok()) {
      while (!value_.empty() && value_.back() == '\0') {
        value_.pop_back();
      }
    }

    return status;
  }

  // Gets the parsed value. This must not be called until the parse has been
  // successfully completed.
  const T& value() const {
    assert(total_read_ >= value_.size());
    return value_;
  }

  // Gets the parsed value. This must not be called until the parse has been
  // successfully completed.
  T* mutable_value() {
    assert(total_read_ >= value_.size());
    return &value_;
  }

 private:
  T value_;
  T default_value_;
  std::size_t total_read_;
};

using StringParser = ByteParser<std::string>;
using BinaryParser = ByteParser<std::vector<std::uint8_t>>;

}  // namespace webm

#endif  // SRC_BYTE_PARSER_H_
