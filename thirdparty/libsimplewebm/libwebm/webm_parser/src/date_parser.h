// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_DATE_PARSER_H_
#define SRC_DATE_PARSER_H_

#include <cassert>
#include <cstdint>

#include "src/element_parser.h"
#include "webm/callback.h"
#include "webm/element.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Parses an EBML date from a byte stream. EBML dates are signed integer values
// that represent the offset, in nanoseconds, from 2001-01-01T00:00:00.00 UTC.
class DateParser : public ElementParser {
 public:
  // Constructs a new parser which will use the given default_value as the
  // value for the element if its size is zero. Defaults to the value zero (as
  // the EBML spec indicates).
  explicit DateParser(std::int64_t default_value = 0);

  DateParser(DateParser&&) = default;
  DateParser& operator=(DateParser&&) = default;

  DateParser(const DateParser&) = delete;
  DateParser& operator=(const DateParser&) = delete;

  Status Init(const ElementMetadata& metadata, std::uint64_t max_size) override;

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override;

  // Gets the parsed date. This must not be called until the parse had been
  // successfully completed.
  std::int64_t value() const {
    assert(num_bytes_remaining_ == 0);
    return value_;
  }

  // Gets the parsed date. This must not be called until the parse had been
  // successfully completed.
  std::int64_t* mutable_value() {
    assert(num_bytes_remaining_ == 0);
    return &value_;
  }

 private:
  std::int64_t value_;
  std::int64_t default_value_;
  int num_bytes_remaining_ = -1;
};

}  // namespace webm

#endif  // SRC_DATE_PARSER_H_
