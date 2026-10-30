// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/var_int_parser.h"

#include <cassert>
#include <cstdint>
#include <limits>

#include "src/bit_utils.h"
#include "src/parser_utils.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Spec references:
// http://matroska.org/technical/specs/index.html#EBML_ex
// https://github.com/Matroska-Org/ebml-specification/blob/master/specification.markdown#variable-size-integer
Status VarIntParser::Feed(Callback* callback, Reader* reader,
                          std::uint64_t* num_bytes_read) {
  assert(callback != nullptr);
  assert(reader != nullptr);
  assert(num_bytes_read != nullptr);
  assert(num_bytes_remaining_ != 0);

  *num_bytes_read = 0;

  if (num_bytes_remaining_ == -1) {
    std::uint8_t first_byte;
    const Status status = ReadByte(reader, &first_byte);
    if (!status.completed_ok()) {
      return status;
    }
    ++*num_bytes_read;

    // The first byte must have a marker bit set to indicate how many octets are
    // used.
    if (first_byte == 0) {
      return Status(Status::kInvalidElementValue);
    }

    total_data_bytes_ = CountLeadingZeros(first_byte);
    num_bytes_remaining_ = total_data_bytes_;

    value_ = first_byte;
  }

  std::uint64_t local_num_bytes_read;
  const Status status = AccumulateIntegerBytes(num_bytes_remaining_, reader,
                                               &value_, &local_num_bytes_read);
  *num_bytes_read += local_num_bytes_read;
  num_bytes_remaining_ -= static_cast<int>(local_num_bytes_read);

  if (!status.completed_ok()) {
    return status;
  }

  // Clear the marker bit.
  constexpr std::uint64_t all_bits = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t data_bits = all_bits >> (57 - 7 * total_data_bytes_);
  value_ &= data_bits;

  return Status(Status::kOkCompleted);
}

}  // namespace webm
