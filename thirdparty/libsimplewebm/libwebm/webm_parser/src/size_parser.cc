// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/size_parser.h"

#include <cassert>
#include <cstdint>
#include <limits>

#include "src/bit_utils.h"
#include "src/parser_utils.h"
#include "webm/element.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Spec references:
// http://matroska.org/technical/specs/index.html#EBML_ex
// https://github.com/Matroska-Org/ebml-specification/blob/master/specification.markdown#element-data-size
Status SizeParser::Feed(Callback* callback, Reader* reader,
                        std::uint64_t* num_bytes_read) {
  assert(callback != nullptr);
  assert(reader != nullptr);
  assert(num_bytes_read != nullptr);

  // Within the EBML header, the size can be encoded with 1-4 octets. After
  // the EBML header, the size can be encoded with 1-8 octets (though not more
  // than EBMLMaxSizeLength).

  Status status = uint_parser_.Feed(callback, reader, num_bytes_read);

  if (status.code == Status::kInvalidElementValue) {
    status.code = Status::kInvalidElementSize;
  }

  return status;
}

std::uint64_t SizeParser::size() const {
  // If all data bits are set, then it represents an unknown element size.
  const std::uint64_t data_bits =
      std::numeric_limits<std::uint64_t>::max() >>
      (57 - 7 * (uint_parser_.encoded_length() - 1));
  if (uint_parser_.value() == data_bits) {
    return kUnknownElementSize;
  }

  return uint_parser_.value();
}

}  // namespace webm
