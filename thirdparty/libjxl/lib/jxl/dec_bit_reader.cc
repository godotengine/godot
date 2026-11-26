// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_bit_reader.h"

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"

namespace jxl {

void BitReader::BoundsCheckedRefill() {
  const uint8_t* end = end_minus_8_ + 8;

  // Read whole bytes until we have [56, 64) bits (same as LoadLE64)
  for (; bits_in_buf_ < 64 - kBitsPerByte; bits_in_buf_ += kBitsPerByte) {
    if (next_byte_ >= end) break;
    buf_ |= static_cast<uint64_t>(*next_byte_++) << bits_in_buf_;
  }
  JXL_DASSERT(bits_in_buf_ < 64);

  // Add extra bytes as 0 at the end of the stream in the bit_buffer_. If
  // these bits are read, Close() will return a failure.
  size_t extra_bytes = (63 - bits_in_buf_) / kBitsPerByte;
  overread_bytes_ += extra_bytes;
  bits_in_buf_ += extra_bytes * kBitsPerByte;

  JXL_DASSERT(bits_in_buf_ < 64);
  JXL_DASSERT(bits_in_buf_ >= 56);
}

}  // namespace jxl
