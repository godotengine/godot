// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/bit_utils.h"

#include <cstdint>

namespace webm {

std::uint8_t CountLeadingZeros(std::uint8_t value) {
  // Special case for 0 since we can't shift by sizeof(T) * 8 bytes.
  if (value == 0)
    return 8;

  std::uint8_t count = 0;
  while (!(value & (0x80 >> count))) {
    ++count;
  }
  return count;
}

}  // namespace webm
