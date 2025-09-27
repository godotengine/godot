// Copyright (c) 2015-2016 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "source/spirv_endian.h"

#include <cstring>

enum {
  I32_ENDIAN_LITTLE = 0x03020100ul,
  I32_ENDIAN_BIG = 0x00010203ul,
};

// This constant value allows the detection of the host machine's endianness.
// Accessing it through the "value" member is valid due to C++11 section 3.10
// paragraph 10.
static const union {
  unsigned char bytes[4];
  uint32_t value;
} o32_host_order = {{0, 1, 2, 3}};

#define I32_ENDIAN_HOST (o32_host_order.value)

uint32_t spvFixWord(const uint32_t word, const spv_endianness_t endian) {
  if ((SPV_ENDIANNESS_LITTLE == endian && I32_ENDIAN_HOST == I32_ENDIAN_BIG) ||
      (SPV_ENDIANNESS_BIG == endian && I32_ENDIAN_HOST == I32_ENDIAN_LITTLE)) {
    return (word & 0x000000ff) << 24 | (word & 0x0000ff00) << 8 |
           (word & 0x00ff0000) >> 8 | (word & 0xff000000) >> 24;
  }

  return word;
}

uint64_t spvFixDoubleWord(const uint32_t low, const uint32_t high,
                          const spv_endianness_t endian) {
  return (uint64_t(spvFixWord(high, endian)) << 32) | spvFixWord(low, endian);
}

spv_result_t spvBinaryEndianness(spv_const_binary binary,
                                 spv_endianness_t* pEndian) {
  if (!binary->code || !binary->wordCount) return SPV_ERROR_INVALID_BINARY;
  if (!pEndian) return SPV_ERROR_INVALID_POINTER;

  uint8_t bytes[4];
  memcpy(bytes, binary->code, sizeof(uint32_t));

  if (0x03 == bytes[0] && 0x02 == bytes[1] && 0x23 == bytes[2] &&
      0x07 == bytes[3]) {
    *pEndian = SPV_ENDIANNESS_LITTLE;
    return SPV_SUCCESS;
  }

  if (0x07 == bytes[0] && 0x23 == bytes[1] && 0x02 == bytes[2] &&
      0x03 == bytes[3]) {
    *pEndian = SPV_ENDIANNESS_BIG;
    return SPV_SUCCESS;
  }

  return SPV_ERROR_INVALID_BINARY;
}

bool spvIsHostEndian(spv_endianness_t endian) {
  return ((SPV_ENDIANNESS_LITTLE == endian) &&
          (I32_ENDIAN_LITTLE == I32_ENDIAN_HOST)) ||
         ((SPV_ENDIANNESS_BIG == endian) &&
          (I32_ENDIAN_BIG == I32_ENDIAN_HOST));
}
