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

#include "test/unit_spirv.h"

namespace spvtools {
namespace {

TEST(FixWord, Default) {
  spv_endianness_t endian;
  if (I32_ENDIAN_HOST == I32_ENDIAN_LITTLE) {
    endian = SPV_ENDIANNESS_LITTLE;
  } else {
    endian = SPV_ENDIANNESS_BIG;
  }
  uint32_t word = 0x53780921;
  ASSERT_EQ(word, spvFixWord(word, endian));
}

TEST(FixWord, Reorder) {
  spv_endianness_t endian;
  if (I32_ENDIAN_HOST == I32_ENDIAN_LITTLE) {
    endian = SPV_ENDIANNESS_BIG;
  } else {
    endian = SPV_ENDIANNESS_LITTLE;
  }
  uint32_t word = 0x53780921;
  uint32_t result = 0x21097853;
  ASSERT_EQ(result, spvFixWord(word, endian));
}

TEST(FixDoubleWord, Default) {
  spv_endianness_t endian =
      (I32_ENDIAN_HOST == I32_ENDIAN_LITTLE ? SPV_ENDIANNESS_LITTLE
                                            : SPV_ENDIANNESS_BIG);
  uint32_t low = 0x53780921;
  uint32_t high = 0xdeadbeef;
  uint64_t result = 0xdeadbeef53780921;
  ASSERT_EQ(result, spvFixDoubleWord(low, high, endian));
}

TEST(FixDoubleWord, Reorder) {
  spv_endianness_t endian =
      (I32_ENDIAN_HOST == I32_ENDIAN_LITTLE ? SPV_ENDIANNESS_BIG
                                            : SPV_ENDIANNESS_LITTLE);
  uint32_t low = 0x53780921;
  uint32_t high = 0xdeadbeef;
  uint64_t result = 0xefbeadde21097853;
  ASSERT_EQ(result, spvFixDoubleWord(low, high, endian));
}

}  // namespace
}  // namespace spvtools
