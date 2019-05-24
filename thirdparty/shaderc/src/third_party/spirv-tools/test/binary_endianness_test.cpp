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

TEST(BinaryEndianness, InvalidCode) {
  uint32_t invalidMagicNumber[] = {0};
  spv_const_binary_t binary = {invalidMagicNumber, 1};
  spv_endianness_t endian;
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, spvBinaryEndianness(&binary, &endian));
}

TEST(BinaryEndianness, Little) {
  uint32_t magicNumber;
  if (I32_ENDIAN_HOST == I32_ENDIAN_LITTLE) {
    magicNumber = 0x07230203;
  } else {
    magicNumber = 0x03022307;
  }
  spv_const_binary_t binary = {&magicNumber, 1};
  spv_endianness_t endian;
  ASSERT_EQ(SPV_SUCCESS, spvBinaryEndianness(&binary, &endian));
  ASSERT_EQ(SPV_ENDIANNESS_LITTLE, endian);
}

TEST(BinaryEndianness, Big) {
  uint32_t magicNumber;
  if (I32_ENDIAN_HOST == I32_ENDIAN_BIG) {
    magicNumber = 0x07230203;
  } else {
    magicNumber = 0x03022307;
  }
  spv_const_binary_t binary = {&magicNumber, 1};
  spv_endianness_t endian;
  ASSERT_EQ(SPV_SUCCESS, spvBinaryEndianness(&binary, &endian));
  ASSERT_EQ(SPV_ENDIANNESS_BIG, endian);
}

}  // namespace
}  // namespace spvtools
