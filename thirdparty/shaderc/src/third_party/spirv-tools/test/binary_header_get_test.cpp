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

#include "source/spirv_constant.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

class BinaryHeaderGet : public ::testing::Test {
 public:
  BinaryHeaderGet() { memset(code, 0, sizeof(code)); }

  virtual void SetUp() {
    code[0] = SpvMagicNumber;
    code[1] = SpvVersion;
    code[2] = SPV_GENERATOR_CODEPLAY;
    code[3] = 1;  // NOTE: Bound
    code[4] = 0;  // NOTE: Schema; reserved
    code[5] = 0;  // NOTE: Instructions

    binary.code = code;
    binary.wordCount = 6;
  }
  spv_const_binary_t get_const_binary() {
    return spv_const_binary_t{binary.code, binary.wordCount};
  }
  virtual void TearDown() {}

  uint32_t code[6];
  spv_binary_t binary;
};

TEST_F(BinaryHeaderGet, Default) {
  spv_endianness_t endian;
  spv_const_binary_t const_bin = get_const_binary();
  ASSERT_EQ(SPV_SUCCESS, spvBinaryEndianness(&const_bin, &endian));

  spv_header_t header;
  ASSERT_EQ(SPV_SUCCESS, spvBinaryHeaderGet(&const_bin, endian, &header));

  ASSERT_EQ(static_cast<uint32_t>(SpvMagicNumber), header.magic);
  ASSERT_EQ(0x00010300u, header.version);
  ASSERT_EQ(static_cast<uint32_t>(SPV_GENERATOR_CODEPLAY), header.generator);
  ASSERT_EQ(1u, header.bound);
  ASSERT_EQ(0u, header.schema);
  ASSERT_EQ(&code[5], header.instructions);
}

TEST_F(BinaryHeaderGet, InvalidCode) {
  spv_const_binary_t my_binary = {nullptr, 0};
  spv_header_t header;
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY,
            spvBinaryHeaderGet(&my_binary, SPV_ENDIANNESS_LITTLE, &header));
}

TEST_F(BinaryHeaderGet, InvalidPointerHeader) {
  spv_const_binary_t const_bin = get_const_binary();
  ASSERT_EQ(SPV_ERROR_INVALID_POINTER,
            spvBinaryHeaderGet(&const_bin, SPV_ENDIANNESS_LITTLE, nullptr));
}

TEST_F(BinaryHeaderGet, TruncatedHeader) {
  for (uint8_t i = 1; i < SPV_INDEX_INSTRUCTION; i++) {
    binary.wordCount = i;
    spv_const_binary_t const_bin = get_const_binary();
    ASSERT_EQ(SPV_ERROR_INVALID_BINARY,
              spvBinaryHeaderGet(&const_bin, SPV_ENDIANNESS_LITTLE, nullptr));
  }
}

}  // namespace
}  // namespace spvtools
