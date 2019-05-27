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

#include <vector>

#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using GetTargetTest = ::testing::TestWithParam<spv_target_env>;
using ::testing::ValuesIn;

TEST_P(GetTargetTest, Default) {
  spv_operand_table table;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&table, GetParam()));
  ASSERT_NE(0u, table->count);
  ASSERT_NE(nullptr, table->types);
}

TEST_P(GetTargetTest, InvalidPointerTable) {
  ASSERT_EQ(SPV_ERROR_INVALID_POINTER, spvOperandTableGet(nullptr, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(OperandTableGet, GetTargetTest,
                         ValuesIn(std::vector<spv_target_env>{
                             SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                             SPV_ENV_VULKAN_1_0}));

TEST(OperandString, AllAreDefinedExceptVariable) {
  // None has no string, so don't test it.
  EXPECT_EQ(0u, SPV_OPERAND_TYPE_NONE);
  // Start testing at enum with value 1, skipping None.
  for (int i = 1; i < int(SPV_OPERAND_TYPE_FIRST_VARIABLE_TYPE); i++) {
    EXPECT_NE(nullptr, spvOperandTypeStr(static_cast<spv_operand_type_t>(i)))
        << " Operand type " << i;
  }
}

TEST(OperandIsConcreteMask, Sample) {
  // Check a few operand types preceding the concrete mask types.
  EXPECT_FALSE(spvOperandIsConcreteMask(SPV_OPERAND_TYPE_NONE));
  EXPECT_FALSE(spvOperandIsConcreteMask(SPV_OPERAND_TYPE_ID));
  EXPECT_FALSE(spvOperandIsConcreteMask(SPV_OPERAND_TYPE_LITERAL_INTEGER));
  EXPECT_FALSE(spvOperandIsConcreteMask(SPV_OPERAND_TYPE_CAPABILITY));

  // Check all the concrete mask operand types.
  EXPECT_TRUE(spvOperandIsConcreteMask(SPV_OPERAND_TYPE_IMAGE));
  EXPECT_TRUE(spvOperandIsConcreteMask(SPV_OPERAND_TYPE_FP_FAST_MATH_MODE));
  EXPECT_TRUE(spvOperandIsConcreteMask(SPV_OPERAND_TYPE_SELECTION_CONTROL));
  EXPECT_TRUE(spvOperandIsConcreteMask(SPV_OPERAND_TYPE_LOOP_CONTROL));
  EXPECT_TRUE(spvOperandIsConcreteMask(SPV_OPERAND_TYPE_FUNCTION_CONTROL));
  EXPECT_TRUE(spvOperandIsConcreteMask(SPV_OPERAND_TYPE_MEMORY_ACCESS));

  // Check a few operand types after the concrete mask types, including the
  // optional forms for Image and MemoryAccess.
  EXPECT_FALSE(spvOperandIsConcreteMask(SPV_OPERAND_TYPE_OPTIONAL_ID));
  EXPECT_FALSE(spvOperandIsConcreteMask(SPV_OPERAND_TYPE_OPTIONAL_IMAGE));
  EXPECT_FALSE(
      spvOperandIsConcreteMask(SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS));
}

}  // namespace
}  // namespace spvtools
