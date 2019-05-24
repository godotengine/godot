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

#include "gmock/gmock.h"
#include "source/operand.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using ::testing::Eq;

TEST(OperandPattern, InitiallyEmpty) {
  spv_operand_pattern_t empty;
  EXPECT_THAT(empty, Eq(spv_operand_pattern_t{}));
  EXPECT_EQ(0u, empty.size());
  EXPECT_TRUE(empty.empty());
}

TEST(OperandPattern, PushBacksAreOnTheRight) {
  spv_operand_pattern_t pattern;

  pattern.push_back(SPV_OPERAND_TYPE_ID);
  EXPECT_THAT(pattern, Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_ID}));
  EXPECT_EQ(1u, pattern.size());
  EXPECT_TRUE(!pattern.empty());
  EXPECT_EQ(SPV_OPERAND_TYPE_ID, pattern.back());

  pattern.push_back(SPV_OPERAND_TYPE_NONE);
  EXPECT_THAT(pattern, Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_ID,
                                                SPV_OPERAND_TYPE_NONE}));
  EXPECT_EQ(2u, pattern.size());
  EXPECT_TRUE(!pattern.empty());
  EXPECT_EQ(SPV_OPERAND_TYPE_NONE, pattern.back());
}

TEST(OperandPattern, PopBacksAreOnTheRight) {
  spv_operand_pattern_t pattern{SPV_OPERAND_TYPE_ID,
                                SPV_OPERAND_TYPE_LITERAL_INTEGER};

  pattern.pop_back();
  EXPECT_THAT(pattern, Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_ID}));

  pattern.pop_back();
  EXPECT_THAT(pattern, Eq(spv_operand_pattern_t{}));
}

// A test case for typed mask expansion
struct MaskExpansionCase {
  spv_operand_type_t type;
  uint32_t mask;
  spv_operand_pattern_t initial;
  spv_operand_pattern_t expected;
};

using MaskExpansionTest = ::testing::TestWithParam<MaskExpansionCase>;

TEST_P(MaskExpansionTest, Sample) {
  spv_operand_table operandTable = nullptr;
  auto env = SPV_ENV_UNIVERSAL_1_0;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable, env));

  spv_operand_pattern_t pattern(GetParam().initial);
  spvPushOperandTypesForMask(env, operandTable, GetParam().type,
                             GetParam().mask, &pattern);
  EXPECT_THAT(pattern, Eq(GetParam().expected));
}

// These macros let us write non-trivial examples without too much text.
#define PREFIX0 SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_NONE
#define PREFIX1                                                         \
  SPV_OPERAND_TYPE_STORAGE_CLASS, SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE, \
      SPV_OPERAND_TYPE_ID
INSTANTIATE_TEST_SUITE_P(
    OperandPattern, MaskExpansionTest,
    ::testing::ValuesIn(std::vector<MaskExpansionCase>{
        // No bits means no change.
        {SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS, 0, {PREFIX0}, {PREFIX0}},
        // Unknown bits means no change.  Use all bits that aren't in the
        // grammar.
        // The last mask enum is 0x20
        {SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS,
         0xffffffc0,
         {PREFIX1},
         {PREFIX1}},
        // Volatile has no operands.
        {SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS,
         SpvMemoryAccessVolatileMask,
         {PREFIX0},
         {PREFIX0}},
        // Aligned has one literal number operand.
        {SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS,
         SpvMemoryAccessAlignedMask,
         {PREFIX1},
         {PREFIX1, SPV_OPERAND_TYPE_LITERAL_INTEGER}},
        // Volatile with Aligned still has just one literal number operand.
        {SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS,
         SpvMemoryAccessVolatileMask | SpvMemoryAccessAlignedMask,
         {PREFIX1},
         {PREFIX1, SPV_OPERAND_TYPE_LITERAL_INTEGER}},
    }));
#undef PREFIX0
#undef PREFIX1

// Returns a vector of all operand types that can be used in a pattern.
std::vector<spv_operand_type_t> allOperandTypes() {
  std::vector<spv_operand_type_t> result;
  for (int i = 0; i < SPV_OPERAND_TYPE_NUM_OPERAND_TYPES; i++) {
    result.push_back(spv_operand_type_t(i));
  }
  return result;
}

using MatchableOperandExpansionTest =
    ::testing::TestWithParam<spv_operand_type_t>;

TEST_P(MatchableOperandExpansionTest, MatchableOperandsDontExpand) {
  const spv_operand_type_t type = GetParam();
  if (!spvOperandIsVariable(type)) {
    spv_operand_pattern_t pattern;
    const bool did_expand = spvExpandOperandSequenceOnce(type, &pattern);
    EXPECT_FALSE(did_expand);
    EXPECT_THAT(pattern, Eq(spv_operand_pattern_t{}));
  }
}

INSTANTIATE_TEST_SUITE_P(MatchableOperandExpansion,
                         MatchableOperandExpansionTest,
                         ::testing::ValuesIn(allOperandTypes()));

using VariableOperandExpansionTest =
    ::testing::TestWithParam<spv_operand_type_t>;

TEST_P(VariableOperandExpansionTest, NonMatchableOperandsExpand) {
  const spv_operand_type_t type = GetParam();
  if (spvOperandIsVariable(type)) {
    spv_operand_pattern_t pattern;
    const bool did_expand = spvExpandOperandSequenceOnce(type, &pattern);
    EXPECT_TRUE(did_expand);
    EXPECT_FALSE(pattern.empty());
    // For the existing rules, the first expansion of a zero-or-more operand
    // type yields a matchable operand type.  This isn't strictly necessary.
    EXPECT_FALSE(spvOperandIsVariable(pattern.back()));
  }
}

INSTANTIATE_TEST_SUITE_P(NonMatchableOperandExpansion,
                         VariableOperandExpansionTest,
                         ::testing::ValuesIn(allOperandTypes()));

TEST(AlternatePatternFollowingImmediate, Empty) {
  EXPECT_THAT(spvAlternatePatternFollowingImmediate({}),
              Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV}));
}

TEST(AlternatePatternFollowingImmediate, SingleElement) {
  // Spot-check a random selection of types.
  EXPECT_THAT(spvAlternatePatternFollowingImmediate(
                  {SPV_OPERAND_TYPE_VARIABLE_ID_LITERAL_INTEGER}),
              Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV}));
  EXPECT_THAT(
      spvAlternatePatternFollowingImmediate({SPV_OPERAND_TYPE_CAPABILITY}),
      Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV}));
  EXPECT_THAT(
      spvAlternatePatternFollowingImmediate({SPV_OPERAND_TYPE_LOOP_CONTROL}),
      Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV}));
  EXPECT_THAT(spvAlternatePatternFollowingImmediate(
                  {SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER}),
              Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV}));
  EXPECT_THAT(spvAlternatePatternFollowingImmediate({SPV_OPERAND_TYPE_ID}),
              Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV}));
}

TEST(AlternatePatternFollowingImmediate, SingleResultId) {
  EXPECT_THAT(
      spvAlternatePatternFollowingImmediate({SPV_OPERAND_TYPE_RESULT_ID}),
      Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV,
                               SPV_OPERAND_TYPE_RESULT_ID}));
}

TEST(AlternatePatternFollowingImmediate, MultipleNonResultIds) {
  EXPECT_THAT(
      spvAlternatePatternFollowingImmediate(
          {SPV_OPERAND_TYPE_VARIABLE_ID_LITERAL_INTEGER,
           SPV_OPERAND_TYPE_CAPABILITY, SPV_OPERAND_TYPE_LOOP_CONTROL,
           SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER, SPV_OPERAND_TYPE_ID}),
      Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV}));
}

TEST(AlternatePatternFollowingImmediate, ResultIdFront) {
  EXPECT_THAT(spvAlternatePatternFollowingImmediate(
                  {SPV_OPERAND_TYPE_RESULT_ID, SPV_OPERAND_TYPE_ID}),
              Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV,
                                       SPV_OPERAND_TYPE_RESULT_ID,
                                       SPV_OPERAND_TYPE_OPTIONAL_CIV}));
  EXPECT_THAT(
      spvAlternatePatternFollowingImmediate({SPV_OPERAND_TYPE_RESULT_ID,
                                             SPV_OPERAND_TYPE_FP_ROUNDING_MODE,
                                             SPV_OPERAND_TYPE_ID}),
      Eq(spv_operand_pattern_t{
          SPV_OPERAND_TYPE_OPTIONAL_CIV, SPV_OPERAND_TYPE_RESULT_ID,
          SPV_OPERAND_TYPE_OPTIONAL_CIV, SPV_OPERAND_TYPE_OPTIONAL_CIV}));
  EXPECT_THAT(
      spvAlternatePatternFollowingImmediate(
          {SPV_OPERAND_TYPE_RESULT_ID, SPV_OPERAND_TYPE_DIMENSIONALITY,
           SPV_OPERAND_TYPE_LINKAGE_TYPE,
           SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE,
           SPV_OPERAND_TYPE_FP_ROUNDING_MODE, SPV_OPERAND_TYPE_ID,
           SPV_OPERAND_TYPE_VARIABLE_ID}),
      Eq(spv_operand_pattern_t{
          SPV_OPERAND_TYPE_OPTIONAL_CIV, SPV_OPERAND_TYPE_RESULT_ID,
          SPV_OPERAND_TYPE_OPTIONAL_CIV, SPV_OPERAND_TYPE_OPTIONAL_CIV,
          SPV_OPERAND_TYPE_OPTIONAL_CIV, SPV_OPERAND_TYPE_OPTIONAL_CIV,
          SPV_OPERAND_TYPE_OPTIONAL_CIV, SPV_OPERAND_TYPE_OPTIONAL_CIV}));
}

TEST(AlternatePatternFollowingImmediate, ResultIdMiddle) {
  EXPECT_THAT(spvAlternatePatternFollowingImmediate(
                  {SPV_OPERAND_TYPE_FP_ROUNDING_MODE,
                   SPV_OPERAND_TYPE_RESULT_ID, SPV_OPERAND_TYPE_ID}),
              Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV,
                                       SPV_OPERAND_TYPE_RESULT_ID,
                                       SPV_OPERAND_TYPE_OPTIONAL_CIV}));
  EXPECT_THAT(
      spvAlternatePatternFollowingImmediate(
          {SPV_OPERAND_TYPE_DIMENSIONALITY, SPV_OPERAND_TYPE_LINKAGE_TYPE,
           SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE,
           SPV_OPERAND_TYPE_RESULT_ID, SPV_OPERAND_TYPE_FP_ROUNDING_MODE,
           SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_VARIABLE_ID}),
      Eq(spv_operand_pattern_t{
          SPV_OPERAND_TYPE_OPTIONAL_CIV, SPV_OPERAND_TYPE_RESULT_ID,
          SPV_OPERAND_TYPE_OPTIONAL_CIV, SPV_OPERAND_TYPE_OPTIONAL_CIV,
          SPV_OPERAND_TYPE_OPTIONAL_CIV}));
}

TEST(AlternatePatternFollowingImmediate, ResultIdBack) {
  EXPECT_THAT(spvAlternatePatternFollowingImmediate(
                  {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_RESULT_ID}),
              Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV,
                                       SPV_OPERAND_TYPE_RESULT_ID}));
  EXPECT_THAT(spvAlternatePatternFollowingImmediate(
                  {SPV_OPERAND_TYPE_FP_ROUNDING_MODE, SPV_OPERAND_TYPE_ID,
                   SPV_OPERAND_TYPE_RESULT_ID}),
              Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV,
                                       SPV_OPERAND_TYPE_RESULT_ID}));
  EXPECT_THAT(
      spvAlternatePatternFollowingImmediate(
          {SPV_OPERAND_TYPE_DIMENSIONALITY, SPV_OPERAND_TYPE_LINKAGE_TYPE,
           SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE,
           SPV_OPERAND_TYPE_FP_ROUNDING_MODE, SPV_OPERAND_TYPE_ID,
           SPV_OPERAND_TYPE_VARIABLE_ID, SPV_OPERAND_TYPE_RESULT_ID}),
      Eq(spv_operand_pattern_t{SPV_OPERAND_TYPE_OPTIONAL_CIV,
                               SPV_OPERAND_TYPE_RESULT_ID}));
}

}  // namespace
}  // namespace spvtools
