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

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "source/spirv_constant.h"
#include "source/util/bitutils.h"
#include "source/util/hex_float.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::AutoText;
using spvtest::Concatenate;
using spvtest::MakeInstruction;
using spvtest::ScopedContext;
using spvtest::TextToBinaryTest;
using testing::Eq;
using testing::IsNull;
using testing::NotNull;

// An mask parsing test case.
struct MaskCase {
  spv_operand_type_t which_enum;
  uint32_t expected_value;
  const char* expression;
};

using GoodMaskParseTest = ::testing::TestWithParam<MaskCase>;

TEST_P(GoodMaskParseTest, GoodMaskExpressions) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_0);

  uint32_t value;
  EXPECT_EQ(SPV_SUCCESS,
            AssemblyGrammar(context).parseMaskOperand(
                GetParam().which_enum, GetParam().expression, &value));
  EXPECT_EQ(GetParam().expected_value, value);

  spvContextDestroy(context);
}

INSTANTIATE_TEST_SUITE_P(
    ParseMask, GoodMaskParseTest,
    ::testing::ValuesIn(std::vector<MaskCase>{
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 0, "None"},
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 1, "NotNaN"},
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 2, "NotInf"},
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 3, "NotNaN|NotInf"},
        // Mask experssions are symmetric.
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 3, "NotInf|NotNaN"},
        // Repeating a value has no effect.
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 3, "NotInf|NotNaN|NotInf"},
        // Using 3 operands still works.
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 0x13, "NotInf|NotNaN|Fast"},
        {SPV_OPERAND_TYPE_SELECTION_CONTROL, 0, "None"},
        {SPV_OPERAND_TYPE_SELECTION_CONTROL, 1, "Flatten"},
        {SPV_OPERAND_TYPE_SELECTION_CONTROL, 2, "DontFlatten"},
        // Weirdly, you can specify to flatten and don't flatten a selection.
        {SPV_OPERAND_TYPE_SELECTION_CONTROL, 3, "Flatten|DontFlatten"},
        {SPV_OPERAND_TYPE_LOOP_CONTROL, 0, "None"},
        {SPV_OPERAND_TYPE_LOOP_CONTROL, 1, "Unroll"},
        {SPV_OPERAND_TYPE_LOOP_CONTROL, 2, "DontUnroll"},
        // Weirdly, you can specify to unroll and don't unroll a loop.
        {SPV_OPERAND_TYPE_LOOP_CONTROL, 3, "Unroll|DontUnroll"},
        {SPV_OPERAND_TYPE_FUNCTION_CONTROL, 0, "None"},
        {SPV_OPERAND_TYPE_FUNCTION_CONTROL, 1, "Inline"},
        {SPV_OPERAND_TYPE_FUNCTION_CONTROL, 2, "DontInline"},
        {SPV_OPERAND_TYPE_FUNCTION_CONTROL, 4, "Pure"},
        {SPV_OPERAND_TYPE_FUNCTION_CONTROL, 8, "Const"},
        {SPV_OPERAND_TYPE_FUNCTION_CONTROL, 0xd, "Inline|Const|Pure"},
    }));

using BadFPFastMathMaskParseTest = ::testing::TestWithParam<const char*>;

TEST_P(BadFPFastMathMaskParseTest, BadMaskExpressions) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_0);

  uint32_t value;
  EXPECT_NE(SPV_SUCCESS,
            AssemblyGrammar(context).parseMaskOperand(
                SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, GetParam(), &value));

  spvContextDestroy(context);
}

INSTANTIATE_TEST_SUITE_P(ParseMask, BadFPFastMathMaskParseTest,
                         ::testing::ValuesIn(std::vector<const char*>{
                             nullptr, "", "NotValidEnum", "|", "NotInf|",
                             "|NotInf", "NotInf||NotNaN",
                             "Unroll"  // A good word, but for the wrong enum
                         }));

TEST_F(TextToBinaryTest, InvalidText) {
  ASSERT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(ScopedContext().context, nullptr, 0, &binary,
                            &diagnostic));
  EXPECT_NE(nullptr, diagnostic);
  EXPECT_THAT(diagnostic->error, Eq(std::string("Missing assembly text.")));
}

TEST_F(TextToBinaryTest, InvalidPointer) {
  SetText(
      "OpEntryPoint Kernel 0 \"\"\nOpExecutionMode 0 LocalSizeHint 1 1 1\n");
  ASSERT_EQ(SPV_ERROR_INVALID_POINTER,
            spvTextToBinary(ScopedContext().context, text.str, text.length,
                            nullptr, &diagnostic));
}

TEST_F(TextToBinaryTest, InvalidPrefix) {
  EXPECT_EQ(
      "Expected <opcode> or <result-id> at the beginning of an instruction, "
      "found 'Invalid'.",
      CompileFailure("Invalid"));
}

TEST_F(TextToBinaryTest, EmptyAssemblyString) {
  // An empty assembly module is valid!
  // It should produce a valid module with zero instructions.
  EXPECT_THAT(CompiledInstructions(""), Eq(std::vector<uint32_t>{}));
}

TEST_F(TextToBinaryTest, StringSpace) {
  const std::string code = ("OpSourceExtension \"string with spaces\"\n");
  EXPECT_EQ(code, EncodeAndDecodeSuccessfully(code));
}

TEST_F(TextToBinaryTest, UnknownBeginningOfInstruction) {
  EXPECT_EQ(
      "Expected <opcode> or <result-id> at the beginning of an instruction, "
      "found 'Google'.",
      CompileFailure(
          "\nOpSource OpenCL_C 12\nOpMemoryModel Physical64 OpenCL\nGoogle\n"));
  EXPECT_EQ(4u, diagnostic->position.line + 1);
  EXPECT_EQ(1u, diagnostic->position.column + 1);
}

TEST_F(TextToBinaryTest, NoEqualSign) {
  EXPECT_EQ("Expected '=', found end of stream.",
            CompileFailure("\nOpSource OpenCL_C 12\n"
                           "OpMemoryModel Physical64 OpenCL\n%2\n"));
  EXPECT_EQ(5u, diagnostic->position.line + 1);
  EXPECT_EQ(1u, diagnostic->position.column + 1);
}

TEST_F(TextToBinaryTest, NoOpCode) {
  EXPECT_EQ("Expected opcode, found end of stream.",
            CompileFailure("\nOpSource OpenCL_C 12\n"
                           "OpMemoryModel Physical64 OpenCL\n%2 =\n"));
  EXPECT_EQ(5u, diagnostic->position.line + 1);
  EXPECT_EQ(1u, diagnostic->position.column + 1);
}

TEST_F(TextToBinaryTest, WrongOpCode) {
  EXPECT_EQ("Invalid Opcode prefix 'Wahahaha'.",
            CompileFailure("\nOpSource OpenCL_C 12\n"
                           "OpMemoryModel Physical64 OpenCL\n%2 = Wahahaha\n"));
  EXPECT_EQ(4u, diagnostic->position.line + 1);
  EXPECT_EQ(6u, diagnostic->position.column + 1);
}

TEST_F(TextToBinaryTest, CRLF) {
  const std::string input =
      "%i32 = OpTypeInt 32 1\r\n%c = OpConstant %i32 123\r\n";
  EXPECT_THAT(CompiledInstructions(input),
              Eq(Concatenate({MakeInstruction(SpvOpTypeInt, {1, 32, 1}),
                              MakeInstruction(SpvOpConstant, {1, 2, 123})})));
}

using TextToBinaryFloatValueTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<std::pair<std::string, uint32_t>>>;

TEST_P(TextToBinaryFloatValueTest, Samples) {
  const std::string input =
      "%1 = OpTypeFloat 32\n%2 = OpConstant %1 " + GetParam().first;
  EXPECT_THAT(CompiledInstructions(input),
              Eq(Concatenate({MakeInstruction(SpvOpTypeFloat, {1, 32}),
                              MakeInstruction(SpvOpConstant,
                                              {1, 2, GetParam().second})})));
}

INSTANTIATE_TEST_SUITE_P(
    FloatValues, TextToBinaryFloatValueTest,
    ::testing::ValuesIn(std::vector<std::pair<std::string, uint32_t>>{
        {"0.0", 0x00000000},          // +0
        {"!0x00000001", 0x00000001},  // +denorm
        {"!0x00800000", 0x00800000},  // +norm
        {"1.5", 0x3fc00000},
        {"!0x7f800000", 0x7f800000},  // +inf
        {"!0x7f800001", 0x7f800001},  // NaN

        {"-0.0", 0x80000000},         // -0
        {"!0x80000001", 0x80000001},  // -denorm
        {"!0x80800000", 0x80800000},  // -norm
        {"-2.5", 0xc0200000},
        {"!0xff800000", 0xff800000},  // -inf
        {"!0xff800001", 0xff800001},  // NaN
    }));

using TextToBinaryHalfValueTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<std::pair<std::string, uint32_t>>>;

TEST_P(TextToBinaryHalfValueTest, Samples) {
  const std::string input =
      "%1 = OpTypeFloat 16\n%2 = OpConstant %1 " + GetParam().first;
  EXPECT_THAT(CompiledInstructions(input),
              Eq(Concatenate({MakeInstruction(SpvOpTypeFloat, {1, 16}),
                              MakeInstruction(SpvOpConstant,
                                              {1, 2, GetParam().second})})));
}

INSTANTIATE_TEST_SUITE_P(
    HalfValues, TextToBinaryHalfValueTest,
    ::testing::ValuesIn(std::vector<std::pair<std::string, uint32_t>>{
        {"0.0", 0x00000000},
        {"1.0", 0x00003c00},
        {"1.000844", 0x00003c00},  // Truncate to 1.0
        {"1.000977", 0x00003c01},  // Don't have to truncate
        {"1.001465", 0x00003c01},  // Truncate to 1.0000977
        {"1.5", 0x00003e00},
        {"-1.0", 0x0000bc00},
        {"2.0", 0x00004000},
        {"-2.0", 0x0000c000},
        {"0x1p1", 0x00004000},
        {"-0x1p1", 0x0000c000},
        {"0x1.8p1", 0x00004200},
        {"0x1.8p4", 0x00004e00},
        {"0x1.801p4", 0x00004e00},
        {"0x1.804p4", 0x00004e01},
    }));

TEST(CreateContext, InvalidEnvironment) {
  spv_target_env env;
  std::memset(&env, 99, sizeof(env));
  EXPECT_THAT(spvContextCreate(env), IsNull());
}

TEST(CreateContext, UniversalEnvironment) {
  auto c = spvContextCreate(SPV_ENV_UNIVERSAL_1_0);
  EXPECT_THAT(c, NotNull());
  spvContextDestroy(c);
}

TEST(CreateContext, VulkanEnvironment) {
  auto c = spvContextCreate(SPV_ENV_VULKAN_1_0);
  EXPECT_THAT(c, NotNull());
  spvContextDestroy(c);
}

}  // namespace
}  // namespace spvtools
