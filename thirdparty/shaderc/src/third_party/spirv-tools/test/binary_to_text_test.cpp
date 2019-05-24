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

#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "source/spirv_constant.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::AutoText;
using spvtest::ScopedContext;
using spvtest::TextToBinaryTest;
using ::testing::Combine;
using ::testing::Eq;
using ::testing::HasSubstr;

class BinaryToText : public ::testing::Test {
 public:
  BinaryToText()
      : context(spvContextCreate(SPV_ENV_UNIVERSAL_1_0)), binary(nullptr) {}
  ~BinaryToText() {
    spvBinaryDestroy(binary);
    spvContextDestroy(context);
  }

  virtual void SetUp() {
    const char* textStr = R"(
      OpSource OpenCL_C 12
      OpMemoryModel Physical64 OpenCL
      OpSourceExtension "PlaceholderExtensionName"
      OpEntryPoint Kernel %1 "foo"
      OpExecutionMode %1 LocalSizeHint 1 1 1
 %2 = OpTypeVoid
 %3 = OpTypeBool
 %4 = OpTypeInt 8 0
 %5 = OpTypeInt 8 1
 %6 = OpTypeInt 16 0
 %7 = OpTypeInt 16 1
 %8 = OpTypeInt 32 0
 %9 = OpTypeInt 32 1
%10 = OpTypeInt 64 0
%11 = OpTypeInt 64 1
%12 = OpTypeFloat 16
%13 = OpTypeFloat 32
%14 = OpTypeFloat 64
%15 = OpTypeVector %4 2
)";
    spv_text_t text = {textStr, strlen(textStr)};
    spv_diagnostic diagnostic = nullptr;
    spv_result_t error =
        spvTextToBinary(context, text.str, text.length, &binary, &diagnostic);
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_EQ(SPV_SUCCESS, error);
  }

  virtual void TearDown() {
    spvBinaryDestroy(binary);
    binary = nullptr;
  }

  // Compiles the given assembly text, and saves it into 'binary'.
  void CompileSuccessfully(std::string text) {
    spvBinaryDestroy(binary);
    binary = nullptr;
    spv_diagnostic diagnostic = nullptr;
    EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(context, text.c_str(), text.size(),
                                           &binary, &diagnostic));
  }

  spv_context context;
  spv_binary binary;
};

TEST_F(BinaryToText, Default) {
  spv_text text = nullptr;
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(
      SPV_SUCCESS,
      spvBinaryToText(context, binary->code, binary->wordCount,
                      SPV_BINARY_TO_TEXT_OPTION_NONE, &text, &diagnostic));
  printf("%s", text->str);
  spvTextDestroy(text);
}

TEST_F(BinaryToText, MissingModule) {
  spv_text text;
  spv_diagnostic diagnostic = nullptr;
  EXPECT_EQ(
      SPV_ERROR_INVALID_BINARY,
      spvBinaryToText(context, nullptr, 42, SPV_BINARY_TO_TEXT_OPTION_NONE,
                      &text, &diagnostic));
  EXPECT_THAT(diagnostic->error, Eq(std::string("Missing module.")));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
}

TEST_F(BinaryToText, TruncatedModule) {
  // Make a valid module with zero instructions.
  CompileSuccessfully("");
  EXPECT_EQ(SPV_INDEX_INSTRUCTION, binary->wordCount);

  for (size_t length = 0; length < SPV_INDEX_INSTRUCTION; length++) {
    spv_text text = nullptr;
    spv_diagnostic diagnostic = nullptr;
    EXPECT_EQ(
        SPV_ERROR_INVALID_BINARY,
        spvBinaryToText(context, binary->code, length,
                        SPV_BINARY_TO_TEXT_OPTION_NONE, &text, &diagnostic));
    ASSERT_NE(nullptr, diagnostic);
    std::stringstream expected;
    expected << "Module has incomplete header: only " << length
             << " words instead of " << SPV_INDEX_INSTRUCTION;
    EXPECT_THAT(diagnostic->error, Eq(expected.str()));
    spvDiagnosticDestroy(diagnostic);
  }
}

TEST_F(BinaryToText, InvalidMagicNumber) {
  CompileSuccessfully("");
  std::vector<uint32_t> damaged_binary(binary->code,
                                       binary->code + binary->wordCount);
  damaged_binary[SPV_INDEX_MAGIC_NUMBER] ^= 123;

  spv_diagnostic diagnostic = nullptr;
  spv_text text;
  EXPECT_EQ(
      SPV_ERROR_INVALID_BINARY,
      spvBinaryToText(context, damaged_binary.data(), damaged_binary.size(),
                      SPV_BINARY_TO_TEXT_OPTION_NONE, &text, &diagnostic));
  ASSERT_NE(nullptr, diagnostic);
  std::stringstream expected;
  expected << "Invalid SPIR-V magic number '" << std::hex
           << damaged_binary[SPV_INDEX_MAGIC_NUMBER] << "'.";
  EXPECT_THAT(diagnostic->error, Eq(expected.str()));
  spvDiagnosticDestroy(diagnostic);
}

struct FailedDecodeCase {
  std::string source_text;
  std::vector<uint32_t> appended_instruction;
  std::string expected_error_message;
};

using BinaryToTextFail =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<FailedDecodeCase>>;

TEST_P(BinaryToTextFail, EncodeSuccessfullyDecodeFailed) {
  EXPECT_THAT(EncodeSuccessfullyDecodeFailed(GetParam().source_text,
                                             GetParam().appended_instruction),
              Eq(GetParam().expected_error_message));
}

INSTANTIATE_TEST_SUITE_P(
    InvalidIds, BinaryToTextFail,
    ::testing::ValuesIn(std::vector<FailedDecodeCase>{
        {"", spvtest::MakeInstruction(SpvOpTypeVoid, {0}),
         "Error: Result Id is 0"},
        {"", spvtest::MakeInstruction(SpvOpConstant, {0, 1, 42}),
         "Error: Type Id is 0"},
        {"%1 = OpTypeVoid", spvtest::MakeInstruction(SpvOpTypeVoid, {1}),
         "Id 1 is defined more than once"},
        {"%1 = OpTypeVoid\n"
         "%2 = OpNot %1 %foo",
         spvtest::MakeInstruction(SpvOpNot, {1, 2, 3}),
         "Id 2 is defined more than once"},
        {"%1 = OpTypeVoid\n"
         "%2 = OpNot %1 %foo",
         spvtest::MakeInstruction(SpvOpNot, {1, 1, 3}),
         "Id 1 is defined more than once"},
        // The following are the two failure cases for
        // Parser::setNumericTypeInfoForType.
        {"", spvtest::MakeInstruction(SpvOpConstant, {500, 1, 42}),
         "Type Id 500 is not a type"},
        {"%1 = OpTypeInt 32 0\n"
         "%2 = OpTypeVector %1 4",
         spvtest::MakeInstruction(SpvOpConstant, {2, 3, 999}),
         "Type Id 2 is not a scalar numeric type"},
    }));

INSTANTIATE_TEST_SUITE_P(
    InvalidIdsCheckedDuringLiteralCaseParsing, BinaryToTextFail,
    ::testing::ValuesIn(std::vector<FailedDecodeCase>{
        {"", spvtest::MakeInstruction(SpvOpSwitch, {1, 2, 3, 4}),
         "Invalid OpSwitch: selector id 1 has no type"},
        {"%1 = OpTypeVoid\n",
         spvtest::MakeInstruction(SpvOpSwitch, {1, 2, 3, 4}),
         "Invalid OpSwitch: selector id 1 is a type, not a value"},
        {"%1 = OpConstantTrue !500",
         spvtest::MakeInstruction(SpvOpSwitch, {1, 2, 3, 4}),
         "Type Id 500 is not a type"},
        {"%1 = OpTypeFloat 32\n%2 = OpConstant %1 1.5",
         spvtest::MakeInstruction(SpvOpSwitch, {2, 3, 4, 5}),
         "Invalid OpSwitch: selector id 2 is not a scalar integer"},
    }));

TEST_F(TextToBinaryTest, OneInstruction) {
  const std::string input = "OpSource OpenCL_C 12\n";
  EXPECT_EQ(input, EncodeAndDecodeSuccessfully(input));
}

// Exercise the case where an operand itself has operands.
// This could detect problems in updating the expected-set-of-operands
// list.
TEST_F(TextToBinaryTest, OperandWithOperands) {
  const std::string input = R"(OpEntryPoint Kernel %1 "foo"
OpExecutionMode %1 LocalSizeHint 100 200 300
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%1 = OpFunction %1 None %3
)";
  EXPECT_EQ(input, EncodeAndDecodeSuccessfully(input));
}

using RoundTripInstructionsTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<std::tuple<spv_target_env, std::string>>>;

TEST_P(RoundTripInstructionsTest, Sample) {
  EXPECT_THAT(EncodeAndDecodeSuccessfully(std::get<1>(GetParam()),
                                          SPV_BINARY_TO_TEXT_OPTION_NONE,
                                          std::get<0>(GetParam())),
              Eq(std::get<1>(GetParam())));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    NumericLiterals, RoundTripInstructionsTest,
    // This test is independent of environment, so just test the one.
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                              SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3),
            ::testing::ValuesIn(std::vector<std::string>{
                "%1 = OpTypeInt 12 0\n%2 = OpConstant %1 1867\n",
                "%1 = OpTypeInt 12 1\n%2 = OpConstant %1 1867\n",
                "%1 = OpTypeInt 12 1\n%2 = OpConstant %1 -1867\n",
                "%1 = OpTypeInt 32 0\n%2 = OpConstant %1 1867\n",
                "%1 = OpTypeInt 32 1\n%2 = OpConstant %1 1867\n",
                "%1 = OpTypeInt 32 1\n%2 = OpConstant %1 -1867\n",
                "%1 = OpTypeInt 64 0\n%2 = OpConstant %1 18446744073709551615\n",
                "%1 = OpTypeInt 64 1\n%2 = OpConstant %1 9223372036854775807\n",
                "%1 = OpTypeInt 64 1\n%2 = OpConstant %1 -9223372036854775808\n",
                // 16-bit floats print as hex floats.
                "%1 = OpTypeFloat 16\n%2 = OpConstant %1 0x1.ff4p+16\n",
                "%1 = OpTypeFloat 16\n%2 = OpConstant %1 -0x1.d2cp-10\n",
                // 32-bit floats
                "%1 = OpTypeFloat 32\n%2 = OpConstant %1 -3.125\n",
                "%1 = OpTypeFloat 32\n%2 = OpConstant %1 0x1.8p+128\n", // NaN
                "%1 = OpTypeFloat 32\n%2 = OpConstant %1 -0x1.0002p+128\n", // NaN
                "%1 = OpTypeFloat 32\n%2 = OpConstant %1 0x1p+128\n", // Inf
                "%1 = OpTypeFloat 32\n%2 = OpConstant %1 -0x1p+128\n", // -Inf
                // 64-bit floats
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 -3.125\n",
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 0x1.ffffffffffffap-1023\n", // small normal
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 -0x1.ffffffffffffap-1023\n",
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 0x1.8p+1024\n", // NaN
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 -0x1.0002p+1024\n", // NaN
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 0x1p+1024\n", // Inf
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 -0x1p+1024\n", // -Inf
            })));
// clang-format on

INSTANTIATE_TEST_SUITE_P(
    MemoryAccessMasks, RoundTripInstructionsTest,
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                              SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3),
            ::testing::ValuesIn(std::vector<std::string>{
                "OpStore %1 %2\n",       // 3 words long.
                "OpStore %1 %2 None\n",  // 4 words long, explicit final 0.
                "OpStore %1 %2 Volatile\n",
                "OpStore %1 %2 Aligned 8\n",
                "OpStore %1 %2 Nontemporal\n",
                // Combinations show the names from LSB to MSB
                "OpStore %1 %2 Volatile|Aligned 16\n",
                "OpStore %1 %2 Volatile|Nontemporal\n",
                "OpStore %1 %2 Volatile|Aligned|Nontemporal 32\n",
            })));

INSTANTIATE_TEST_SUITE_P(
    FPFastMathModeMasks, RoundTripInstructionsTest,
    Combine(
        ::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                          SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3),
        ::testing::ValuesIn(std::vector<std::string>{
            "OpDecorate %1 FPFastMathMode None\n",
            "OpDecorate %1 FPFastMathMode NotNaN\n",
            "OpDecorate %1 FPFastMathMode NotInf\n",
            "OpDecorate %1 FPFastMathMode NSZ\n",
            "OpDecorate %1 FPFastMathMode AllowRecip\n",
            "OpDecorate %1 FPFastMathMode Fast\n",
            // Combinations show the names from LSB to MSB
            "OpDecorate %1 FPFastMathMode NotNaN|NotInf\n",
            "OpDecorate %1 FPFastMathMode NSZ|AllowRecip\n",
            "OpDecorate %1 FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|Fast\n",
        })));

INSTANTIATE_TEST_SUITE_P(
    LoopControlMasks, RoundTripInstructionsTest,
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                              SPV_ENV_UNIVERSAL_1_3, SPV_ENV_UNIVERSAL_1_2),
            ::testing::ValuesIn(std::vector<std::string>{
                "OpLoopMerge %1 %2 None\n",
                "OpLoopMerge %1 %2 Unroll\n",
                "OpLoopMerge %1 %2 DontUnroll\n",
                "OpLoopMerge %1 %2 Unroll|DontUnroll\n",
            })));

INSTANTIATE_TEST_SUITE_P(LoopControlMasksV11, RoundTripInstructionsTest,
                         Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_1,
                                                   SPV_ENV_UNIVERSAL_1_2,
                                                   SPV_ENV_UNIVERSAL_1_3),
                                 ::testing::ValuesIn(std::vector<std::string>{
                                     "OpLoopMerge %1 %2 DependencyInfinite\n",
                                     "OpLoopMerge %1 %2 DependencyLength 8\n",
                                 })));

INSTANTIATE_TEST_SUITE_P(
    SelectionControlMasks, RoundTripInstructionsTest,
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                              SPV_ENV_UNIVERSAL_1_3, SPV_ENV_UNIVERSAL_1_2),
            ::testing::ValuesIn(std::vector<std::string>{
                "OpSelectionMerge %1 None\n",
                "OpSelectionMerge %1 Flatten\n",
                "OpSelectionMerge %1 DontFlatten\n",
                "OpSelectionMerge %1 Flatten|DontFlatten\n",
            })));

INSTANTIATE_TEST_SUITE_P(
    FunctionControlMasks, RoundTripInstructionsTest,
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                              SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3),
            ::testing::ValuesIn(std::vector<std::string>{
                "%2 = OpFunction %1 None %3\n",
                "%2 = OpFunction %1 Inline %3\n",
                "%2 = OpFunction %1 DontInline %3\n",
                "%2 = OpFunction %1 Pure %3\n",
                "%2 = OpFunction %1 Const %3\n",
                "%2 = OpFunction %1 Inline|Pure|Const %3\n",
                "%2 = OpFunction %1 DontInline|Const %3\n",
            })));

INSTANTIATE_TEST_SUITE_P(
    ImageMasks, RoundTripInstructionsTest,
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                              SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3),
            ::testing::ValuesIn(std::vector<std::string>{
                "%2 = OpImageFetch %1 %3 %4\n",
                "%2 = OpImageFetch %1 %3 %4 None\n",
                "%2 = OpImageFetch %1 %3 %4 Bias %5\n",
                "%2 = OpImageFetch %1 %3 %4 Lod %5\n",
                "%2 = OpImageFetch %1 %3 %4 Grad %5 %6\n",
                "%2 = OpImageFetch %1 %3 %4 ConstOffset %5\n",
                "%2 = OpImageFetch %1 %3 %4 Offset %5\n",
                "%2 = OpImageFetch %1 %3 %4 ConstOffsets %5\n",
                "%2 = OpImageFetch %1 %3 %4 Sample %5\n",
                "%2 = OpImageFetch %1 %3 %4 MinLod %5\n",
                "%2 = OpImageFetch %1 %3 %4 Bias|Lod|Grad %5 %6 %7 %8\n",
                "%2 = OpImageFetch %1 %3 %4 ConstOffset|Offset|ConstOffsets"
                " %5 %6 %7\n",
                "%2 = OpImageFetch %1 %3 %4 Sample|MinLod %5 %6\n",
                "%2 = OpImageFetch %1 %3 %4"
                " Bias|Lod|Grad|ConstOffset|Offset|ConstOffsets|Sample|MinLod"
                " %5 %6 %7 %8 %9 %10 %11 %12 %13\n"})));

INSTANTIATE_TEST_SUITE_P(
    NewInstructionsInSPIRV1_2, RoundTripInstructionsTest,
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3),
            ::testing::ValuesIn(std::vector<std::string>{
                "OpExecutionModeId %1 SubgroupsPerWorkgroupId %2\n",
                "OpExecutionModeId %1 LocalSizeId %2 %3 %4\n",
                "OpExecutionModeId %1 LocalSizeHintId %2\n",
                "OpDecorateId %1 AlignmentId %2\n",
                "OpDecorateId %1 MaxByteOffsetId %2\n",
            })));

using MaskSorting = TextToBinaryTest;

TEST_F(MaskSorting, MasksAreSortedFromLSBToMSB) {
  EXPECT_THAT(EncodeAndDecodeSuccessfully(
                  "OpStore %1 %2 Nontemporal|Aligned|Volatile 32"),
              Eq("OpStore %1 %2 Volatile|Aligned|Nontemporal 32\n"));
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully(
          "OpDecorate %1 FPFastMathMode NotInf|Fast|AllowRecip|NotNaN|NSZ"),
      Eq("OpDecorate %1 FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|Fast\n"));
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully("OpLoopMerge %1 %2 DontUnroll|Unroll"),
      Eq("OpLoopMerge %1 %2 Unroll|DontUnroll\n"));
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully("OpSelectionMerge %1 DontFlatten|Flatten"),
      Eq("OpSelectionMerge %1 Flatten|DontFlatten\n"));
  EXPECT_THAT(EncodeAndDecodeSuccessfully(
                  "%2 = OpFunction %1 DontInline|Const|Pure|Inline %3"),
              Eq("%2 = OpFunction %1 Inline|DontInline|Pure|Const %3\n"));
  EXPECT_THAT(EncodeAndDecodeSuccessfully(
                  "%2 = OpImageFetch %1 %3 %4"
                  " MinLod|Sample|Offset|Lod|Grad|ConstOffsets|ConstOffset|Bias"
                  " %5 %6 %7 %8 %9 %10 %11 %12 %13\n"),
              Eq("%2 = OpImageFetch %1 %3 %4"
                 " Bias|Lod|Grad|ConstOffset|Offset|ConstOffsets|Sample|MinLod"
                 " %5 %6 %7 %8 %9 %10 %11 %12 %13\n"));
}

using OperandTypeTest = TextToBinaryTest;

TEST_F(OperandTypeTest, OptionalTypedLiteralNumber) {
  const std::string input =
      "%1 = OpTypeInt 32 0\n"
      "%2 = OpConstant %1 42\n"
      "OpSwitch %2 %3 100 %4\n";
  EXPECT_EQ(input, EncodeAndDecodeSuccessfully(input));
}

using IndentTest = spvtest::TextToBinaryTest;

TEST_F(IndentTest, Sample) {
  const std::string input = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1 %3 %4 %5 %6 %7 %8 %9 %10 ; force IDs into double digits
%11 = OpConstant %1 42
OpStore %2 %3 Aligned|Volatile 4 ; bogus, but not indented
)";
  const std::string expected =
      R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
          %1 = OpTypeInt 32 0
          %2 = OpTypeStruct %1 %3 %4 %5 %6 %7 %8 %9 %10
         %11 = OpConstant %1 42
               OpStore %2 %3 Volatile|Aligned 4
)";
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully(input, SPV_BINARY_TO_TEXT_OPTION_INDENT),
      expected);
}

using FriendlyNameDisassemblyTest = spvtest::TextToBinaryTest;

TEST_F(FriendlyNameDisassemblyTest, Sample) {
  const std::string input = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1 %3 %4 %5 %6 %7 %8 %9 %10 ; force IDs into double digits
%11 = OpConstant %1 42
)";
  const std::string expected =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%_struct_2 = OpTypeStruct %uint %3 %4 %5 %6 %7 %8 %9 %10
%uint_42 = OpConstant %uint 42
)";
  EXPECT_THAT(EncodeAndDecodeSuccessfully(
                  input, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES),
              expected);
}

TEST_F(TextToBinaryTest, ShowByteOffsetsWhenRequested) {
  const std::string input = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeVoid
)";
  const std::string expected =
      R"(OpCapability Shader ; 0x00000014
OpMemoryModel Logical GLSL450 ; 0x0000001c
%1 = OpTypeInt 32 0 ; 0x00000028
%2 = OpTypeVoid ; 0x00000038
)";
  EXPECT_THAT(EncodeAndDecodeSuccessfully(
                  input, SPV_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET),
              expected);
}

// Test version string.
TEST_F(TextToBinaryTest, VersionString) {
  auto words = CompileSuccessfully("");
  spv_text decoded_text = nullptr;
  EXPECT_THAT(spvBinaryToText(ScopedContext().context, words.data(),
                              words.size(), SPV_BINARY_TO_TEXT_OPTION_NONE,
                              &decoded_text, &diagnostic),
              Eq(SPV_SUCCESS));
  EXPECT_EQ(nullptr, diagnostic);

  EXPECT_THAT(decoded_text->str, HasSubstr("Version: 1.0\n"))
      << EncodeAndDecodeSuccessfully("");
  spvTextDestroy(decoded_text);
}

// Test generator string.

// A test case for the generator string.  This allows us to
// test both of the 16-bit components of the generator word.
struct GeneratorStringCase {
  uint16_t generator;
  uint16_t misc;
  std::string expected;
};

using GeneratorStringTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<GeneratorStringCase>>;

TEST_P(GeneratorStringTest, Sample) {
  auto words = CompileSuccessfully("");
  EXPECT_EQ(2u, SPV_INDEX_GENERATOR_NUMBER);
  words[SPV_INDEX_GENERATOR_NUMBER] =
      SPV_GENERATOR_WORD(GetParam().generator, GetParam().misc);

  spv_text decoded_text = nullptr;
  EXPECT_THAT(spvBinaryToText(ScopedContext().context, words.data(),
                              words.size(), SPV_BINARY_TO_TEXT_OPTION_NONE,
                              &decoded_text, &diagnostic),
              Eq(SPV_SUCCESS));
  EXPECT_THAT(diagnostic, Eq(nullptr));
  EXPECT_THAT(std::string(decoded_text->str), HasSubstr(GetParam().expected));
  spvTextDestroy(decoded_text);
}

INSTANTIATE_TEST_SUITE_P(GeneratorStrings, GeneratorStringTest,
                         ::testing::ValuesIn(std::vector<GeneratorStringCase>{
                             {SPV_GENERATOR_KHRONOS, 12, "Khronos; 12"},
                             {SPV_GENERATOR_LUNARG, 99, "LunarG; 99"},
                             {SPV_GENERATOR_VALVE, 1, "Valve; 1"},
                             {SPV_GENERATOR_CODEPLAY, 65535, "Codeplay; 65535"},
                             {SPV_GENERATOR_NVIDIA, 19, "NVIDIA; 19"},
                             {SPV_GENERATOR_ARM, 1000, "ARM; 1000"},
                             {SPV_GENERATOR_KHRONOS_LLVM_TRANSLATOR, 38,
                              "Khronos LLVM/SPIR-V Translator; 38"},
                             {SPV_GENERATOR_KHRONOS_ASSEMBLER, 2,
                              "Khronos SPIR-V Tools Assembler; 2"},
                             {SPV_GENERATOR_KHRONOS_GLSLANG, 1,
                              "Khronos Glslang Reference Front End; 1"},
                             {1000, 18, "Unknown(1000); 18"},
                             {65535, 32767, "Unknown(65535); 32767"},
                         }));

// TODO(dneto): Test new instructions and enums in SPIR-V 1.3

}  // namespace
}  // namespace spvtools
