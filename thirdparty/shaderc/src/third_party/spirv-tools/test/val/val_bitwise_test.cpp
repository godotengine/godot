// Copyright (c) 2017 Google Inc.
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

// Tests for unique type declaration rules validator.

#include <string>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

using ValidateBitwise = spvtest::ValidateBase<bool>;

std::string GenerateShaderCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "") {
  const std::string capabilities =
      R"(
OpCapability Shader
OpCapability Int64
OpCapability Float64)";

  const std::string after_extension_before_body =
      R"(
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%s32 = OpTypeInt 32 1
%f64 = OpTypeFloat 64
%u64 = OpTypeInt 64 0
%s64 = OpTypeInt 64 1
%boolvec2 = OpTypeVector %bool 2
%s32vec2 = OpTypeVector %s32 2
%u32vec2 = OpTypeVector %u32 2
%u64vec2 = OpTypeVector %u64 2
%f32vec2 = OpTypeVector %f32 2
%f64vec2 = OpTypeVector %f64 2
%boolvec3 = OpTypeVector %bool 3
%u32vec3 = OpTypeVector %u32 3
%u64vec3 = OpTypeVector %u64 3
%s32vec3 = OpTypeVector %s32 3
%f32vec3 = OpTypeVector %f32 3
%f64vec3 = OpTypeVector %f64 3
%boolvec4 = OpTypeVector %bool 4
%u32vec4 = OpTypeVector %u32 4
%u64vec4 = OpTypeVector %u64 4
%s32vec4 = OpTypeVector %s32 4
%f32vec4 = OpTypeVector %f32 4
%f64vec4 = OpTypeVector %f64 4

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%f32_2 = OpConstant %f32 2
%f32_3 = OpConstant %f32 3
%f32_4 = OpConstant %f32 4

%s32_0 = OpConstant %s32 0
%s32_1 = OpConstant %s32 1
%s32_2 = OpConstant %s32 2
%s32_3 = OpConstant %s32 3
%s32_4 = OpConstant %s32 4
%s32_m1 = OpConstant %s32 -1

%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u32_2 = OpConstant %u32 2
%u32_3 = OpConstant %u32 3
%u32_4 = OpConstant %u32 4

%f64_0 = OpConstant %f64 0
%f64_1 = OpConstant %f64 1
%f64_2 = OpConstant %f64 2
%f64_3 = OpConstant %f64 3
%f64_4 = OpConstant %f64 4

%s64_0 = OpConstant %s64 0
%s64_1 = OpConstant %s64 1
%s64_2 = OpConstant %s64 2
%s64_3 = OpConstant %s64 3
%s64_4 = OpConstant %s64 4
%s64_m1 = OpConstant %s64 -1

%u64_0 = OpConstant %u64 0
%u64_1 = OpConstant %u64 1
%u64_2 = OpConstant %u64 2
%u64_3 = OpConstant %u64 3
%u64_4 = OpConstant %u64 4

%u32vec2_01 = OpConstantComposite %u32vec2 %u32_0 %u32_1
%u32vec2_12 = OpConstantComposite %u32vec2 %u32_1 %u32_2
%u32vec3_012 = OpConstantComposite %u32vec3 %u32_0 %u32_1 %u32_2
%u32vec3_123 = OpConstantComposite %u32vec3 %u32_1 %u32_2 %u32_3
%u32vec4_0123 = OpConstantComposite %u32vec4 %u32_0 %u32_1 %u32_2 %u32_3
%u32vec4_1234 = OpConstantComposite %u32vec4 %u32_1 %u32_2 %u32_3 %u32_4

%s32vec2_01 = OpConstantComposite %s32vec2 %s32_0 %s32_1
%s32vec2_12 = OpConstantComposite %s32vec2 %s32_1 %s32_2
%s32vec3_012 = OpConstantComposite %s32vec3 %s32_0 %s32_1 %s32_2
%s32vec3_123 = OpConstantComposite %s32vec3 %s32_1 %s32_2 %s32_3
%s32vec4_0123 = OpConstantComposite %s32vec4 %s32_0 %s32_1 %s32_2 %s32_3
%s32vec4_1234 = OpConstantComposite %s32vec4 %s32_1 %s32_2 %s32_3 %s32_4

%f32vec2_01 = OpConstantComposite %f32vec2 %f32_0 %f32_1
%f32vec2_12 = OpConstantComposite %f32vec2 %f32_1 %f32_2
%f32vec3_012 = OpConstantComposite %f32vec3 %f32_0 %f32_1 %f32_2
%f32vec3_123 = OpConstantComposite %f32vec3 %f32_1 %f32_2 %f32_3
%f32vec4_0123 = OpConstantComposite %f32vec4 %f32_0 %f32_1 %f32_2 %f32_3
%f32vec4_1234 = OpConstantComposite %f32vec4 %f32_1 %f32_2 %f32_3 %f32_4

%main = OpFunction %void None %func
%main_entry = OpLabel)";

  const std::string after_body =
      R"(
OpReturn
OpFunctionEnd)";

  return capabilities + capabilities_and_extensions +
         after_extension_before_body + body + after_body;
}

TEST_F(ValidateBitwise, ShiftAllSuccess) {
  const std::string body = R"(
%val1 = OpShiftRightLogical %u64 %u64_1 %s32_2
%val2 = OpShiftRightArithmetic %s32vec2 %s32vec2_12 %s32vec2_12
%val3 = OpShiftLeftLogical %u32vec2 %s32vec2_12 %u32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateBitwise, OpShiftRightLogicalWrongResultType) {
  const std::string body = R"(
%val1 = OpShiftRightLogical %bool %u64_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected int scalar or vector type as Result Type: "
                        "ShiftRightLogical"));
}

TEST_F(ValidateBitwise, OpShiftRightLogicalBaseNotInt) {
  const std::string body = R"(
%val1 = OpShiftRightLogical %u32 %f32_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Base to be int scalar or vector: ShiftRightLogical"));
}

TEST_F(ValidateBitwise, OpShiftRightLogicalBaseWrongDimension) {
  const std::string body = R"(
%val1 = OpShiftRightLogical %u32 %u32vec2_12 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Base to have the same dimension as Result Type: "
                "ShiftRightLogical"));
}

TEST_F(ValidateBitwise, OpShiftRightLogicalBaseWrongBitWidth) {
  const std::string body = R"(
%val1 = OpShiftRightLogical %u64 %u32_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Base to have the same bit width as Result Type: "
                "ShiftRightLogical"));
}

TEST_F(ValidateBitwise, OpShiftRightLogicalShiftNotInt) {
  const std::string body = R"(
%val1 = OpShiftRightLogical %u32 %u32_1 %f32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected Shift to be int scalar or vector: ShiftRightLogical"));
}

TEST_F(ValidateBitwise, OpShiftRightLogicalShiftWrongDimension) {
  const std::string body = R"(
%val1 = OpShiftRightLogical %u32 %u32_1 %s32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Shift to have the same dimension as Result Type: "
                "ShiftRightLogical"));
}

TEST_F(ValidateBitwise, LogicAllSuccess) {
  const std::string body = R"(
%val1 = OpBitwiseOr %u64 %u64_1 %s64_0
%val2 = OpBitwiseAnd %s64 %s64_1 %u64_0
%val3 = OpBitwiseXor %s32vec2 %s32vec2_12 %u32vec2_01
%val4 = OpNot %s32vec2 %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateBitwise, OpBitwiseAndWrongResultType) {
  const std::string body = R"(
%val1 = OpBitwiseAnd %bool %u64_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected int scalar or vector type as Result Type: BitwiseAnd"));
}

TEST_F(ValidateBitwise, OpBitwiseAndLeftNotInt) {
  const std::string body = R"(
%val1 = OpBitwiseAnd %u32 %f32_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected int scalar or vector as operand: BitwiseAnd "
                        "operand index 2"));
}

TEST_F(ValidateBitwise, OpBitwiseAndRightNotInt) {
  const std::string body = R"(
%val1 = OpBitwiseAnd %u32 %u32_1 %f32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected int scalar or vector as operand: BitwiseAnd "
                        "operand index 3"));
}

TEST_F(ValidateBitwise, OpBitwiseAndLeftWrongDimension) {
  const std::string body = R"(
%val1 = OpBitwiseAnd %u32 %u32vec2_12 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected operands to have the same dimension as Result Type: "
                "BitwiseAnd operand index 2"));
}

TEST_F(ValidateBitwise, OpBitwiseAndRightWrongDimension) {
  const std::string body = R"(
%val1 = OpBitwiseAnd %u32 %s32_2 %u32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected operands to have the same dimension as Result Type: "
                "BitwiseAnd operand index 3"));
}

TEST_F(ValidateBitwise, OpBitwiseAndLeftWrongBitWidth) {
  const std::string body = R"(
%val1 = OpBitwiseAnd %u64 %u32_1 %s64_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected operands to have the same bit width as Result Type: "
                "BitwiseAnd operand index 2"));
}

TEST_F(ValidateBitwise, OpBitwiseAndRightWrongBitWidth) {
  const std::string body = R"(
%val1 = OpBitwiseAnd %u64 %u64_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected operands to have the same bit width as Result Type: "
                "BitwiseAnd operand index 3"));
}

TEST_F(ValidateBitwise, OpBitFieldInsertSuccess) {
  const std::string body = R"(
%val1 = OpBitFieldInsert %u64 %u64_1 %u64_2 %s32_1 %s32_2
%val2 = OpBitFieldInsert %s32vec2 %s32vec2_12 %s32vec2_12 %s32_1 %u32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateBitwise, OpBitFieldInsertWrongResultType) {
  const std::string body = R"(
%val1 = OpBitFieldInsert %bool %u64_1 %u64_2 %s32_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected int scalar or vector type as Result Type: BitFieldInsert"));
}

TEST_F(ValidateBitwise, OpBitFieldInsertWrongBaseType) {
  const std::string body = R"(
%val1 = OpBitFieldInsert %u64 %s64_1 %u64_2 %s32_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected Base Type to be equal to Result Type: BitFieldInsert"));
}

TEST_F(ValidateBitwise, OpBitFieldInsertWrongInsertType) {
  const std::string body = R"(
%val1 = OpBitFieldInsert %u64 %u64_1 %s64_2 %s32_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected Insert Type to be equal to Result Type: BitFieldInsert"));
}

TEST_F(ValidateBitwise, OpBitFieldInsertOffsetNotInt) {
  const std::string body = R"(
%val1 = OpBitFieldInsert %u64 %u64_1 %u64_2 %f32_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Offset Type to be int scalar: BitFieldInsert"));
}

TEST_F(ValidateBitwise, OpBitFieldInsertCountNotInt) {
  const std::string body = R"(
%val1 = OpBitFieldInsert %u64 %u64_1 %u64_2 %u32_1 %f32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Count Type to be int scalar: BitFieldInsert"));
}

TEST_F(ValidateBitwise, OpBitFieldSExtractSuccess) {
  const std::string body = R"(
%val1 = OpBitFieldSExtract %u64 %u64_1 %s32_1 %s32_2
%val2 = OpBitFieldSExtract %s32vec2 %s32vec2_12 %s32_1 %u32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateBitwise, OpBitFieldSExtractWrongResultType) {
  const std::string body = R"(
%val1 = OpBitFieldSExtract %bool %u64_1 %s32_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected int scalar or vector type as Result Type: "
                        "BitFieldSExtract"));
}

TEST_F(ValidateBitwise, OpBitFieldSExtractWrongBaseType) {
  const std::string body = R"(
%val1 = OpBitFieldSExtract %u64 %s64_1 %s32_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected Base Type to be equal to Result Type: BitFieldSExtract"));
}

TEST_F(ValidateBitwise, OpBitFieldSExtractOffsetNotInt) {
  const std::string body = R"(
%val1 = OpBitFieldSExtract %u64 %u64_1 %f32_1 %s32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Offset Type to be int scalar: BitFieldSExtract"));
}

TEST_F(ValidateBitwise, OpBitFieldSExtractCountNotInt) {
  const std::string body = R"(
%val1 = OpBitFieldSExtract %u64 %u64_1 %u32_1 %f32_2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Count Type to be int scalar: BitFieldSExtract"));
}

TEST_F(ValidateBitwise, OpBitReverseSuccess) {
  const std::string body = R"(
%val1 = OpBitReverse %u64 %u64_1
%val2 = OpBitReverse %s32vec2 %s32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateBitwise, OpBitReverseWrongResultType) {
  const std::string body = R"(
%val1 = OpBitReverse %bool %u64_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected int scalar or vector type as Result Type: BitReverse"));
}

TEST_F(ValidateBitwise, OpBitReverseWrongBaseType) {
  const std::string body = R"(
%val1 = OpBitReverse %u64 %s64_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Base Type to be equal to Result Type: BitReverse"));
}

TEST_F(ValidateBitwise, OpBitCountSuccess) {
  const std::string body = R"(
%val1 = OpBitCount %s32 %u64_1
%val2 = OpBitCount %u32vec2 %s32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateBitwise, OpBitCountWrongResultType) {
  const std::string body = R"(
%val1 = OpBitCount %bool %u64_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected int scalar or vector type as Result Type: BitCount"));
}

TEST_F(ValidateBitwise, OpBitCountBaseNotInt) {
  const std::string body = R"(
%val1 = OpBitCount %u32 %f64_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Base Type to be int scalar or vector: BitCount"));
}

TEST_F(ValidateBitwise, OpBitCountBaseWrongDimension) {
  const std::string body = R"(
%val1 = OpBitCount %u32 %u32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Base dimension to be equal to Result Type dimension: "
                "BitCount"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
