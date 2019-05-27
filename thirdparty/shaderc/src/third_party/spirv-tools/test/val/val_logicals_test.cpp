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

using ValidateLogicals = spvtest::ValidateBase<bool>;

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
%ext_inst = OpExtInstImport "GLSL.std.450"
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

%f64vec2_01 = OpConstantComposite %f64vec2 %f64_0 %f64_1
%f64vec2_12 = OpConstantComposite %f64vec2 %f64_1 %f64_2
%f64vec3_012 = OpConstantComposite %f64vec3 %f64_0 %f64_1 %f64_2
%f64vec3_123 = OpConstantComposite %f64vec3 %f64_1 %f64_2 %f64_3
%f64vec4_0123 = OpConstantComposite %f64vec4 %f64_0 %f64_1 %f64_2 %f64_3
%f64vec4_1234 = OpConstantComposite %f64vec4 %f64_1 %f64_2 %f64_3 %f64_4

%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%boolvec2_tf = OpConstantComposite %boolvec2 %true %false
%boolvec3_tft = OpConstantComposite %boolvec3 %true %false %true
%boolvec4_tftf = OpConstantComposite %boolvec4 %true %false %true %false

%f32vec4ptr = OpTypePointer Function %f32vec4

%main = OpFunction %void None %func
%main_entry = OpLabel)";

  const std::string after_body =
      R"(
OpReturn
OpFunctionEnd)";

  return capabilities + capabilities_and_extensions +
         after_extension_before_body + body + after_body;
}

std::string GenerateKernelCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "") {
  const std::string capabilities =
      R"(
OpCapability Addresses
OpCapability Kernel
OpCapability Linkage
OpCapability Int64
OpCapability Float64)";

  const std::string after_extension_before_body =
      R"(
OpMemoryModel Physical32 OpenCL
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%f64 = OpTypeFloat 64
%u64 = OpTypeInt 64 0
%boolvec2 = OpTypeVector %bool 2
%u32vec2 = OpTypeVector %u32 2
%u64vec2 = OpTypeVector %u64 2
%f32vec2 = OpTypeVector %f32 2
%f64vec2 = OpTypeVector %f64 2
%boolvec3 = OpTypeVector %bool 3
%u32vec3 = OpTypeVector %u32 3
%u64vec3 = OpTypeVector %u64 3
%f32vec3 = OpTypeVector %f32 3
%f64vec3 = OpTypeVector %f64 3
%boolvec4 = OpTypeVector %bool 4
%u32vec4 = OpTypeVector %u32 4
%u64vec4 = OpTypeVector %u64 4
%f32vec4 = OpTypeVector %f32 4
%f64vec4 = OpTypeVector %f64 4

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%f32_2 = OpConstant %f32 2
%f32_3 = OpConstant %f32 3
%f32_4 = OpConstant %f32 4

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

%f32vec2_01 = OpConstantComposite %f32vec2 %f32_0 %f32_1
%f32vec2_12 = OpConstantComposite %f32vec2 %f32_1 %f32_2
%f32vec3_012 = OpConstantComposite %f32vec3 %f32_0 %f32_1 %f32_2
%f32vec3_123 = OpConstantComposite %f32vec3 %f32_1 %f32_2 %f32_3
%f32vec4_0123 = OpConstantComposite %f32vec4 %f32_0 %f32_1 %f32_2 %f32_3
%f32vec4_1234 = OpConstantComposite %f32vec4 %f32_1 %f32_2 %f32_3 %f32_4

%f64vec2_01 = OpConstantComposite %f64vec2 %f64_0 %f64_1
%f64vec2_12 = OpConstantComposite %f64vec2 %f64_1 %f64_2
%f64vec3_012 = OpConstantComposite %f64vec3 %f64_0 %f64_1 %f64_2
%f64vec3_123 = OpConstantComposite %f64vec3 %f64_1 %f64_2 %f64_3
%f64vec4_0123 = OpConstantComposite %f64vec4 %f64_0 %f64_1 %f64_2 %f64_3
%f64vec4_1234 = OpConstantComposite %f64vec4 %f64_1 %f64_2 %f64_3 %f64_4

%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%boolvec2_tf = OpConstantComposite %boolvec2 %true %false
%boolvec3_tft = OpConstantComposite %boolvec3 %true %false %true
%boolvec4_tftf = OpConstantComposite %boolvec4 %true %false %true %false

%f32vec4ptr = OpTypePointer Function %f32vec4

%main = OpFunction %void None %func
%main_entry = OpLabel)";

  const std::string after_body =
      R"(
OpReturn
OpFunctionEnd)";

  return capabilities + capabilities_and_extensions +
         after_extension_before_body + body + after_body;
}

TEST_F(ValidateLogicals, OpAnySuccess) {
  const std::string body = R"(
%val1 = OpAny %bool %boolvec2_tf
%val2 = OpAny %bool %boolvec3_tft
%val3 = OpAny %bool %boolvec4_tftf
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLogicals, OpAnyWrongTypeId) {
  const std::string body = R"(
%val = OpAny %u32 %boolvec2_tf
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected bool scalar type as Result Type: Any"));
}

TEST_F(ValidateLogicals, OpAnyWrongOperand) {
  const std::string body = R"(
%val = OpAny %bool %u32vec3_123
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected operand to be vector bool: Any"));
}

TEST_F(ValidateLogicals, OpIsNanSuccess) {
  const std::string body = R"(
%val1 = OpIsNan %bool %f32_1
%val2 = OpIsNan %bool %f64_0
%val3 = OpIsNan %boolvec2 %f32vec2_12
%val4 = OpIsNan %boolvec3 %f32vec3_123
%val5 = OpIsNan %boolvec4 %f32vec4_1234
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLogicals, OpIsNanWrongTypeId) {
  const std::string body = R"(
%val1 = OpIsNan %u32 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected bool scalar or vector type as Result Type: IsNan"));
}

TEST_F(ValidateLogicals, OpIsNanOperandNotFloat) {
  const std::string body = R"(
%val1 = OpIsNan %bool %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected operand to be scalar or vector float: IsNan"));
}

TEST_F(ValidateLogicals, OpIsNanOperandWrongSize) {
  const std::string body = R"(
%val1 = OpIsNan %bool %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected vector sizes of Result Type and the operand to be equal: "
          "IsNan"));
}

TEST_F(ValidateLogicals, OpLessOrGreaterSuccess) {
  const std::string body = R"(
%val1 = OpLessOrGreater %bool %f32_0 %f32_1
%val2 = OpLessOrGreater %bool %f64_0 %f64_0
%val3 = OpLessOrGreater %boolvec2 %f32vec2_12 %f32vec2_12
%val4 = OpLessOrGreater %boolvec3 %f32vec3_123 %f32vec3_123
%val5 = OpLessOrGreater %boolvec4 %f32vec4_1234 %f32vec4_1234
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLogicals, OpLessOrGreaterWrongTypeId) {
  const std::string body = R"(
%val1 = OpLessOrGreater %u32 %f32_1 %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected bool scalar or vector type as Result Type: LessOrGreater"));
}

TEST_F(ValidateLogicals, OpLessOrGreaterLeftOperandNotFloat) {
  const std::string body = R"(
%val1 = OpLessOrGreater %bool %u32_1 %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected operands to be scalar or vector float: LessOrGreater"));
}

TEST_F(ValidateLogicals, OpLessOrGreaterLeftOperandWrongSize) {
  const std::string body = R"(
%val1 = OpLessOrGreater %bool %f32vec2_12 %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected vector sizes of Result Type and the operands to be equal: "
          "LessOrGreater"));
}

TEST_F(ValidateLogicals, OpLessOrGreaterOperandsDifferentType) {
  const std::string body = R"(
%val1 = OpLessOrGreater %bool %f32_1 %f64_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected left and right operands to have the same type: "
                "LessOrGreater"));
}

TEST_F(ValidateLogicals, OpFOrdEqualSuccess) {
  const std::string body = R"(
%val1 = OpFOrdEqual %bool %f32_0 %f32_1
%val2 = OpFOrdEqual %bool %f64_0 %f64_0
%val3 = OpFOrdEqual %boolvec2 %f32vec2_12 %f32vec2_12
%val4 = OpFOrdEqual %boolvec3 %f32vec3_123 %f32vec3_123
%val5 = OpFOrdEqual %boolvec4 %f32vec4_1234 %f32vec4_1234
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLogicals, OpFOrdEqualWrongTypeId) {
  const std::string body = R"(
%val1 = OpFOrdEqual %u32 %f32_1 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected bool scalar or vector type as Result Type: FOrdEqual"));
}

TEST_F(ValidateLogicals, OpFOrdEqualLeftOperandNotFloat) {
  const std::string body = R"(
%val1 = OpFOrdEqual %bool %u32_1 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected operands to be scalar or vector float: FOrdEqual"));
}

TEST_F(ValidateLogicals, OpFOrdEqualLeftOperandWrongSize) {
  const std::string body = R"(
%val1 = OpFOrdEqual %bool %f32vec2_12 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected vector sizes of Result Type and the operands to be equal: "
          "FOrdEqual"));
}

TEST_F(ValidateLogicals, OpFOrdEqualOperandsDifferentType) {
  const std::string body = R"(
%val1 = OpFOrdEqual %bool %f32_1 %f64_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected left and right operands to have the same type: "
                "FOrdEqual"));
}

TEST_F(ValidateLogicals, OpLogicalEqualSuccess) {
  const std::string body = R"(
%val1 = OpLogicalEqual %bool %true %false
%val2 = OpLogicalEqual %boolvec2 %boolvec2_tf   %boolvec2_tf
%val3 = OpLogicalEqual %boolvec3 %boolvec3_tft  %boolvec3_tft
%val4 = OpLogicalEqual %boolvec4 %boolvec4_tftf %boolvec4_tftf
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLogicals, OpLogicalEqualWrongTypeId) {
  const std::string body = R"(
%val1 = OpLogicalEqual %u32 %true %false
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected bool scalar or vector type as Result Type: LogicalEqual"));
}

TEST_F(ValidateLogicals, OpLogicalEqualWrongLeftOperand) {
  const std::string body = R"(
%val1 = OpLogicalEqual %bool %boolvec2_tf %false
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected both operands to be of Result Type: LogicalEqual"));
}

TEST_F(ValidateLogicals, OpLogicalEqualWrongRightOperand) {
  const std::string body = R"(
%val1 = OpLogicalEqual %boolvec2 %boolvec2_tf %false
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected both operands to be of Result Type: LogicalEqual"));
}

TEST_F(ValidateLogicals, OpLogicalNotSuccess) {
  const std::string body = R"(
%val1 = OpLogicalNot %bool %true
%val2 = OpLogicalNot %boolvec2 %boolvec2_tf
%val3 = OpLogicalNot %boolvec3 %boolvec3_tft
%val4 = OpLogicalNot %boolvec4 %boolvec4_tftf
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLogicals, OpLogicalNotWrongTypeId) {
  const std::string body = R"(
%val1 = OpLogicalNot %u32 %true
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected bool scalar or vector type as Result Type: LogicalNot"));
}

TEST_F(ValidateLogicals, OpLogicalNotWrongOperand) {
  const std::string body = R"(
%val1 = OpLogicalNot %bool %boolvec2_tf
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected operand to be of Result Type: LogicalNot"));
}

TEST_F(ValidateLogicals, OpSelectSuccess) {
  const std::string body = R"(
%val1 = OpSelect %u32 %true %u32_0 %u32_1
%val2 = OpSelect %f32 %true %f32_0 %f32_1
%val3 = OpSelect %f64 %true %f64_0 %f64_1
%val4 = OpSelect %f32vec2 %boolvec2_tf %f32vec2_01 %f32vec2_12
%val5 = OpSelect %f32vec4 %boolvec4_tftf %f32vec4_0123 %f32vec4_1234
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLogicals, OpSelectWrongTypeId) {
  const std::string body = R"(
%val1 = OpSelect %void %true %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected scalar or vector type as Result Type: Select"));
}

TEST_F(ValidateLogicals, OpSelectPointerNoCapability) {
  const std::string body = R"(
%x = OpVariable %f32vec4ptr Function
%y = OpVariable %f32vec4ptr Function
OpStore %x %f32vec4_0123
OpStore %y %f32vec4_1234
%val1 = OpSelect %f32vec4ptr %true %x %y
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Using pointers with OpSelect requires capability VariablePointers "
          "or VariablePointersStorageBuffer"));
}

TEST_F(ValidateLogicals, OpSelectPointerWithCapability1) {
  const std::string body = R"(
%x = OpVariable %f32vec4ptr Function
%y = OpVariable %f32vec4ptr Function
OpStore %x %f32vec4_0123
OpStore %y %f32vec4_1234
%val1 = OpSelect %f32vec4ptr %true %x %y
)";

  const std::string extra_cap_ext = R"(
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra_cap_ext).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLogicals, OpSelectPointerWithCapability2) {
  const std::string body = R"(
%x = OpVariable %f32vec4ptr Function
%y = OpVariable %f32vec4ptr Function
OpStore %x %f32vec4_0123
OpStore %y %f32vec4_1234
%val1 = OpSelect %f32vec4ptr %true %x %y
)";

  const std::string extra_cap_ext = R"(
OpCapability VariablePointersStorageBuffer
OpExtension "SPV_KHR_variable_pointers"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra_cap_ext).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLogicals, OpSelectWrongCondition) {
  const std::string body = R"(
%val1 = OpSelect %u32 %u32_1 %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected bool scalar or vector type as condition: Select"));
}

TEST_F(ValidateLogicals, OpSelectWrongConditionDimension) {
  const std::string body = R"(
%val1 = OpSelect %u32vec2 %true %u32vec2_01 %u32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected vector sizes of Result Type and the condition to be equal: "
          "Select"));
}

TEST_F(ValidateLogicals, OpSelectWrongLeftObject) {
  const std::string body = R"(
%val1 = OpSelect %bool %true %u32vec2_01 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected both objects to be of Result Type: Select"));
}

TEST_F(ValidateLogicals, OpSelectWrongRightObject) {
  const std::string body = R"(
%val1 = OpSelect %bool %true %u32_1 %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected both objects to be of Result Type: Select"));
}

TEST_F(ValidateLogicals, OpIEqualSuccess) {
  const std::string body = R"(
%val1 = OpIEqual %bool %u32_0 %s32_1
%val2 = OpIEqual %bool %s64_0 %u64_0
%val3 = OpIEqual %boolvec2 %s32vec2_12 %u32vec2_12
%val4 = OpIEqual %boolvec3 %s32vec3_123 %u32vec3_123
%val5 = OpIEqual %boolvec4 %s32vec4_1234 %u32vec4_1234
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLogicals, OpIEqualWrongTypeId) {
  const std::string body = R"(
%val1 = OpIEqual %u32 %s32_1 %s32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected bool scalar or vector type as Result Type: IEqual"));
}

TEST_F(ValidateLogicals, OpIEqualLeftOperandNotInt) {
  const std::string body = R"(
%val1 = OpIEqual %bool %f32_1 %s32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected operands to be scalar or vector int: IEqual"));
}

TEST_F(ValidateLogicals, OpIEqualLeftOperandWrongSize) {
  const std::string body = R"(
%val1 = OpIEqual %bool %s32vec2_12 %s32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected vector sizes of Result Type and the operands to be equal: "
          "IEqual"));
}

TEST_F(ValidateLogicals, OpIEqualRightOperandNotInt) {
  const std::string body = R"(
%val1 = OpIEqual %bool %u32_1 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected operands to be scalar or vector int: IEqual"));
}

TEST_F(ValidateLogicals, OpIEqualDifferentBitWidth) {
  const std::string body = R"(
%val1 = OpIEqual %bool %u32_1 %u64_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected both operands to have the same component bit "
                        "width: IEqual"));
}

TEST_F(ValidateLogicals, OpUGreaterThanSuccess) {
  const std::string body = R"(
%val1 = OpUGreaterThan %bool %u32_0 %u32_1
%val2 = OpUGreaterThan %bool %s32_0 %u32_1
%val3 = OpUGreaterThan %bool %u64_0 %u64_0
%val4 = OpUGreaterThan %bool %u64_0 %s64_0
%val5 = OpUGreaterThan %boolvec2 %u32vec2_12 %u32vec2_12
%val6 = OpUGreaterThan %boolvec3 %s32vec3_123 %u32vec3_123
%val7 = OpUGreaterThan %boolvec4 %u32vec4_1234 %u32vec4_1234
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLogicals, OpUGreaterThanWrongTypeId) {
  const std::string body = R"(
%val1 = OpUGreaterThan %u32 %u32_1 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected bool scalar or vector type as Result Type: UGreaterThan"));
}

TEST_F(ValidateLogicals, OpUGreaterThanLeftOperandNotInt) {
  const std::string body = R"(
%val1 = OpUGreaterThan %bool %f32_1 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected operands to be scalar or vector int: UGreaterThan"));
}

TEST_F(ValidateLogicals, OpUGreaterThanLeftOperandWrongSize) {
  const std::string body = R"(
%val1 = OpUGreaterThan %bool %u32vec2_12 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected vector sizes of Result Type and the operands to be equal: "
          "UGreaterThan"));
}

TEST_F(ValidateLogicals, OpUGreaterThanRightOperandNotInt) {
  const std::string body = R"(
%val1 = OpUGreaterThan %bool %u32_1 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected operands to be scalar or vector int: UGreaterThan"));
}

TEST_F(ValidateLogicals, OpUGreaterThanDifferentBitWidth) {
  const std::string body = R"(
%val1 = OpUGreaterThan %bool %u32_1 %u64_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected both operands to have the same component bit width: "
                "UGreaterThan"));
}

TEST_F(ValidateLogicals, OpSGreaterThanSuccess) {
  const std::string body = R"(
%val1 = OpSGreaterThan %bool %s32_0 %s32_1
%val2 = OpSGreaterThan %bool %u32_0 %s32_1
%val3 = OpSGreaterThan %bool %s64_0 %s64_0
%val4 = OpSGreaterThan %bool %s64_0 %u64_0
%val5 = OpSGreaterThan %boolvec2 %s32vec2_12 %s32vec2_12
%val6 = OpSGreaterThan %boolvec3 %s32vec3_123 %u32vec3_123
%val7 = OpSGreaterThan %boolvec4 %s32vec4_1234 %s32vec4_1234
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLogicals, OpSGreaterThanWrongTypeId) {
  const std::string body = R"(
%val1 = OpSGreaterThan %s32 %s32_1 %s32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected bool scalar or vector type as Result Type: SGreaterThan"));
}

TEST_F(ValidateLogicals, OpSGreaterThanLeftOperandNotInt) {
  const std::string body = R"(
%val1 = OpSGreaterThan %bool %f32_1 %s32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected operands to be scalar or vector int: SGreaterThan"));
}

TEST_F(ValidateLogicals, OpSGreaterThanLeftOperandWrongSize) {
  const std::string body = R"(
%val1 = OpSGreaterThan %bool %s32vec2_12 %s32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected vector sizes of Result Type and the operands to be equal: "
          "SGreaterThan"));
}

TEST_F(ValidateLogicals, OpSGreaterThanRightOperandNotInt) {
  const std::string body = R"(
%val1 = OpSGreaterThan %bool %s32_1 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected operands to be scalar or vector int: SGreaterThan"));
}

TEST_F(ValidateLogicals, OpSGreaterThanDifferentBitWidth) {
  const std::string body = R"(
%val1 = OpSGreaterThan %bool %s32_1 %s64_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected both operands to have the same component bit "
                        "width: SGreaterThan"));
}

TEST_F(ValidateLogicals, PSBSelectSuccess) {
  const std::string body = R"(
OpCapability PhysicalStorageBufferAddressesEXT
OpCapability Int64
OpCapability Shader
OpExtension "SPV_EXT_physical_storage_buffer"
OpMemoryModel PhysicalStorageBuffer64EXT GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpDecorate %val1 AliasedPointerEXT
%uint64 = OpTypeInt 64 0
%bool = OpTypeBool
%true = OpConstantTrue %bool
%ptr = OpTypePointer PhysicalStorageBufferEXT %uint64
%pptr_f = OpTypePointer Function %ptr
%void = OpTypeVoid
%voidfn = OpTypeFunction %void
%main = OpFunction %void None %voidfn
%entry = OpLabel
%val1 = OpVariable %pptr_f Function
%val2 = OpLoad %ptr %val1
%val3 = OpSelect %ptr %true %val2 %val2
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

}  // namespace
}  // namespace val
}  // namespace spvtools
