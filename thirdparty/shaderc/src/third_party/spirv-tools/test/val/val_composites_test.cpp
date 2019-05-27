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

#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

using ValidateComposites = spvtest::ValidateBase<bool>;

std::string GenerateShaderCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "",
    const std::string& execution_model = "Fragment") {
  std::ostringstream ss;
  ss << R"(
OpCapability Shader
OpCapability Float64
)";

  ss << capabilities_and_extensions;
  ss << "OpMemoryModel Logical GLSL450\n";
  ss << "OpEntryPoint " << execution_model << " %main \"main\"\n";
  if (execution_model == "Fragment") {
    ss << "OpExecutionMode %main OriginUpperLeft\n";
  }

  ss << R"(
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%f64 = OpTypeFloat 64
%u32 = OpTypeInt 32 0
%s32 = OpTypeInt 32 1
%f32vec2 = OpTypeVector %f32 2
%f32vec3 = OpTypeVector %f32 3
%f32vec4 = OpTypeVector %f32 4
%f64vec2 = OpTypeVector %f64 2
%u32vec2 = OpTypeVector %u32 2
%u32vec4 = OpTypeVector %u32 4
%f64mat22 = OpTypeMatrix %f64vec2 2
%f32mat22 = OpTypeMatrix %f32vec2 2
%f32mat23 = OpTypeMatrix %f32vec2 3
%f32mat32 = OpTypeMatrix %f32vec3 2

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%f32_2 = OpConstant %f32 2
%f32_3 = OpConstant %f32 3
%f32vec2_01 = OpConstantComposite %f32vec2 %f32_0 %f32_1
%f32vec2_12 = OpConstantComposite %f32vec2 %f32_1 %f32_2
%f32vec4_0123 = OpConstantComposite %f32vec4 %f32_0 %f32_1 %f32_2 %f32_3

%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u32_2 = OpConstant %u32 2
%u32_3 = OpConstant %u32 3

%u32vec2_01 = OpConstantComposite %u32vec2 %u32_0 %u32_1
%u32vec4_0123 = OpConstantComposite %u32vec4 %u32_0 %u32_1 %u32_2 %u32_3

%f32mat22_1212 = OpConstantComposite %f32mat22 %f32vec2_12 %f32vec2_12
%f32mat23_121212 = OpConstantComposite %f32mat23 %f32vec2_12 %f32vec2_12 %f32vec2_12

%f32vec2arr3 = OpTypeArray %f32vec2 %u32_3
%f32vec2rarr = OpTypeRuntimeArray %f32vec2

%f32u32struct = OpTypeStruct %f32 %u32
%big_struct = OpTypeStruct %f32 %f32vec4 %f32mat23 %f32vec2arr3 %f32vec2rarr %f32u32struct

%ptr_big_struct = OpTypePointer Uniform %big_struct
%var_big_struct = OpVariable %ptr_big_struct Uniform

%main = OpFunction %void None %func
%main_entry = OpLabel
)";

  ss << body;

  ss << R"(
OpReturn
OpFunctionEnd)";

  return ss.str();
}

// Returns header for legacy tests taken from val_id_test.cpp.
std::string GetHeaderForTestsFromValId() {
  return R"(
OpCapability Shader
OpCapability Linkage
OpCapability Addresses
OpCapability Pipes
OpCapability LiteralSampler
OpCapability DeviceEnqueue
OpCapability Vector16
OpCapability Int8
OpCapability Int16
OpCapability Int64
OpCapability Float64
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%void_f  = OpTypeFunction %void
%int = OpTypeInt 32 0
%float = OpTypeFloat 32
%v3float = OpTypeVector %float 3
%mat4x3 = OpTypeMatrix %v3float 4
%_ptr_Private_mat4x3 = OpTypePointer Private %mat4x3
%_ptr_Private_float = OpTypePointer Private %float
%my_matrix = OpVariable %_ptr_Private_mat4x3 Private
%my_float_var = OpVariable %_ptr_Private_float Private
%_ptr_Function_float = OpTypePointer Function %float
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%int_3 = OpConstant %int 3
%int_5 = OpConstant %int 5

; Making the following nested structures.
;
; struct S {
;   bool b;
;   vec4 v[5];
;   int i;
;   mat4x3 m[5];
; }
; uniform blockName {
;   S s;
;   bool cond;
;   RunTimeArray arr;
; }

%f32arr = OpTypeRuntimeArray %float
%v4float = OpTypeVector %float 4
%array5_mat4x3 = OpTypeArray %mat4x3 %int_5
%array5_vec4 = OpTypeArray %v4float %int_5
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Function_vec4 = OpTypePointer Function %v4float
%_ptr_Uniform_vec4 = OpTypePointer Uniform %v4float
%struct_s = OpTypeStruct %int %array5_vec4 %int %array5_mat4x3
%struct_blockName = OpTypeStruct %struct_s %int %f32arr
%_ptr_Uniform_blockName = OpTypePointer Uniform %struct_blockName
%_ptr_Uniform_struct_s = OpTypePointer Uniform %struct_s
%_ptr_Uniform_array5_mat4x3 = OpTypePointer Uniform %array5_mat4x3
%_ptr_Uniform_mat4x3 = OpTypePointer Uniform %mat4x3
%_ptr_Uniform_v3float = OpTypePointer Uniform %v3float
%blockName_var = OpVariable %_ptr_Uniform_blockName Uniform
%spec_int = OpSpecConstant %int 2
%func = OpFunction %void None %void_f
%my_label = OpLabel
)";
}

TEST_F(ValidateComposites, VectorExtractDynamicSuccess) {
  const std::string body = R"(
%val1 = OpVectorExtractDynamic %f32 %f32vec4_0123 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, VectorExtractDynamicWrongResultType) {
  const std::string body = R"(
%val1 = OpVectorExtractDynamic %f32vec4 %f32vec4_0123 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a scalar type"));
}

TEST_F(ValidateComposites, VectorExtractDynamicNotVector) {
  const std::string body = R"(
%val1 = OpVectorExtractDynamic %f32 %f32mat22_1212 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Vector type to be OpTypeVector"));
}

TEST_F(ValidateComposites, VectorExtractDynamicWrongVectorComponent) {
  const std::string body = R"(
%val1 = OpVectorExtractDynamic %f32 %u32vec4_0123 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Vector component type to be equal to Result Type"));
}

TEST_F(ValidateComposites, VectorExtractDynamicWrongIndexType) {
  const std::string body = R"(
%val1 = OpVectorExtractDynamic %f32 %f32vec4_0123 %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Index to be int scalar"));
}

TEST_F(ValidateComposites, VectorInsertDynamicSuccess) {
  const std::string body = R"(
%val1 = OpVectorInsertDynamic %f32vec4 %f32vec4_0123 %f32_1 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, VectorInsertDynamicWrongResultType) {
  const std::string body = R"(
%val1 = OpVectorInsertDynamic %f32 %f32vec4_0123 %f32_1 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be OpTypeVector"));
}

TEST_F(ValidateComposites, VectorInsertDynamicNotVector) {
  const std::string body = R"(
%val1 = OpVectorInsertDynamic %f32vec4 %f32mat22_1212 %f32_1 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Vector type to be equal to Result Type"));
}

TEST_F(ValidateComposites, VectorInsertDynamicWrongComponentType) {
  const std::string body = R"(
%val1 = OpVectorInsertDynamic %f32vec4 %f32vec4_0123 %u32_1 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Component type to be equal to Result Type "
                        "component type"));
}

TEST_F(ValidateComposites, VectorInsertDynamicWrongIndexType) {
  const std::string body = R"(
%val1 = OpVectorInsertDynamic %f32vec4 %f32vec4_0123 %f32_1 %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Index to be int scalar"));
}

TEST_F(ValidateComposites, CompositeConstructNotComposite) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a composite type"));
}

TEST_F(ValidateComposites, CompositeConstructVectorSuccess) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32vec2_12 %f32vec2_12
%val2 = OpCompositeConstruct %f32vec4 %f32vec2_12 %f32_0 %f32_0
%val3 = OpCompositeConstruct %f32vec4 %f32_0 %f32_0 %f32vec2_12
%val4 = OpCompositeConstruct %f32vec4 %f32_0 %f32_1 %f32_2 %f32_3
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, CompositeConstructVectorOnlyOneConstituent) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32vec4_0123
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected number of constituents to be at least 2"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongConsituent1) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Operand 5[%float] cannot be a "
                                               "type"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongConsituent2) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32vec2_12 %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Constituents to be scalars or vectors of the same "
                "type as Result Type components"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongConsituent3) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32vec2_12 %u32_0 %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Constituents to be scalars or vectors of the same "
                "type as Result Type components"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongComponentNumber1) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32vec2_12 %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected total number of given components to be equal to the "
                "size of Result Type vector"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongComponentNumber2) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32vec2_12 %f32vec2_12 %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected total number of given components to be equal to the "
                "size of Result Type vector"));
}

TEST_F(ValidateComposites, CompositeConstructMatrixSuccess) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32mat22 %f32vec2_12 %f32vec2_12
%val2 = OpCompositeConstruct %f32mat23 %f32vec2_12 %f32vec2_12 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongConsituentNumber1) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32mat22 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected total number of Constituents to be equal to the "
                "number of columns of Result Type matrix"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongConsituentNumber2) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32mat22 %f32vec2_12 %f32vec2_12 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected total number of Constituents to be equal to the "
                "number of columns of Result Type matrix"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongConsituent) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32mat22 %f32vec2_12 %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Constituent type to be equal to the column type "
                "Result Type matrix"));
}

TEST_F(ValidateComposites, CompositeConstructArraySuccess) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec2arr3 %f32vec2_12 %f32vec2_12 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, CompositeConstructArrayWrongConsituentNumber1) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec2arr3 %f32vec2_12 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected total number of Constituents to be equal to the "
                "number of elements of Result Type array"));
}

TEST_F(ValidateComposites, CompositeConstructArrayWrongConsituentNumber2) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec2arr3 %f32vec2_12 %f32vec2_12 %f32vec2_12 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected total number of Constituents to be equal to the "
                "number of elements of Result Type array"));
}

TEST_F(ValidateComposites, CompositeConstructArrayWrongConsituent) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec2arr3 %f32vec2_12 %u32vec2_01 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Constituent type to be equal to the column type "
                "Result Type array"));
}

TEST_F(ValidateComposites, CompositeConstructStructSuccess) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32u32struct %f32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, CompositeConstructStructWrongConstituentNumber1) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32u32struct %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected total number of Constituents to be equal to the "
                "number of members of Result Type struct"));
}

TEST_F(ValidateComposites, CompositeConstructStructWrongConstituentNumber2) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32u32struct %f32_0 %u32_1 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected total number of Constituents to be equal to the "
                "number of members of Result Type struct"));
}

TEST_F(ValidateComposites, CompositeConstructStructWrongConstituent) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32u32struct %f32_0 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Constituent type to be equal to the "
                        "corresponding member type of Result Type struct"));
}

TEST_F(ValidateComposites, CopyObjectSuccess) {
  const std::string body = R"(
%val1 = OpCopyObject %f32 %f32_0
%val2 = OpCopyObject %f32vec4 %f32vec4_0123
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, CopyObjectResultTypeNotType) {
  const std::string body = R"(
%val1 = OpCopyObject %f32_0 %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("ID 19[%float_0] is not a type id"));
}

TEST_F(ValidateComposites, CopyObjectWrongOperandType) {
  const std::string body = R"(
%val1 = OpCopyObject %f32 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Result Type and Operand type to be the same"));
}

TEST_F(ValidateComposites, TransposeSuccess) {
  const std::string body = R"(
%val1 = OpTranspose %f32mat32 %f32mat23_121212
%val2 = OpTranspose %f32mat22 %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, TransposeResultTypeNotMatrix) {
  const std::string body = R"(
%val1 = OpTranspose %f32vec4 %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a matrix type"));
}

TEST_F(ValidateComposites, TransposeDifferentComponentTypes) {
  const std::string body = R"(
%val1 = OpTranspose %f64mat22 %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected component types of Matrix and Result Type to be "
                "identical"));
}

TEST_F(ValidateComposites, TransposeIncompatibleDimensions1) {
  const std::string body = R"(
%val1 = OpTranspose %f32mat23 %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected number of columns and the column size "
                        "of Matrix to be the reverse of those of Result Type"));
}

TEST_F(ValidateComposites, TransposeIncompatibleDimensions2) {
  const std::string body = R"(
%val1 = OpTranspose %f32mat32 %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected number of columns and the column size "
                        "of Matrix to be the reverse of those of Result Type"));
}

TEST_F(ValidateComposites, TransposeIncompatibleDimensions3) {
  const std::string body = R"(
%val1 = OpTranspose %f32mat23 %f32mat23_121212
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected number of columns and the column size "
                        "of Matrix to be the reverse of those of Result Type"));
}

TEST_F(ValidateComposites, CompositeExtractSuccess) {
  const std::string body = R"(
%val1 = OpCompositeExtract %f32 %f32vec4_0123 1
%val2 = OpCompositeExtract %u32 %u32vec4_0123 0
%val3 = OpCompositeExtract %f32 %f32mat22_1212 0 1
%val4 = OpCompositeExtract %f32vec2 %f32mat22_1212 0
%array = OpCompositeConstruct %f32vec2arr3 %f32vec2_12 %f32vec2_12 %f32vec2_12
%val5 = OpCompositeExtract %f32vec2 %array 2
%val6 = OpCompositeExtract %f32 %array 2 1
%struct = OpLoad %big_struct %var_big_struct
%val7 = OpCompositeExtract %f32 %struct 0
%val8 = OpCompositeExtract %f32vec4 %struct 1
%val9 = OpCompositeExtract %f32 %struct 1 2
%val10 = OpCompositeExtract %f32mat23 %struct 2
%val11 = OpCompositeExtract %f32vec2 %struct 2 2
%val12 = OpCompositeExtract %f32 %struct 2 2 1
%val13 = OpCompositeExtract %f32vec2 %struct 3 2
%val14 = OpCompositeExtract %f32 %struct 3 2 1
%val15 = OpCompositeExtract %f32vec2 %struct 4 100
%val16 = OpCompositeExtract %f32 %struct 4 1000 1
%val17 = OpCompositeExtract %f32 %struct 5 0
%val18 = OpCompositeExtract %u32 %struct 5 1
%val19 = OpCompositeExtract %big_struct %struct
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, CompositeExtractNotObject) {
  const std::string body = R"(
%val1 = OpCompositeExtract %f32 %f32vec4 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Operand 11[%v4float] cannot "
                                               "be a type"));
}

TEST_F(ValidateComposites, CompositeExtractNotComposite) {
  const std::string body = R"(
%val1 = OpCompositeExtract %f32 %f32_1 0
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Reached non-composite type while indexes still remain "
                        "to be traversed."));
}

TEST_F(ValidateComposites, CompositeExtractVectorOutOfBounds) {
  const std::string body = R"(
%val1 = OpCompositeExtract %f32 %f32vec4_0123 4
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vector access is out of bounds, "
                        "vector size is 4, but access index is 4"));
}

TEST_F(ValidateComposites, CompositeExtractMatrixOutOfCols) {
  const std::string body = R"(
%val1 = OpCompositeExtract %f32 %f32mat23_121212 3 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Matrix access is out of bounds, "
                        "matrix has 3 columns, but access index is 3"));
}

TEST_F(ValidateComposites, CompositeExtractMatrixOutOfRows) {
  const std::string body = R"(
%val1 = OpCompositeExtract %f32 %f32mat23_121212 2 5
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vector access is out of bounds, "
                        "vector size is 2, but access index is 5"));
}

TEST_F(ValidateComposites, CompositeExtractArrayOutOfBounds) {
  const std::string body = R"(
%array = OpCompositeConstruct %f32vec2arr3 %f32vec2_12 %f32vec2_12 %f32vec2_12
%val1 = OpCompositeExtract %f32vec2 %array 3
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Array access is out of bounds, "
                        "array size is 3, but access index is 3"));
}

TEST_F(ValidateComposites, CompositeExtractStructOutOfBounds) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeExtract %f32 %struct 6
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Index is out of bounds, can not find index 6 in the "
                        "structure <id> '37'. This structure has 6 members. "
                        "Largest valid index is 5."));
}

TEST_F(ValidateComposites, CompositeExtractNestedVectorOutOfBounds) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeExtract %f32 %struct 3 1 5
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vector access is out of bounds, "
                        "vector size is 2, but access index is 5"));
}

TEST_F(ValidateComposites, CompositeExtractTooManyIndices) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeExtract %f32 %struct 3 1 1 2
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Reached non-composite type while "
                        "indexes still remain to be traversed."));
}

TEST_F(ValidateComposites, CompositeExtractWrongType1) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeExtract %f32vec2 %struct 3 1 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Result type (OpTypeVector) does not match the type that results "
          "from indexing into the composite (OpTypeFloat)."));
}

TEST_F(ValidateComposites, CompositeExtractWrongType2) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeExtract %f32 %struct 3 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result type (OpTypeFloat) does not match the type "
                        "that results from indexing into the composite "
                        "(OpTypeVector)."));
}

TEST_F(ValidateComposites, CompositeExtractWrongType3) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeExtract %f32 %struct 2 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result type (OpTypeFloat) does not match the type "
                        "that results from indexing into the composite "
                        "(OpTypeVector)."));
}

TEST_F(ValidateComposites, CompositeExtractWrongType4) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeExtract %f32 %struct 4 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result type (OpTypeFloat) does not match the type "
                        "that results from indexing into the composite "
                        "(OpTypeVector)."));
}

TEST_F(ValidateComposites, CompositeExtractWrongType5) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeExtract %f32 %struct 5 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Result type (OpTypeFloat) does not match the "
          "type that results from indexing into the composite (OpTypeInt)."));
}

TEST_F(ValidateComposites, CompositeInsertSuccess) {
  const std::string body = R"(
%val1 = OpCompositeInsert %f32vec4 %f32_1 %f32vec4_0123 0
%val2 = OpCompositeInsert %u32vec4 %u32_1 %u32vec4_0123 0
%val3 = OpCompositeInsert %f32mat22 %f32_2 %f32mat22_1212 0 1
%val4 = OpCompositeInsert %f32mat22 %f32vec2_01 %f32mat22_1212 0
%array = OpCompositeConstruct %f32vec2arr3 %f32vec2_12 %f32vec2_12 %f32vec2_12
%val5 = OpCompositeInsert %f32vec2arr3 %f32vec2_01 %array 2
%val6 = OpCompositeInsert %f32vec2arr3 %f32_3 %array 2 1
%struct = OpLoad %big_struct %var_big_struct
%val7 = OpCompositeInsert %big_struct %f32_3 %struct 0
%val8 = OpCompositeInsert %big_struct %f32vec4_0123 %struct 1
%val9 = OpCompositeInsert %big_struct %f32_3 %struct 1 2
%val10 = OpCompositeInsert %big_struct %f32mat23_121212 %struct 2
%val11 = OpCompositeInsert %big_struct %f32vec2_01 %struct 2 2
%val12 = OpCompositeInsert %big_struct %f32_3 %struct 2 2 1
%val13 = OpCompositeInsert %big_struct %f32vec2_01 %struct 3 2
%val14 = OpCompositeInsert %big_struct %f32_3 %struct 3 2 1
%val15 = OpCompositeInsert %big_struct %f32vec2_01 %struct 4 100
%val16 = OpCompositeInsert %big_struct %f32_3 %struct 4 1000 1
%val17 = OpCompositeInsert %big_struct %f32_3 %struct 5 0
%val18 = OpCompositeInsert %big_struct %u32_3 %struct 5 1
%val19 = OpCompositeInsert %big_struct %struct %struct
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, CompositeInsertResultTypeDifferentFromComposite) {
  const std::string body = R"(
%val1 = OpCompositeInsert %f32 %f32_1 %f32vec4_0123 0
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The Result Type must be the same as Composite type in "
                        "OpCompositeInsert yielding Result Id 5."));
}

TEST_F(ValidateComposites, CompositeInsertNotComposite) {
  const std::string body = R"(
%val1 = OpCompositeInsert %f32 %f32_1 %f32_0 0
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Reached non-composite type while indexes still remain "
                        "to be traversed."));
}

TEST_F(ValidateComposites, CompositeInsertVectorOutOfBounds) {
  const std::string body = R"(
%val1 = OpCompositeInsert %f32vec4 %f32_1 %f32vec4_0123 4
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vector access is out of bounds, "
                        "vector size is 4, but access index is 4"));
}

TEST_F(ValidateComposites, CompositeInsertMatrixOutOfCols) {
  const std::string body = R"(
%val1 = OpCompositeInsert %f32mat23 %f32_1 %f32mat23_121212 3 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Matrix access is out of bounds, "
                        "matrix has 3 columns, but access index is 3"));
}

TEST_F(ValidateComposites, CompositeInsertMatrixOutOfRows) {
  const std::string body = R"(
%val1 = OpCompositeInsert %f32mat23 %f32_1 %f32mat23_121212 2 5
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vector access is out of bounds, "
                        "vector size is 2, but access index is 5"));
}

TEST_F(ValidateComposites, CompositeInsertArrayOutOfBounds) {
  const std::string body = R"(
%array = OpCompositeConstruct %f32vec2arr3 %f32vec2_12 %f32vec2_12 %f32vec2_12
%val1 = OpCompositeInsert %f32vec2arr3 %f32vec2_01 %array 3
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Array access is out of bounds, array "
                        "size is 3, but access index is 3"));
}

TEST_F(ValidateComposites, CompositeInsertStructOutOfBounds) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeInsert %big_struct %f32_1 %struct 6
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Index is out of bounds, can not find index 6 in the "
                        "structure <id> '37'. This structure has 6 members. "
                        "Largest valid index is 5."));
}

TEST_F(ValidateComposites, CompositeInsertNestedVectorOutOfBounds) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeInsert %big_struct %f32_1 %struct 3 1 5
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vector access is out of bounds, "
                        "vector size is 2, but access index is 5"));
}

TEST_F(ValidateComposites, CompositeInsertTooManyIndices) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeInsert %big_struct %f32_1 %struct 3 1 1 2
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Reached non-composite type while indexes still remain "
                        "to be traversed."));
}

TEST_F(ValidateComposites, CompositeInsertWrongType1) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeInsert %big_struct %f32vec2_01 %struct 3 1 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The Object type (OpTypeVector) does not match the "
                        "type that results from indexing into the Composite "
                        "(OpTypeFloat)."));
}

TEST_F(ValidateComposites, CompositeInsertWrongType2) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeInsert %big_struct %f32_1 %struct 3 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The Object type (OpTypeFloat) does not match the type "
                        "that results from indexing into the Composite "
                        "(OpTypeVector)."));
}

TEST_F(ValidateComposites, CompositeInsertWrongType3) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeInsert %big_struct %f32_1 %struct 2 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The Object type (OpTypeFloat) does not match the type "
                        "that results from indexing into the Composite "
                        "(OpTypeVector)."));
}

TEST_F(ValidateComposites, CompositeInsertWrongType4) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeInsert %big_struct %f32_1 %struct 4 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The Object type (OpTypeFloat) does not match the type "
                        "that results from indexing into the Composite "
                        "(OpTypeVector)."));
}

TEST_F(ValidateComposites, CompositeInsertWrongType5) {
  const std::string body = R"(
%struct = OpLoad %big_struct %var_big_struct
%val1 = OpCompositeInsert %big_struct %f32_1 %struct 5 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The Object type (OpTypeFloat) does not match the type "
                        "that results from indexing into the Composite "
                        "(OpTypeInt)."));
}

// Tests ported from val_id_test.cpp.

// Valid. Tests both CompositeExtract and CompositeInsert with 255 indexes.
TEST_F(ValidateComposites, CompositeExtractInsertLimitsGood) {
  int depth = 255;
  std::string header = GetHeaderForTestsFromValId();
  header.erase(header.find("%func"));
  std::ostringstream spirv;
  spirv << header << std::endl;

  // Build nested structures. Struct 'i' contains struct 'i-1'
  spirv << "%s_depth_1 = OpTypeStruct %float\n";
  for (int i = 2; i <= depth; ++i) {
    spirv << "%s_depth_" << i << " = OpTypeStruct %s_depth_" << i - 1 << "\n";
  }

  // Define Pointer and Variable to use for CompositeExtract/Insert.
  spirv << "%_ptr_Uniform_deep_struct = OpTypePointer Uniform %s_depth_"
        << depth << "\n";
  spirv << "%deep_var = OpVariable %_ptr_Uniform_deep_struct Uniform\n";

  // Function Start
  spirv << R"(
  %func = OpFunction %void None %void_f
  %my_label = OpLabel
  )";

  // OpCompositeExtract/Insert with 'n' indexes (n = depth)
  spirv << "%deep = OpLoad %s_depth_" << depth << " %deep_var" << std::endl;
  spirv << "%entry = OpCompositeExtract  %float %deep";
  for (int i = 0; i < depth; ++i) {
    spirv << " 0";
  }
  spirv << std::endl;
  spirv << "%new_composite = OpCompositeInsert %s_depth_" << depth
        << " %entry %deep";
  for (int i = 0; i < depth; ++i) {
    spirv << " 0";
  }
  spirv << std::endl;

  // Function end
  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: 256 indexes passed to OpCompositeExtract. Limit is 255.
TEST_F(ValidateComposites, CompositeExtractArgCountExceededLimitBad) {
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << std::endl;
  spirv << "%matrix = OpLoad %mat4x3 %my_matrix" << std::endl;
  spirv << "%entry = OpCompositeExtract %float %matrix";
  for (int i = 0; i < 256; ++i) {
    spirv << " 0";
  }
  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The number of indexes in OpCompositeExtract may not "
                        "exceed 255. Found 256 indexes."));
}

// Invalid: 256 indexes passed to OpCompositeInsert. Limit is 255.
TEST_F(ValidateComposites, CompositeInsertArgCountExceededLimitBad) {
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << std::endl;
  spirv << "%matrix = OpLoad %mat4x3 %my_matrix" << std::endl;
  spirv << "%new_composite = OpCompositeInsert %mat4x3 %int_0 %matrix";
  for (int i = 0; i < 256; ++i) {
    spirv << " 0";
  }
  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The number of indexes in OpCompositeInsert may not "
                        "exceed 255. Found 256 indexes."));
}

// Invalid: In OpCompositeInsert, result type must be the same as composite type
TEST_F(ValidateComposites, CompositeInsertWrongResultTypeBad) {
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << std::endl;
  spirv << "%matrix = OpLoad %mat4x3 %my_matrix" << std::endl;
  spirv << "%float_entry = OpCompositeExtract  %float %matrix 0 1" << std::endl;
  spirv << "%new_composite = OpCompositeInsert %float %float_entry %matrix 0 1"
        << std::endl;
  spirv << R"(OpReturn
              OpFunctionEnd)";
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The Result Type must be the same as Composite type"));
}

// Valid: No Indexes were passed to OpCompositeExtract, and the Result Type is
// the same as the Base Composite type.
TEST_F(ValidateComposites, CompositeExtractNoIndexesGood) {
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << std::endl;
  spirv << "%matrix = OpLoad %mat4x3 %my_matrix" << std::endl;
  spirv << "%float_entry = OpCompositeExtract  %mat4x3 %matrix" << std::endl;
  spirv << R"(OpReturn
              OpFunctionEnd)";
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: No Indexes were passed to OpCompositeExtract, but the Result Type is
// different from the Base Composite type.
TEST_F(ValidateComposites, CompositeExtractNoIndexesBad) {
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << std::endl;
  spirv << "%matrix = OpLoad %mat4x3 %my_matrix" << std::endl;
  spirv << "%float_entry = OpCompositeExtract  %float %matrix" << std::endl;
  spirv << R"(OpReturn
              OpFunctionEnd)";
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result type (OpTypeFloat) does not match the type "
                        "that results from indexing into the composite "
                        "(OpTypeMatrix)."));
}

// Valid: No Indexes were passed to OpCompositeInsert, and the type of the
// Object<id> argument matches the Composite type.
TEST_F(ValidateComposites, CompositeInsertMissingIndexesGood) {
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << std::endl;
  spirv << "%matrix   = OpLoad %mat4x3 %my_matrix" << std::endl;
  spirv << "%matrix_2 = OpLoad %mat4x3 %my_matrix" << std::endl;
  spirv << "%new_composite = OpCompositeInsert %mat4x3 %matrix_2 %matrix";
  spirv << R"(
              OpReturn
              OpFunctionEnd)";
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: No Indexes were passed to OpCompositeInsert, but the type of the
// Object<id> argument does not match the Composite type.
TEST_F(ValidateComposites, CompositeInsertMissingIndexesBad) {
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << std::endl;
  spirv << "%matrix = OpLoad %mat4x3 %my_matrix" << std::endl;
  spirv << "%new_composite = OpCompositeInsert %mat4x3 %int_0 %matrix";
  spirv << R"(
              OpReturn
              OpFunctionEnd)";
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The Object type (OpTypeInt) does not match the type "
                        "that results from indexing into the Composite "
                        "(OpTypeMatrix)."));
}

// Valid: Tests that we can index into Struct, Array, Matrix, and Vector!
TEST_F(ValidateComposites, CompositeExtractInsertIndexIntoAllTypesGood) {
  // indexes that we are passing are: 0, 3, 1, 2, 0
  // 0 will select the struct_s within the base struct (blockName)
  // 3 will select the Array that contains 5 matrices
  // 1 will select the Matrix that is at index 1 of the array
  // 2 will select the column (which is a vector) within the matrix at index 2
  // 0 will select the element at the index 0 of the vector. (which is a float).
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << R"(
    %myblock = OpLoad %struct_blockName %blockName_var
    %ss = OpCompositeExtract %struct_s %myblock 0
    %sa = OpCompositeExtract %array5_mat4x3 %myblock 0 3
    %sm = OpCompositeExtract %mat4x3 %myblock 0 3 1
    %sc = OpCompositeExtract %v3float %myblock 0 3 1 2
    %fl = OpCompositeExtract %float %myblock 0 3 1 2 0
    ;
    ; Now let's insert back at different levels...
    ;
    %b1 = OpCompositeInsert %struct_blockName %ss %myblock 0
    %b2 = OpCompositeInsert %struct_blockName %sa %myblock 0 3
    %b3 = OpCompositeInsert %struct_blockName %sm %myblock 0 3 1
    %b4 = OpCompositeInsert %struct_blockName %sc %myblock 0 3 1 2
    %b5 = OpCompositeInsert %struct_blockName %fl %myblock 0 3 1 2 0
    OpReturn
    OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid. More indexes are provided than needed for OpCompositeExtract.
TEST_F(ValidateComposites, CompositeExtractReachedScalarBad) {
  // indexes that we are passing are: 0, 3, 1, 2, 0
  // 0 will select the struct_s within the base struct (blockName)
  // 3 will select the Array that contains 5 matrices
  // 1 will select the Matrix that is at index 1 of the array
  // 2 will select the column (which is a vector) within the matrix at index 2
  // 0 will select the element at the index 0 of the vector. (which is a float).
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << R"(
    %myblock = OpLoad %struct_blockName %blockName_var
    %fl = OpCompositeExtract %float %myblock 0 3 1 2 0 1
    OpReturn
    OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Reached non-composite type while indexes still remain "
                        "to be traversed."));
}

// Invalid. More indexes are provided than needed for OpCompositeInsert.
TEST_F(ValidateComposites, CompositeInsertReachedScalarBad) {
  // indexes that we are passing are: 0, 3, 1, 2, 0
  // 0 will select the struct_s within the base struct (blockName)
  // 3 will select the Array that contains 5 matrices
  // 1 will select the Matrix that is at index 1 of the array
  // 2 will select the column (which is a vector) within the matrix at index 2
  // 0 will select the element at the index 0 of the vector. (which is a float).
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << R"(
    %myblock = OpLoad %struct_blockName %blockName_var
    %fl = OpCompositeExtract %float %myblock 0 3 1 2 0
    %b5 = OpCompositeInsert %struct_blockName %fl %myblock 0 3 1 2 0 1
    OpReturn
    OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Reached non-composite type while indexes still remain "
                        "to be traversed."));
}

// Invalid. Result type doesn't match the type we get from indexing into
// the composite.
TEST_F(ValidateComposites,
       CompositeExtractResultTypeDoesntMatchIndexedTypeBad) {
  // indexes that we are passing are: 0, 3, 1, 2, 0
  // 0 will select the struct_s within the base struct (blockName)
  // 3 will select the Array that contains 5 matrices
  // 1 will select the Matrix that is at index 1 of the array
  // 2 will select the column (which is a vector) within the matrix at index 2
  // 0 will select the element at the index 0 of the vector. (which is a float).
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << R"(
    %myblock = OpLoad %struct_blockName %blockName_var
    %fl = OpCompositeExtract %int %myblock 0 3 1 2 0
    OpReturn
    OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result type (OpTypeInt) does not match the type that "
                        "results from indexing into the composite "
                        "(OpTypeFloat)."));
}

// Invalid. Given object type doesn't match the type we get from indexing into
// the composite.
TEST_F(ValidateComposites, CompositeInsertObjectTypeDoesntMatchIndexedTypeBad) {
  // indexes that we are passing are: 0, 3, 1, 2, 0
  // 0 will select the struct_s within the base struct (blockName)
  // 3 will select the Array that contains 5 matrices
  // 1 will select the Matrix that is at index 1 of the array
  // 2 will select the column (which is a vector) within the matrix at index 2
  // 0 will select the element at the index 0 of the vector. (which is a float).
  // We are trying to insert an integer where we should be inserting a float.
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << R"(
    %myblock = OpLoad %struct_blockName %blockName_var
    %b5 = OpCompositeInsert %struct_blockName %int_0 %myblock 0 3 1 2 0
    OpReturn
    OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The Object type (OpTypeInt) does not match the type "
                        "that results from indexing into the Composite "
                        "(OpTypeFloat)."));
}

// Invalid. Index into a struct is larger than the number of struct members.
TEST_F(ValidateComposites, CompositeExtractStructIndexOutOfBoundBad) {
  // struct_blockName has 3 members (index 0,1,2). We'll try to access index 3.
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << R"(
    %myblock = OpLoad %struct_blockName %blockName_var
    %ss = OpCompositeExtract %struct_s %myblock 3
    OpReturn
    OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Index is out of bounds, can not find index 3 in the "
                        "structure <id> '25'. This structure has 3 members. "
                        "Largest valid index is 2."));
}

// Invalid. Index into a struct is larger than the number of struct members.
TEST_F(ValidateComposites, CompositeInsertStructIndexOutOfBoundBad) {
  // struct_blockName has 3 members (index 0,1,2). We'll try to access index 3.
  std::ostringstream spirv;
  spirv << GetHeaderForTestsFromValId() << R"(
    %myblock = OpLoad %struct_blockName %blockName_var
    %ss = OpCompositeExtract %struct_s %myblock 0
    %new_composite = OpCompositeInsert %struct_blockName %ss %myblock 3
    OpReturn
    OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Index is out of bounds, can not find index 3 in the structure "
                "<id> '25'. This structure has 3 members. Largest valid index "
                "is 2."));
}

// #1403: Ensure that the default spec constant value is not used to check the
// extract index.
TEST_F(ValidateComposites, ExtractFromSpecConstantSizedArray) {
  std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpDecorate %spec_const SpecId 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%spec_const = OpSpecConstant %uint 3
%uint_array = OpTypeArray %uint %spec_const
%undef = OpUndef %uint_array
%voidf = OpTypeFunction %void
%func = OpFunction %void None %voidf
%1 = OpLabel
%2 = OpCompositeExtract %uint %undef 4
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// #1403: Ensure that spec constant ops do not produce false positives.
TEST_F(ValidateComposites, ExtractFromSpecConstantOpSizedArray) {
  std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpDecorate %spec_const SpecId 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%const = OpConstant %uint 1
%spec_const = OpSpecConstant %uint 3
%spec_const_op = OpSpecConstantOp %uint IAdd %spec_const %const
%uint_array = OpTypeArray %uint %spec_const_op
%undef = OpUndef %uint_array
%voidf = OpTypeFunction %void
%func = OpFunction %void None %voidf
%1 = OpLabel
%2 = OpCompositeExtract %uint %undef 4
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// #1403: Ensure that the default spec constant value is not used to check the
// size of the array for a composite construct. This code has limited actual
// value as it is incorrect unless the specialization constant is assigned the
// value of 2, but it is still a valid module.
TEST_F(ValidateComposites, CompositeConstructSpecConstantSizedArray) {
  std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpDecorate %spec_const SpecId 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%spec_const = OpSpecConstant %uint 3
%uint_array = OpTypeArray %uint %spec_const
%voidf = OpTypeFunction %void
%func = OpFunction %void None %voidf
%1 = OpLabel
%2 = OpCompositeConstruct %uint_array %uint_0 %uint_0
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, CoopMatConstantCompositeMismatchFail) {
  const std::string body =
      R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixNV
OpExtension "SPV_NV_cooperative_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f16 = OpTypeFloat 16
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0

%u32_8 = OpConstant %u32 8
%subgroup = OpConstant %u32 3

%f16mat = OpTypeCooperativeMatrixNV %f16 %subgroup %u32_8 %u32_8

%f32_1 = OpConstant %f32 1

%f16mat_1 = OpConstantComposite %f16mat %f32_1

%main = OpFunction %void None %func
%main_entry = OpLabel

OpReturn
OpFunctionEnd)";

  CompileSuccessfully(body.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpConstantComposite Constituent <id> '11[%float_1]' type does "
                "not match the Result Type <id> '10[%10]'s component type."));
}

TEST_F(ValidateComposites, CoopMatCompositeConstructMismatchFail) {
  const std::string body =
      R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixNV
OpExtension "SPV_NV_cooperative_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f16 = OpTypeFloat 16
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0

%u32_8 = OpConstant %u32 8
%subgroup = OpConstant %u32 3

%f16mat = OpTypeCooperativeMatrixNV %f16 %subgroup %u32_8 %u32_8

%f32_1 = OpConstant %f32 1

%main = OpFunction %void None %func
%main_entry = OpLabel

%f16mat_1 = OpCompositeConstruct %f16mat %f32_1

OpReturn
OpFunctionEnd)";

  CompileSuccessfully(body.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Constituent type to be equal to the component type"));
}

TEST_F(ValidateComposites, ExtractDynamicLabelIndex) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%void_fn = OpTypeFunction %void
%float_0 = OpConstant %float 0
%v4float_0 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%func = OpFunction %void None %void_fn
%1 = OpLabel
%ex = OpVectorExtractDynamic %float %v4float_0 %v4float_0
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Index to be int scalar"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
