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

using ValidateTypeUnique = spvtest::ValidateBase<bool>;

const spv_result_t kDuplicateTypeError = SPV_ERROR_INVALID_DATA;

const std::string& GetHeader() {
  static const std::string header = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%floatt = OpTypeFloat 32
%vec2t = OpTypeVector %floatt 2
%vec3t = OpTypeVector %floatt 3
%vec4t = OpTypeVector %floatt 4
%mat22t = OpTypeMatrix %vec2t 2
%mat33t = OpTypeMatrix %vec3t 3
%mat44t = OpTypeMatrix %vec4t 4
%intt = OpTypeInt 32 1
%uintt = OpTypeInt 32 0
%num3 = OpConstant %uintt 3
%const3 = OpConstant %uintt 3
%val3 = OpConstant %uintt 3
%array = OpTypeArray %vec3t %num3
%struct = OpTypeStruct %floatt %floatt %vec3t
%boolt = OpTypeBool
%array2 = OpTypeArray %vec3t %num3
%voidt = OpTypeVoid
%vfunct = OpTypeFunction %voidt
%struct2 = OpTypeStruct %floatt %floatt %vec3t
%false = OpConstantFalse %boolt
%true = OpConstantTrue %boolt
%runtime_arrayt = OpTypeRuntimeArray %floatt
%runtime_arrayt2 = OpTypeRuntimeArray %floatt
)";

  return header;
}

const std::string& GetBody() {
  static const std::string body = R"(
%main = OpFunction %voidt None %vfunct
%mainl = OpLabel
%a = OpIAdd %uintt %const3 %val3
%b = OpIAdd %uintt %const3 %val3
OpSelectionMerge %endl None
OpBranchConditional %true %truel %falsel
%truel = OpLabel
%add1 = OpIAdd %uintt %a %b
%add2 = OpIAdd %uintt %a %b
OpBranch %endl
%falsel = OpLabel
%sub1 = OpISub %uintt %a %b
%sub2 = OpISub %uintt %a %b
OpBranch %endl
%endl = OpLabel
OpReturn
OpFunctionEnd
)";

  return body;
}

// Returns expected error string if |opcode| produces a duplicate type
// declaration.
std::string GetErrorString(SpvOp opcode) {
  return "Duplicate non-aggregate type declarations are not allowed. Opcode: " +
         std::string(spvOpcodeString(opcode));
}

TEST_F(ValidateTypeUnique, success) {
  std::string str = GetHeader() + GetBody();
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateTypeUnique, duplicate_void) {
  std::string str = GetHeader() + R"(
%boolt2 = OpTypeVoid
)" + GetBody();
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(kDuplicateTypeError, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(GetErrorString(SpvOpTypeVoid)));
}

TEST_F(ValidateTypeUnique, duplicate_bool) {
  std::string str = GetHeader() + R"(
%boolt2 = OpTypeBool
)" + GetBody();
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(kDuplicateTypeError, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(GetErrorString(SpvOpTypeBool)));
}

TEST_F(ValidateTypeUnique, duplicate_int) {
  std::string str = GetHeader() + R"(
%uintt2 = OpTypeInt 32 0
)" + GetBody();
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(kDuplicateTypeError, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(GetErrorString(SpvOpTypeInt)));
}

TEST_F(ValidateTypeUnique, duplicate_float) {
  std::string str = GetHeader() + R"(
%floatt2 = OpTypeFloat 32
)" + GetBody();
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(kDuplicateTypeError, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(GetErrorString(SpvOpTypeFloat)));
}

TEST_F(ValidateTypeUnique, duplicate_vec3) {
  std::string str = GetHeader() + R"(
%vec3t2 = OpTypeVector %floatt 3
)" + GetBody();
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(kDuplicateTypeError, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr(GetErrorString(SpvOpTypeVector)));
}

TEST_F(ValidateTypeUnique, duplicate_mat33) {
  std::string str = GetHeader() + R"(
%mat33t2 = OpTypeMatrix %vec3t 3
)" + GetBody();
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(kDuplicateTypeError, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr(GetErrorString(SpvOpTypeMatrix)));
}

TEST_F(ValidateTypeUnique, duplicate_vfunc) {
  std::string str = GetHeader() + R"(
%vfunct2 = OpTypeFunction %voidt
)" + GetBody();
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(kDuplicateTypeError, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr(GetErrorString(SpvOpTypeFunction)));
}

TEST_F(ValidateTypeUnique, duplicate_pipe_storage) {
  std::string str = R"(
OpCapability Addresses
OpCapability Kernel
OpCapability Linkage
OpCapability Pipes
OpCapability PipeStorage
OpMemoryModel Physical32 OpenCL
%ps = OpTypePipeStorage
%ps2 = OpTypePipeStorage
)";
  CompileSuccessfully(str.c_str(), SPV_ENV_UNIVERSAL_1_1);
  ASSERT_EQ(kDuplicateTypeError, ValidateInstructions(SPV_ENV_UNIVERSAL_1_1));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr(GetErrorString(SpvOpTypePipeStorage)));
}

TEST_F(ValidateTypeUnique, duplicate_named_barrier) {
  std::string str = R"(
OpCapability Addresses
OpCapability Kernel
OpCapability Linkage
OpCapability NamedBarrier
OpMemoryModel Physical32 OpenCL
%nb = OpTypeNamedBarrier
%nb2 = OpTypeNamedBarrier
)";
  CompileSuccessfully(str.c_str(), SPV_ENV_UNIVERSAL_1_1);
  ASSERT_EQ(kDuplicateTypeError, ValidateInstructions(SPV_ENV_UNIVERSAL_1_1));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr(GetErrorString(SpvOpTypeNamedBarrier)));
}

TEST_F(ValidateTypeUnique, duplicate_forward_pointer) {
  std::string str = R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpTypeForwardPointer %ptr Generic
OpTypeForwardPointer %ptr2 Generic
%intt = OpTypeInt 32 0
%floatt = OpTypeFloat 32
%ptr = OpTypePointer Generic %intt
%ptr2 = OpTypePointer Generic %floatt
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateTypeUnique, duplicate_void_with_extension) {
  std::string str = R"(
OpCapability Addresses
OpCapability Kernel
OpCapability Linkage
OpCapability Pipes
OpExtension "SPV_VALIDATOR_ignore_type_decl_unique"
OpMemoryModel Physical32 OpenCL
%voidt = OpTypeVoid
%voidt2 = OpTypeVoid
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              Not(HasSubstr(GetErrorString(SpvOpTypeVoid))));
}

TEST_F(ValidateTypeUnique, DuplicatePointerTypesNoExtension) {
  std::string str = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%u32 = OpTypeInt 32 0
%ptr1 = OpTypePointer Input %u32
%ptr2 = OpTypePointer Input %u32
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateTypeUnique, DuplicatePointerTypesWithExtension) {
  std::string str = R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
%u32 = OpTypeInt 32 0
%ptr1 = OpTypePointer Input %u32
%ptr2 = OpTypePointer Input %u32
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              Not(HasSubstr(GetErrorString(SpvOpTypePointer))));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
