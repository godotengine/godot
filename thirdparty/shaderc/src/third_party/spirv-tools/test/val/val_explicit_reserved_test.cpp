// Copyright (c) 2018 Google LLC.
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

// Validation tests for illegal instructions

#include <string>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;

using ReservedSamplingInstTest = spvtest::ValidateBase<std::string>;

// Generate a shader for use with validation tests for sparse sampling
// instructions.
std::string ShaderAssembly(const std::string& instruction_under_test) {
  std::ostringstream os;
  os << R"(    OpCapability Shader
               OpCapability SparseResidency
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpExecutionMode %1 OriginUpperLeft
               OpSource GLSL 450
               OpDecorate %2 DescriptorSet 0
               OpDecorate %2 Binding 0
       %void = OpTypeVoid
          %4 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
    %float_0 = OpConstant %float 0
          %8 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
          %9 = OpTypeImage %float 2D 0 0 0 1 Unknown
         %10 = OpTypeSampledImage %9
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
          %2 = OpVariable %_ptr_UniformConstant_10 UniformConstant
    %v2float = OpTypeVector %float 2
         %13 = OpConstantComposite %v2float %float_0 %float_0
        %int = OpTypeInt 32 1
 %_struct_15 = OpTypeStruct %int %v4float
          %1 = OpFunction %void None %4
         %16 = OpLabel
         %17 = OpLoad %10 %2
)" << instruction_under_test
     << R"(
               OpReturn
               OpFunctionEnd
)";

  return os.str();
}

TEST_F(ReservedSamplingInstTest, OpImageSparseSampleProjImplicitLod) {
  const std::string input = ShaderAssembly(
      "%result = OpImageSparseSampleProjImplicitLod %_struct_15 %17 %13");
  CompileSuccessfully(input);

  EXPECT_THAT(ValidateInstructions(), Eq(SPV_ERROR_INVALID_BINARY));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Invalid Opcode name 'OpImageSparseSampleProjImplicitLod'"));
}

TEST_F(ReservedSamplingInstTest, OpImageSparseSampleProjExplicitLod) {
  const std::string input = ShaderAssembly(
      "%result = OpImageSparseSampleProjExplicitLod %_struct_15 %17 %13 Lod "
      "%float_0\n");
  CompileSuccessfully(input);

  EXPECT_THAT(ValidateInstructions(), Eq(SPV_ERROR_INVALID_BINARY));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Invalid Opcode name 'OpImageSparseSampleProjExplicitLod'"));
}

TEST_F(ReservedSamplingInstTest, OpImageSparseSampleProjDrefImplicitLod) {
  const std::string input = ShaderAssembly(
      "%result = OpImageSparseSampleProjDrefImplicitLod %_struct_15 %17 %13 "
      "%float_0\n");
  CompileSuccessfully(input);

  EXPECT_THAT(ValidateInstructions(), Eq(SPV_ERROR_INVALID_BINARY));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Invalid Opcode name 'OpImageSparseSampleProjDrefImplicitLod'"));
}

TEST_F(ReservedSamplingInstTest, OpImageSparseSampleProjDrefExplicitLod) {
  const std::string input = ShaderAssembly(
      "%result = OpImageSparseSampleProjDrefExplicitLod %_struct_15 %17 %13 "
      "%float_0 Lod "
      "%float_0\n");
  CompileSuccessfully(input);

  EXPECT_THAT(ValidateInstructions(), Eq(SPV_ERROR_INVALID_BINARY));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Invalid Opcode name 'OpImageSparseSampleProjDrefExplicitLod'"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
