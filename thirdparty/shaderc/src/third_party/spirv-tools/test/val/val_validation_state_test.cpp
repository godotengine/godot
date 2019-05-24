// Copyright (c) 2016 Google Inc.
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

// Basic tests for the ValidationState_t datastructure.

#include <string>

#include "gmock/gmock.h"
#include "source/spirv_validator_options.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;

using ValidationStateTest = spvtest::ValidateBase<bool>;

const char kHeader[] =
    " OpCapability Shader"
    " OpCapability Linkage"
    " OpMemoryModel Logical GLSL450 ";

const char kVulkanMemoryHeader[] =
    " OpCapability Shader"
    " OpCapability VulkanMemoryModelKHR"
    " OpExtension \"SPV_KHR_vulkan_memory_model\""
    " OpMemoryModel Logical VulkanKHR ";

const char kVoidFVoid[] =
    " %void   = OpTypeVoid"
    " %void_f = OpTypeFunction %void"
    " %func   = OpFunction %void None %void_f"
    " %label  = OpLabel"
    "           OpReturn"
    "           OpFunctionEnd ";

// k*RecursiveBody examples originally from test/opt/function_test.cpp
const char* kNonRecursiveBody = R"(
OpEntryPoint Fragment %1 "main"
OpExecutionMode %1 OriginUpperLeft
%void = OpTypeVoid
%4 = OpTypeFunction %void
%float = OpTypeFloat 32
%_struct_6 = OpTypeStruct %float %float
%null = OpConstantNull %_struct_6
%7 = OpTypeFunction %_struct_6
%12 = OpFunction %_struct_6 None %7
%13 = OpLabel
OpReturnValue %null
OpFunctionEnd
%9 = OpFunction %_struct_6 None %7
%10 = OpLabel
%11 = OpFunctionCall %_struct_6 %12
OpReturnValue %null
OpFunctionEnd
%1 = OpFunction %void Pure|Const %4
%8 = OpLabel
%2 = OpFunctionCall %_struct_6 %9
OpKill
OpFunctionEnd
)";

const char* kDirectlyRecursiveBody = R"(
OpEntryPoint Fragment %1 "main"
OpExecutionMode %1 OriginUpperLeft
%void = OpTypeVoid
%4 = OpTypeFunction %void
%float = OpTypeFloat 32
%_struct_6 = OpTypeStruct %float %float
%7 = OpTypeFunction %_struct_6
%9 = OpFunction %_struct_6 None %7
%10 = OpLabel
%11 = OpFunctionCall %_struct_6 %9
OpKill
OpFunctionEnd
%1 = OpFunction %void Pure|Const %4
%8 = OpLabel
%2 = OpFunctionCall %_struct_6 %9
OpReturn
OpFunctionEnd
)";

const char* kIndirectlyRecursiveBody = R"(
OpEntryPoint Fragment %1 "main"
OpExecutionMode %1 OriginUpperLeft
%void = OpTypeVoid
%4 = OpTypeFunction %void
%float = OpTypeFloat 32
%_struct_6 = OpTypeStruct %float %float
%null = OpConstantNull %_struct_6
%7 = OpTypeFunction %_struct_6
%9 = OpFunction %_struct_6 None %7
%10 = OpLabel
%11 = OpFunctionCall %_struct_6 %12
OpReturnValue %null
OpFunctionEnd
%12 = OpFunction %_struct_6 None %7
%13 = OpLabel
%14 = OpFunctionCall %_struct_6 %9
OpReturnValue %null
OpFunctionEnd
%1 = OpFunction %void Pure|Const %4
%8 = OpLabel
%2 = OpFunctionCall %_struct_6 %9
OpKill
OpFunctionEnd
)";

// Tests that the instruction count in ValidationState is correct.
TEST_F(ValidationStateTest, CheckNumInstructions) {
  std::string spirv = std::string(kHeader) + "%int = OpTypeInt 32 0";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
  EXPECT_EQ(size_t(4), vstate_->ordered_instructions().size());
}

// Tests that the number of global variables in ValidationState is correct.
TEST_F(ValidationStateTest, CheckNumGlobalVars) {
  std::string spirv = std::string(kHeader) + R"(
     %int = OpTypeInt 32 0
%_ptr_int = OpTypePointer Input %int
   %var_1 = OpVariable %_ptr_int Input
   %var_2 = OpVariable %_ptr_int Input
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
  EXPECT_EQ(unsigned(2), vstate_->num_global_vars());
}

// Tests that the number of local variables in ValidationState is correct.
TEST_F(ValidationStateTest, CheckNumLocalVars) {
  std::string spirv = std::string(kHeader) + R"(
 %int      = OpTypeInt 32 0
 %_ptr_int = OpTypePointer Function %int
 %voidt    = OpTypeVoid
 %funct    = OpTypeFunction %voidt
 %main     = OpFunction %voidt None %funct
 %entry    = OpLabel
 %var_1    = OpVariable %_ptr_int Function
 %var_2    = OpVariable %_ptr_int Function
 %var_3    = OpVariable %_ptr_int Function
 OpReturn
 OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
  EXPECT_EQ(unsigned(3), vstate_->num_local_vars());
}

// Tests that the "id bound" in ValidationState is correct.
TEST_F(ValidationStateTest, CheckIdBound) {
  std::string spirv = std::string(kHeader) + R"(
 %int      = OpTypeInt 32 0
 %voidt    = OpTypeVoid
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
  EXPECT_EQ(unsigned(3), vstate_->getIdBound());
}

// Tests that the entry_points in ValidationState is correct.
TEST_F(ValidationStateTest, CheckEntryPoints) {
  std::string spirv = std::string(kHeader) +
                      " OpEntryPoint Vertex %func \"shader\"" +
                      std::string(kVoidFVoid);
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
  EXPECT_EQ(size_t(1), vstate_->entry_points().size());
  EXPECT_EQ(SpvOpFunction,
            vstate_->FindDef(vstate_->entry_points()[0])->opcode());
}

TEST_F(ValidationStateTest, CheckStructMemberLimitOption) {
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_struct_members, 32000u);
  EXPECT_EQ(32000u, options_->universal_limits_.max_struct_members);
}

TEST_F(ValidationStateTest, CheckNumGlobalVarsLimitOption) {
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_global_variables, 100u);
  EXPECT_EQ(100u, options_->universal_limits_.max_global_variables);
}

TEST_F(ValidationStateTest, CheckNumLocalVarsLimitOption) {
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_local_variables, 100u);
  EXPECT_EQ(100u, options_->universal_limits_.max_local_variables);
}

TEST_F(ValidationStateTest, CheckStructDepthLimitOption) {
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_struct_depth, 100u);
  EXPECT_EQ(100u, options_->universal_limits_.max_struct_depth);
}

TEST_F(ValidationStateTest, CheckSwitchBranchesLimitOption) {
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_switch_branches, 100u);
  EXPECT_EQ(100u, options_->universal_limits_.max_switch_branches);
}

TEST_F(ValidationStateTest, CheckFunctionArgsLimitOption) {
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_function_args, 100u);
  EXPECT_EQ(100u, options_->universal_limits_.max_function_args);
}

TEST_F(ValidationStateTest, CheckCFGDepthLimitOption) {
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_control_flow_nesting_depth, 100u);
  EXPECT_EQ(100u, options_->universal_limits_.max_control_flow_nesting_depth);
}

TEST_F(ValidationStateTest, CheckAccessChainIndexesLimitOption) {
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_access_chain_indexes, 100u);
  EXPECT_EQ(100u, options_->universal_limits_.max_access_chain_indexes);
}

TEST_F(ValidationStateTest, CheckNonRecursiveBodyGood) {
  std::string spirv = std::string(kHeader) + kNonRecursiveBody;
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidationStateTest, CheckVulkanNonRecursiveBodyGood) {
  std::string spirv = std::string(kVulkanMemoryHeader) + kNonRecursiveBody;
  CompileSuccessfully(spirv, SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS,
            ValidateAndRetrieveValidationState(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidationStateTest, CheckWebGPUNonRecursiveBodyGood) {
  std::string spirv = std::string(kVulkanMemoryHeader) + kNonRecursiveBody;
  CompileSuccessfully(spirv, SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidationStateTest, CheckDirectlyRecursiveBodyGood) {
  std::string spirv = std::string(kHeader) + kDirectlyRecursiveBody;
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidationStateTest, CheckVulkanDirectlyRecursiveBodyBad) {
  std::string spirv = std::string(kVulkanMemoryHeader) + kDirectlyRecursiveBody;
  CompileSuccessfully(spirv, SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            ValidateAndRetrieveValidationState(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Entry points may not have a call graph with cycles.\n "
                        " %1 = OpFunction %void Pure|Const %3\n"));
}

TEST_F(ValidationStateTest, CheckWebGPUDirectlyRecursiveBodyBad) {
  std::string spirv = std::string(kVulkanMemoryHeader) + kDirectlyRecursiveBody;
  CompileSuccessfully(spirv, SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            ValidateAndRetrieveValidationState(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Entry points may not have a call graph with cycles.\n "
                        " %1 = OpFunction %void Pure|Const %3\n"));
}

TEST_F(ValidationStateTest, CheckIndirectlyRecursiveBodyGood) {
  std::string spirv = std::string(kHeader) + kIndirectlyRecursiveBody;
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidationStateTest, CheckVulkanIndirectlyRecursiveBodyBad) {
  std::string spirv =
      std::string(kVulkanMemoryHeader) + kIndirectlyRecursiveBody;
  CompileSuccessfully(spirv, SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            ValidateAndRetrieveValidationState(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Entry points may not have a call graph with cycles.\n "
                        " %1 = OpFunction %void Pure|Const %3\n"));
}

// Indirectly recursive functions are caught by the function definition layout
// rules, because they cause a situation where there are 2 functions that have
// to be before each other, and layout is checked earlier.
TEST_F(ValidationStateTest, CheckWebGPUIndirectlyRecursiveBodyBad) {
  std::string spirv =
      std::string(kVulkanMemoryHeader) + kIndirectlyRecursiveBody;
  CompileSuccessfully(spirv, SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_LAYOUT,
            ValidateAndRetrieveValidationState(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("For WebGPU, functions need to be defined before being "
                        "called.\n  %10 = OpFunctionCall %_struct_5 %11\n"));
}

TEST_F(ValidationStateTest,
       CheckWebGPUDuplicateEntryNamesDifferentFunctionsBad) {
  std::string spirv = std::string(kVulkanMemoryHeader) + R"(
OpEntryPoint Fragment %func_1 "main"
OpEntryPoint Vertex %func_2 "main"
OpExecutionMode %func_1 OriginUpperLeft
%void    = OpTypeVoid
%void_f  = OpTypeFunction %void
%func_1  = OpFunction %void None %void_f
%label_1 = OpLabel
           OpReturn
           OpFunctionEnd
%func_2  = OpFunction %void None %void_f
%label_2 = OpLabel
           OpReturn
           OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            ValidateAndRetrieveValidationState(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Entry point name \"main\" is not unique, which is not allow "
                "in WebGPU env.\n  %1 = OpFunction %void None %4\n"));
}

TEST_F(ValidationStateTest, CheckWebGPUDuplicateEntryNamesSameFunctionBad) {
  std::string spirv = std::string(kVulkanMemoryHeader) + R"(
OpEntryPoint GLCompute %func_1 "main"
OpEntryPoint Vertex %func_1 "main"
%void    = OpTypeVoid
%void_f  = OpTypeFunction %void
%func_1  = OpFunction %void None %void_f
%label_1 = OpLabel
           OpReturn
           OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            ValidateAndRetrieveValidationState(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Entry point name \"main\" is not unique, which is not allow "
                "in WebGPU env.\n  %1 = OpFunction %void None %3\n"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
