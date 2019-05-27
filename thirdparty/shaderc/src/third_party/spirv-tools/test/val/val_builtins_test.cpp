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

// Tests validation rules of GLSL.450.std and OpenCL.std extended instructions.
// Doesn't test OpenCL.std vector size 2, 3, 4, 8 or 16 rules (not supported
// by standard SPIR-V).

#include <cstring>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "source/spirv_target_env.h"
#include "test/unit_spirv.h"
#include "test/val/val_code_generator.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

struct TestResult {
  TestResult(spv_result_t in_validation_result = SPV_SUCCESS,
             const char* in_error_str = nullptr,
             const char* in_error_str2 = nullptr)
      : validation_result(in_validation_result),
        error_str(in_error_str),
        error_str2(in_error_str2) {}
  spv_result_t validation_result;
  const char* error_str;
  const char* error_str2;
};

using ::testing::Combine;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::Values;
using ::testing::ValuesIn;

using ValidateBuiltIns = spvtest::ValidateBase<bool>;
using ValidateVulkanCombineBuiltInExecutionModelDataTypeResult =
    spvtest::ValidateBase<std::tuple<const char*, const char*, const char*,
                                     const char*, TestResult>>;
using ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult =
    spvtest::ValidateBase<std::tuple<const char*, const char*, const char*,
                                     const char*, TestResult>>;
using ValidateVulkanCombineBuiltInArrayedVariable = spvtest::ValidateBase<
    std::tuple<const char*, const char*, const char*, const char*, TestResult>>;
using ValidateWebGPUCombineBuiltInArrayedVariable = spvtest::ValidateBase<
    std::tuple<const char*, const char*, const char*, const char*, TestResult>>;


bool InitializerRequired(spv_target_env env, const char* const storage_class) {
  return spvIsWebGPUEnv(env) && (strncmp(storage_class, "Output", 6) == 0 ||
                                 strncmp(storage_class, "Private", 7) == 0 ||
                                 strncmp(storage_class, "Function", 8) == 0);
}

CodeGenerator GetInMainCodeGenerator(spv_target_env env,
                                     const char* const built_in,
                                     const char* const execution_model,
                                     const char* const storage_class,
                                     const char* const data_type) {
  CodeGenerator generator =
      spvIsWebGPUEnv(env) ? CodeGenerator::GetWebGPUShaderCodeGenerator()
                          : CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = "OpMemberDecorate %built_in_type 0 BuiltIn ";
  generator.before_types_ += built_in;
  generator.before_types_ += "\n";

  std::ostringstream after_types;

  after_types << "%built_in_type = OpTypeStruct " << data_type << "\n";
  if (InitializerRequired(env, storage_class)) {
    after_types << "%built_in_null = OpConstantNull %built_in_type\n";
  }
  after_types << "%built_in_ptr = OpTypePointer " << storage_class
              << " %built_in_type\n";
  after_types << "%built_in_var = OpVariable %built_in_ptr " << storage_class;
  if (InitializerRequired(env, storage_class)) {
    after_types << " %built_in_null";
  }
  after_types << "\n";
  after_types << "%data_ptr = OpTypePointer " << storage_class << " "
              << data_type << "\n";
  generator.after_types_ = after_types.str();

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = execution_model;
  if (strncmp(storage_class, "Input", 5) == 0 ||
      strncmp(storage_class, "Output", 6) == 0) {
    entry_point.interfaces = "%built_in_var";
  }

  std::ostringstream execution_modes;
  if (0 == std::strcmp(execution_model, "Fragment")) {
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " OriginUpperLeft\n";
    if (0 == std::strcmp(built_in, "FragDepth")) {
      execution_modes << "OpExecutionMode %" << entry_point.name
                      << " DepthReplacing\n";
    }
  }
  if (0 == std::strcmp(execution_model, "Geometry")) {
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " InputPoints\n";
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " OutputPoints\n";
  }
  if (0 == std::strcmp(execution_model, "GLCompute")) {
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " LocalSize 1 1 1\n";
  }
  entry_point.execution_modes = execution_modes.str();

  entry_point.body = R"(
%ptr = OpAccessChain %data_ptr %built_in_var %u32_0
)";
  generator.entry_points_.push_back(std::move(entry_point));

  return generator;
}

TEST_P(ValidateVulkanCombineBuiltInExecutionModelDataTypeResult, InMain) {
  const char* const built_in = std::get<0>(GetParam());
  const char* const execution_model = std::get<1>(GetParam());
  const char* const storage_class = std::get<2>(GetParam());
  const char* const data_type = std::get<3>(GetParam());
  const TestResult& test_result = std::get<4>(GetParam());

  CodeGenerator generator = GetInMainCodeGenerator(
      SPV_ENV_VULKAN_1_0, built_in, execution_model, storage_class, data_type);

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(test_result.validation_result,
            ValidateInstructions(SPV_ENV_VULKAN_1_0));
  if (test_result.error_str) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str));
  }
  if (test_result.error_str2) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str2));
  }
}

TEST_P(ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult, InMain) {
  const char* const built_in = std::get<0>(GetParam());
  const char* const execution_model = std::get<1>(GetParam());
  const char* const storage_class = std::get<2>(GetParam());
  const char* const data_type = std::get<3>(GetParam());
  const TestResult& test_result = std::get<4>(GetParam());

  CodeGenerator generator = GetInMainCodeGenerator(
      SPV_ENV_WEBGPU_0, built_in, execution_model, storage_class, data_type);

  CompileSuccessfully(generator.Build(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(test_result.validation_result,
            ValidateInstructions(SPV_ENV_WEBGPU_0));
  if (test_result.error_str) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str));
  }
  if (test_result.error_str2) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str2));
  }
}

CodeGenerator GetInFunctionCodeGenerator(spv_target_env env,
                                         const char* const built_in,
                                         const char* const execution_model,
                                         const char* const storage_class,
                                         const char* const data_type) {
  CodeGenerator generator =
      spvIsWebGPUEnv(env) ? CodeGenerator::GetWebGPUShaderCodeGenerator()
                          : CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = "OpMemberDecorate %built_in_type 0 BuiltIn ";
  generator.before_types_ += built_in;
  generator.before_types_ += "\n";

  std::ostringstream after_types;
  after_types << "%built_in_type = OpTypeStruct " << data_type << "\n";
  if (InitializerRequired(env, storage_class)) {
    after_types << "%built_in_null = OpConstantNull %built_in_type\n";
  }
  after_types << "%built_in_ptr = OpTypePointer " << storage_class
              << " %built_in_type\n";
  after_types << "%built_in_var = OpVariable %built_in_ptr " << storage_class;
  if (InitializerRequired(env, storage_class)) {
    after_types << " %built_in_null";
  }
  after_types << "\n";
  after_types << "%data_ptr = OpTypePointer " << storage_class << " "
              << data_type << "\n";
  generator.after_types_ = after_types.str();

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = execution_model;
  if (strncmp(storage_class, "Input", 5) == 0 ||
      strncmp(storage_class, "Output", 6) == 0) {
    entry_point.interfaces = "%built_in_var";
  }

  std::ostringstream execution_modes;
  if (0 == std::strcmp(execution_model, "Fragment")) {
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " OriginUpperLeft\n";
    if (0 == std::strcmp(built_in, "FragDepth")) {
      execution_modes << "OpExecutionMode %" << entry_point.name
                      << " DepthReplacing\n";
    }
  }
  if (0 == std::strcmp(execution_model, "Geometry")) {
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " InputPoints\n";
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " OutputPoints\n";
  }
  if (0 == std::strcmp(execution_model, "GLCompute")) {
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " LocalSize 1 1 1\n";
  }
  entry_point.execution_modes = execution_modes.str();

  entry_point.body = R"(
%val2 = OpFunctionCall %void %foo
)";

  std::string function_body = R"(
%foo = OpFunction %void None %func
%foo_entry = OpLabel
%ptr = OpAccessChain %data_ptr %built_in_var %u32_0
OpReturn
OpFunctionEnd
)";

  if (spvIsWebGPUEnv(env)) {
    generator.after_types_ += function_body;
  } else {
    generator.add_at_the_end_ = function_body;
  }

  generator.entry_points_.push_back(std::move(entry_point));

  return generator;
}

TEST_P(ValidateVulkanCombineBuiltInExecutionModelDataTypeResult, InFunction) {
  const char* const built_in = std::get<0>(GetParam());
  const char* const execution_model = std::get<1>(GetParam());
  const char* const storage_class = std::get<2>(GetParam());
  const char* const data_type = std::get<3>(GetParam());
  const TestResult& test_result = std::get<4>(GetParam());

  CodeGenerator generator = GetInFunctionCodeGenerator(
      SPV_ENV_VULKAN_1_0, built_in, execution_model, storage_class, data_type);

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(test_result.validation_result,
            ValidateInstructions(SPV_ENV_VULKAN_1_0));
  if (test_result.error_str) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str));
  }
  if (test_result.error_str2) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str2));
  }
}

TEST_P(ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult, InFunction) {
  const char* const built_in = std::get<0>(GetParam());
  const char* const execution_model = std::get<1>(GetParam());
  const char* const storage_class = std::get<2>(GetParam());
  const char* const data_type = std::get<3>(GetParam());
  const TestResult& test_result = std::get<4>(GetParam());

  CodeGenerator generator = GetInFunctionCodeGenerator(
      SPV_ENV_WEBGPU_0, built_in, execution_model, storage_class, data_type);

  CompileSuccessfully(generator.Build(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(test_result.validation_result,
            ValidateInstructions(SPV_ENV_WEBGPU_0));
  if (test_result.error_str) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str));
  }
  if (test_result.error_str2) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str2));
  }
}

CodeGenerator GetVariableCodeGenerator(spv_target_env env,
                                       const char* const built_in,
                                       const char* const execution_model,
                                       const char* const storage_class,
                                       const char* const data_type) {
  CodeGenerator generator =
      spvIsWebGPUEnv(env) ? CodeGenerator::GetWebGPUShaderCodeGenerator()
                          : CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = "OpDecorate %built_in_var BuiltIn ";
  generator.before_types_ += built_in;
  generator.before_types_ += "\n";

  std::ostringstream after_types;
  if (InitializerRequired(env, storage_class)) {
    after_types << "%built_in_null = OpConstantNull " << data_type << "\n";
  }
  after_types << "%built_in_ptr = OpTypePointer " << storage_class << " "
              << data_type << "\n";
  after_types << "%built_in_var = OpVariable %built_in_ptr " << storage_class;
  if (InitializerRequired(env, storage_class)) {
    after_types << " %built_in_null";
  }
  after_types << "\n";
  generator.after_types_ = after_types.str();

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = execution_model;
  if (strncmp(storage_class, "Input", 5) == 0 ||
      strncmp(storage_class, "Output", 6) == 0) {
    entry_point.interfaces = "%built_in_var";
  }
  // Any kind of reference would do.
  entry_point.body = R"(
%val = OpBitcast %u32 %built_in_var
)";

  std::ostringstream execution_modes;
  if (0 == std::strcmp(execution_model, "Fragment")) {
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " OriginUpperLeft\n";
    if (0 == std::strcmp(built_in, "FragDepth")) {
      execution_modes << "OpExecutionMode %" << entry_point.name
                      << " DepthReplacing\n";
    }
  }
  if (0 == std::strcmp(execution_model, "Geometry")) {
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " InputPoints\n";
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " OutputPoints\n";
  }
  if (0 == std::strcmp(execution_model, "GLCompute")) {
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " LocalSize 1 1 1\n";
  }
  entry_point.execution_modes = execution_modes.str();

  generator.entry_points_.push_back(std::move(entry_point));

  return generator;
}

TEST_P(ValidateVulkanCombineBuiltInExecutionModelDataTypeResult, Variable) {
  const char* const built_in = std::get<0>(GetParam());
  const char* const execution_model = std::get<1>(GetParam());
  const char* const storage_class = std::get<2>(GetParam());
  const char* const data_type = std::get<3>(GetParam());
  const TestResult& test_result = std::get<4>(GetParam());

  CodeGenerator generator = GetVariableCodeGenerator(
      SPV_ENV_VULKAN_1_0, built_in, execution_model, storage_class, data_type);

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(test_result.validation_result,
            ValidateInstructions(SPV_ENV_VULKAN_1_0));
  if (test_result.error_str) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str));
  }
  if (test_result.error_str2) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str2));
  }
}

TEST_P(ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult, Variable) {
  const char* const built_in = std::get<0>(GetParam());
  const char* const execution_model = std::get<1>(GetParam());
  const char* const storage_class = std::get<2>(GetParam());
  const char* const data_type = std::get<3>(GetParam());
  const TestResult& test_result = std::get<4>(GetParam());

  CodeGenerator generator = GetVariableCodeGenerator(
      SPV_ENV_WEBGPU_0, built_in, execution_model, storage_class, data_type);

  CompileSuccessfully(generator.Build(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(test_result.validation_result,
            ValidateInstructions(SPV_ENV_WEBGPU_0));
  if (test_result.error_str) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str));
  }
  if (test_result.error_str2) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str2));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ClipAndCullDistanceOutputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("ClipDistance", "CullDistance"),
            Values("Vertex", "Geometry", "TessellationControl",
                   "TessellationEvaluation"),
            Values("Output"), Values("%f32arr2", "%f32arr4"),
            Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    ClipAndCullDistanceInputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("ClipDistance", "CullDistance"),
            Values("Fragment", "Geometry", "TessellationControl",
                   "TessellationEvaluation"),
            Values("Input"), Values("%f32arr2", "%f32arr4"),
            Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    ClipAndCullDistanceFragmentOutput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("ClipDistance", "CullDistance"), Values("Fragment"),
            Values("Output"), Values("%f32arr4"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "Vulkan spec doesn't allow BuiltIn ClipDistance/CullDistance "
                "to be used for variables with Output storage class if "
                "execution model is Fragment.",
                "which is called with execution model Fragment."))));

INSTANTIATE_TEST_SUITE_P(
    VertexIdAndInstanceIdVertexInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("VertexId", "InstanceId"), Values("Vertex"), Values("Input"),
            Values("%u32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "Vulkan spec doesn't allow BuiltIn VertexId/InstanceId to be "
                "used."))));

INSTANTIATE_TEST_SUITE_P(
    ClipAndCullDistanceVertexInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("ClipDistance", "CullDistance"), Values("Vertex"),
            Values("Input"), Values("%f32arr4"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "Vulkan spec doesn't allow BuiltIn ClipDistance/CullDistance "
                "to be used for variables with Input storage class if "
                "execution model is Vertex.",
                "which is called with execution model Vertex."))));

INSTANTIATE_TEST_SUITE_P(
    ClipAndCullInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("ClipDistance", "CullDistance"), Values("GLCompute"),
            Values("Input", "Output"), Values("%f32arr4"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be used only with Fragment, Vertex, TessellationControl, "
                "TessellationEvaluation or Geometry execution models"))));

INSTANTIATE_TEST_SUITE_P(
    ClipAndCullDistanceNotArray,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("ClipDistance", "CullDistance"), Values("Fragment"),
            Values("Input"), Values("%f32vec2", "%f32vec4", "%f32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit float array",
                              "is not an array"))));

INSTANTIATE_TEST_SUITE_P(
    ClipAndCullDistanceNotFloatArray,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("ClipDistance", "CullDistance"), Values("Fragment"),
            Values("Input"), Values("%u32arr2", "%u64arr4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit float array",
                              "components are not float scalar"))));

INSTANTIATE_TEST_SUITE_P(
    ClipAndCullDistanceNotF32Array,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("ClipDistance", "CullDistance"), Values("Fragment"),
            Values("Input"), Values("%f64arr2", "%f64arr4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit float array",
                              "has components with bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    FragCoordSuccess, ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragCoord"), Values("Fragment"), Values("Input"),
            Values("%f32vec4"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    FragCoordSuccess, ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragCoord"), Values("Fragment"), Values("Input"),
            Values("%f32vec4"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    FragCoordNotFragment,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("FragCoord"),
        Values("Vertex", "GLCompute", "Geometry", "TessellationControl",
               "TessellationEvaluation"),
        Values("Input"), Values("%f32vec4"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with Fragment execution model"))));

INSTANTIATE_TEST_SUITE_P(
    FragCoordNotFragment,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("FragCoord"), Values("Vertex", "GLCompute"), Values("Input"),
        Values("%f32vec4"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with Fragment execution model"))));

INSTANTIATE_TEST_SUITE_P(
    FragCoordNotInput, ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragCoord"), Values("Fragment"), Values("Output"),
            Values("%f32vec4"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    FragCoordNotInput, ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragCoord"), Values("Fragment"), Values("Output"),
            Values("%f32vec4"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    FragCoordNotFloatVector,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragCoord"), Values("Fragment"), Values("Input"),
            Values("%f32arr4", "%u32vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float vector",
                              "is not a float vector"))));

INSTANTIATE_TEST_SUITE_P(
    FragCoordNotFloatVector,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragCoord"), Values("Fragment"), Values("Input"),
            Values("%f32arr4", "%u32vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float vector",
                              "is not a float vector"))));

INSTANTIATE_TEST_SUITE_P(
    FragCoordNotFloatVec4,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragCoord"), Values("Fragment"), Values("Input"),
            Values("%f32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float vector",
                              "has 3 components"))));

INSTANTIATE_TEST_SUITE_P(
    FragCoordNotFloatVec4,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragCoord"), Values("Fragment"), Values("Input"),
            Values("%f32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float vector",
                              "has 3 components"))));

INSTANTIATE_TEST_SUITE_P(
    FragCoordNotF32Vec4,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragCoord"), Values("Fragment"), Values("Input"),
            Values("%f64vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float vector",
                              "has components with bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    FragDepthSuccess, ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragDepth"), Values("Fragment"), Values("Output"),
            Values("%f32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    FragDepthSuccess, ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragDepth"), Values("Fragment"), Values("Output"),
            Values("%f32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    FragDepthNotFragment,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("FragDepth"),
        Values("Vertex", "GLCompute", "Geometry", "TessellationControl",
               "TessellationEvaluation"),
        Values("Output"), Values("%f32"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with Fragment execution model"))));

INSTANTIATE_TEST_SUITE_P(
    FragDepthNotFragment,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("FragDepth"), Values("Vertex", "GLCompute"), Values("Output"),
        Values("%f32"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with Fragment execution model"))));

INSTANTIATE_TEST_SUITE_P(
    FragDepthNotOutput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragDepth"), Values("Fragment"), Values("Input"),
            Values("%f32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Output storage class",
                "uses storage class Input"))));

INSTANTIATE_TEST_SUITE_P(
    FragDepthNotOutput,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragDepth"), Values("Fragment"), Values("Input"),
            Values("%f32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Output storage class",
                "uses storage class Input"))));

INSTANTIATE_TEST_SUITE_P(
    FragDepthNotFloatScalar,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragDepth"), Values("Fragment"), Values("Output"),
            Values("%f32vec4", "%u32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit float scalar",
                              "is not a float scalar"))));

INSTANTIATE_TEST_SUITE_P(
    FragDepthNotFloatScalar,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragDepth"), Values("Fragment"), Values("Output"),
            Values("%f32vec4", "%u32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit float scalar",
                              "is not a float scalar"))));

INSTANTIATE_TEST_SUITE_P(
    FragDepthNotF32, ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FragDepth"), Values("Fragment"), Values("Output"),
            Values("%f64"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit float scalar",
                              "has bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    FrontFacingAndHelperInvocationSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FrontFacing", "HelperInvocation"), Values("Fragment"),
            Values("Input"), Values("%bool"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    FrontFacingSuccess,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FrontFacing"), Values("Fragment"), Values("Input"),
            Values("%bool"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    FrontFacingAndHelperInvocationNotFragment,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("FrontFacing", "HelperInvocation"),
        Values("Vertex", "GLCompute", "Geometry", "TessellationControl",
               "TessellationEvaluation"),
        Values("Input"), Values("%bool"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with Fragment execution model"))));

INSTANTIATE_TEST_SUITE_P(
    FrontFacingNotFragment,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("FrontFacing"), Values("Vertex", "GLCompute"), Values("Input"),
        Values("%bool"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with Fragment execution model"))));

INSTANTIATE_TEST_SUITE_P(
    FrontFacingAndHelperInvocationNotInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FrontFacing", "HelperInvocation"), Values("Fragment"),
            Values("Output"), Values("%bool"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    FrontFacingNotInput,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FrontFacing"), Values("Fragment"), Values("Output"),
            Values("%bool"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    FrontFacingAndHelperInvocationNotBool,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FrontFacing", "HelperInvocation"), Values("Fragment"),
            Values("Input"), Values("%f32", "%u32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a bool scalar",
                              "is not a bool scalar"))));

INSTANTIATE_TEST_SUITE_P(
    FrontFacingNotBool,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("FrontFacing"), Values("Fragment"), Values("Input"),
            Values("%f32", "%u32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a bool scalar",
                              "is not a bool scalar"))));

INSTANTIATE_TEST_SUITE_P(
    ComputeShaderInputInt32Vec3Success,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("GlobalInvocationId", "LocalInvocationId", "NumWorkgroups",
                   "WorkgroupId"),
            Values("GLCompute"), Values("Input"), Values("%u32vec3"),
            Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    ComputeShaderInputInt32Vec3Success,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("GlobalInvocationId", "LocalInvocationId", "NumWorkgroups"),
            Values("GLCompute"), Values("Input"), Values("%u32vec3"),
            Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    ComputeShaderInputInt32Vec3NotGLCompute,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("GlobalInvocationId", "LocalInvocationId", "NumWorkgroups",
               "WorkgroupId"),
        Values("Vertex", "Fragment", "Geometry", "TessellationControl",
               "TessellationEvaluation"),
        Values("Input"), Values("%u32vec3"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with GLCompute execution model"))));

INSTANTIATE_TEST_SUITE_P(
    ComputeShaderInputInt32Vec3NotGLCompute,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("GlobalInvocationId", "LocalInvocationId", "NumWorkgroups"),
        Values("Vertex", "Fragment"), Values("Input"), Values("%u32vec3"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with GLCompute execution model"))));

INSTANTIATE_TEST_SUITE_P(
    ComputeShaderInputInt32Vec3NotInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("GlobalInvocationId", "LocalInvocationId", "NumWorkgroups",
                   "WorkgroupId"),
            Values("GLCompute"), Values("Output"), Values("%u32vec3"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    ComputeShaderInputInt32Vec3NotInput,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("GlobalInvocationId", "LocalInvocationId", "NumWorkgroups"),
            Values("GLCompute"), Values("Output"), Values("%u32vec3"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    ComputeShaderInputInt32Vec3NotIntVector,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("GlobalInvocationId", "LocalInvocationId", "NumWorkgroups",
                   "WorkgroupId"),
            Values("GLCompute"), Values("Input"),
            Values("%u32arr3", "%f32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 3-component 32-bit int vector",
                              "is not an int vector"))));

INSTANTIATE_TEST_SUITE_P(
    ComputeShaderInputInt32Vec3NotIntVector,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("GlobalInvocationId", "LocalInvocationId", "NumWorkgroups"),
            Values("GLCompute"), Values("Input"),
            Values("%u32arr3", "%f32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 3-component 32-bit int vector",
                              "is not an int vector"))));

INSTANTIATE_TEST_SUITE_P(
    ComputeShaderInputInt32Vec3NotIntVec3,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("GlobalInvocationId", "LocalInvocationId", "NumWorkgroups",
                   "WorkgroupId"),
            Values("GLCompute"), Values("Input"), Values("%u32vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 3-component 32-bit int vector",
                              "has 4 components"))));

INSTANTIATE_TEST_SUITE_P(
    ComputeShaderInputInt32Vec3NotIntVec3,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("GlobalInvocationId", "LocalInvocationId", "NumWorkgroups"),
            Values("GLCompute"), Values("Input"), Values("%u32vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 3-component 32-bit int vector",
                              "has 4 components"))));

INSTANTIATE_TEST_SUITE_P(
    ComputeShaderInputInt32Vec3NotInt32Vec,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("GlobalInvocationId", "LocalInvocationId", "NumWorkgroups",
                   "WorkgroupId"),
            Values("GLCompute"), Values("Input"), Values("%u64vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 3-component 32-bit int vector",
                              "has components with bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    InvocationIdSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InvocationId"), Values("Geometry", "TessellationControl"),
            Values("Input"), Values("%u32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    InvocationIdInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InvocationId"),
            Values("Vertex", "Fragment", "GLCompute", "TessellationEvaluation"),
            Values("Input"), Values("%u32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "to be used only with TessellationControl or "
                              "Geometry execution models"))));

INSTANTIATE_TEST_SUITE_P(
    InvocationIdNotInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InvocationId"), Values("Geometry", "TessellationControl"),
            Values("Output"), Values("%u32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    InvocationIdNotIntScalar,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InvocationId"), Values("Geometry", "TessellationControl"),
            Values("Input"), Values("%f32", "%u32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "is not an int scalar"))));

INSTANTIATE_TEST_SUITE_P(
    InvocationIdNotInt32,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InvocationId"), Values("Geometry", "TessellationControl"),
            Values("Input"), Values("%u64"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "has bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    InstanceIndexSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InstanceIndex"), Values("Vertex"), Values("Input"),
            Values("%u32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    InstanceIndexSuccess,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InstanceIndex"), Values("Vertex"), Values("Input"),
            Values("%u32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    InstanceIndexInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InstanceIndex"),
            Values("Geometry", "Fragment", "GLCompute", "TessellationControl",
                   "TessellationEvaluation"),
            Values("Input"), Values("%u32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "to be used only with Vertex execution model"))));

INSTANTIATE_TEST_SUITE_P(
    InstanceIndexInvalidExecutionModel,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InstanceIndex"), Values("Fragment", "GLCompute"),
            Values("Input"), Values("%u32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "to be used only with Vertex execution model"))));

INSTANTIATE_TEST_SUITE_P(
    InstanceIndexNotInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InstanceIndex"), Values("Vertex"), Values("Output"),
            Values("%u32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    InstanceIndexNotInput,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InstanceIndex"), Values("Vertex"), Values("Output"),
            Values("%u32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    InstanceIndexNotIntScalar,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InstanceIndex"), Values("Vertex"), Values("Input"),
            Values("%f32", "%u32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "is not an int scalar"))));

INSTANTIATE_TEST_SUITE_P(
    InstanceIndexNotIntScalar,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InstanceIndex"), Values("Vertex"), Values("Input"),
            Values("%f32", "%u32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "is not an int scalar"))));

INSTANTIATE_TEST_SUITE_P(
    InstanceIndexNotInt32,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("InstanceIndex"), Values("Vertex"), Values("Input"),
            Values("%u64"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "has bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    LayerAndViewportIndexInputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Layer", "ViewportIndex"), Values("Fragment"),
            Values("Input"), Values("%u32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    LayerAndViewportIndexOutputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Layer", "ViewportIndex"), Values("Geometry"),
            Values("Output"), Values("%u32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    LayerAndViewportIndexInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Layer", "ViewportIndex"),
            Values("TessellationControl", "GLCompute"), Values("Input"),
            Values("%u32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be used only with Vertex, TessellationEvaluation, "
                "Geometry, or Fragment execution models"))));

INSTANTIATE_TEST_SUITE_P(
    LayerAndViewportIndexExecutionModelEnabledByCapability,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Layer", "ViewportIndex"),
            Values("Vertex", "TessellationEvaluation"), Values("Output"),
            Values("%u32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "requires the ShaderViewportIndexLayerEXT capability"))));

INSTANTIATE_TEST_SUITE_P(
    LayerAndViewportIndexFragmentNotInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("Layer", "ViewportIndex"), Values("Fragment"), Values("Output"),
        Values("%u32"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "Output storage class if execution model is Fragment",
                          "which is called with execution model Fragment"))));

INSTANTIATE_TEST_SUITE_P(
    LayerAndViewportIndexGeometryNotOutput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("Layer", "ViewportIndex"),
        Values("Vertex", "TessellationEvaluation", "Geometry"), Values("Input"),
        Values("%u32"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "Input storage class if execution model is Vertex, "
                          "TessellationEvaluation, or Geometry",
                          "which is called with execution model"))));

INSTANTIATE_TEST_SUITE_P(
    LayerAndViewportIndexNotIntScalar,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Layer", "ViewportIndex"), Values("Fragment"),
            Values("Input"), Values("%f32", "%u32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "is not an int scalar"))));

INSTANTIATE_TEST_SUITE_P(
    LayerAndViewportIndexNotInt32,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Layer", "ViewportIndex"), Values("Fragment"),
            Values("Input"), Values("%u64"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "has bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    PatchVerticesSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PatchVertices"),
            Values("TessellationEvaluation", "TessellationControl"),
            Values("Input"), Values("%u32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    PatchVerticesInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PatchVertices"),
            Values("Vertex", "Fragment", "GLCompute", "Geometry"),
            Values("Input"), Values("%u32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "to be used only with TessellationControl or "
                              "TessellationEvaluation execution models"))));

INSTANTIATE_TEST_SUITE_P(
    PatchVerticesNotInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PatchVertices"),
            Values("TessellationEvaluation", "TessellationControl"),
            Values("Output"), Values("%u32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    PatchVerticesNotIntScalar,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PatchVertices"),
            Values("TessellationEvaluation", "TessellationControl"),
            Values("Input"), Values("%f32", "%u32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "is not an int scalar"))));

INSTANTIATE_TEST_SUITE_P(
    PatchVerticesNotInt32,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PatchVertices"),
            Values("TessellationEvaluation", "TessellationControl"),
            Values("Input"), Values("%u64"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "has bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    PointCoordSuccess, ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PointCoord"), Values("Fragment"), Values("Input"),
            Values("%f32vec2"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    PointCoordNotFragment,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("PointCoord"),
        Values("Vertex", "GLCompute", "Geometry", "TessellationControl",
               "TessellationEvaluation"),
        Values("Input"), Values("%f32vec2"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with Fragment execution model"))));

INSTANTIATE_TEST_SUITE_P(
    PointCoordNotInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PointCoord"), Values("Fragment"), Values("Output"),
            Values("%f32vec2"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    PointCoordNotFloatVector,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PointCoord"), Values("Fragment"), Values("Input"),
            Values("%f32arr2", "%u32vec2"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 2-component 32-bit float vector",
                              "is not a float vector"))));

INSTANTIATE_TEST_SUITE_P(
    PointCoordNotFloatVec3,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PointCoord"), Values("Fragment"), Values("Input"),
            Values("%f32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 2-component 32-bit float vector",
                              "has 3 components"))));

INSTANTIATE_TEST_SUITE_P(
    PointCoordNotF32Vec4,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PointCoord"), Values("Fragment"), Values("Input"),
            Values("%f64vec2"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 2-component 32-bit float vector",
                              "has components with bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    PointSizeOutputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PointSize"),
            Values("Vertex", "Geometry", "TessellationControl",
                   "TessellationEvaluation"),
            Values("Output"), Values("%f32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    PointSizeInputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PointSize"),
            Values("Geometry", "TessellationControl", "TessellationEvaluation"),
            Values("Input"), Values("%f32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    PointSizeVertexInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PointSize"), Values("Vertex"), Values("Input"),
            Values("%f32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "Vulkan spec doesn't allow BuiltIn PointSize "
                "to be used for variables with Input storage class if "
                "execution model is Vertex.",
                "which is called with execution model Vertex."))));

INSTANTIATE_TEST_SUITE_P(
    PointSizeInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PointSize"), Values("GLCompute", "Fragment"),
            Values("Input", "Output"), Values("%f32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be used only with Vertex, TessellationControl, "
                "TessellationEvaluation or Geometry execution models"))));

INSTANTIATE_TEST_SUITE_P(
    PointSizeNotFloatScalar,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PointSize"), Values("Vertex"), Values("Output"),
            Values("%f32vec4", "%u32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit float scalar",
                              "is not a float scalar"))));

INSTANTIATE_TEST_SUITE_P(
    PointSizeNotF32, ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PointSize"), Values("Vertex"), Values("Output"),
            Values("%f64"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit float scalar",
                              "has bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    PositionOutputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Position"),
            Values("Vertex", "Geometry", "TessellationControl",
                   "TessellationEvaluation"),
            Values("Output"), Values("%f32vec4"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    PositionOutputSuccess,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Position"), Values("Vertex"), Values("Output"),
            Values("%f32vec4"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    PositionOutputFailure,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Position"), Values("Fragment", "GLCompute"),
            Values("Output"), Values("%f32vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "WebGPU spec allows BuiltIn Position to be used "
                              "only with the Vertex execution model."))));

INSTANTIATE_TEST_SUITE_P(
    PositionInputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Position"),
            Values("Geometry", "TessellationControl", "TessellationEvaluation"),
            Values("Input"), Values("%f32vec4"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    PositionInputFailure,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("Position"), Values("Vertex", "Fragment", "GLCompute"),
        Values("Input"), Values("%f32vec4"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "WebGPU spec allows BuiltIn Position to be only used "
                          "for variables with Output storage class"))));

INSTANTIATE_TEST_SUITE_P(
    PositionVertexInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Position"), Values("Vertex"), Values("Input"),
            Values("%f32vec4"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "Vulkan spec doesn't allow BuiltIn Position "
                "to be used for variables with Input storage class if "
                "execution model is Vertex.",
                "which is called with execution model Vertex."))));

INSTANTIATE_TEST_SUITE_P(
    PositionInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Position"), Values("GLCompute", "Fragment"),
            Values("Input", "Output"), Values("%f32vec4"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be used only with Vertex, TessellationControl, "
                "TessellationEvaluation or Geometry execution models"))));

INSTANTIATE_TEST_SUITE_P(
    PositionNotFloatVector,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Position"), Values("Geometry"), Values("Input"),
            Values("%f32arr4", "%u32vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float vector",
                              "is not a float vector"))));

INSTANTIATE_TEST_SUITE_P(
    PositionNotFloatVector,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("Position"), Values("Vertex"), Values("Output"),
        Values("%f32arr4", "%u32vec4"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "needs to be a 4-component 32-bit float vector"))));

INSTANTIATE_TEST_SUITE_P(
    PositionNotFloatVec4,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Position"), Values("Geometry"), Values("Input"),
            Values("%f32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float vector",
                              "has 3 components"))));

INSTANTIATE_TEST_SUITE_P(
    PositionNotFloatVec4,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("Position"), Values("Vertex"), Values("Output"),
        Values("%f32vec3"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "needs to be a 4-component 32-bit float vector"))));

INSTANTIATE_TEST_SUITE_P(
    PositionNotF32Vec4,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("Position"), Values("Geometry"), Values("Input"),
            Values("%f64vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float vector",
                              "has components with bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    PrimitiveIdInputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PrimitiveId"),
            Values("Fragment", "TessellationControl", "TessellationEvaluation",
                   "Geometry"),
            Values("Input"), Values("%u32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    PrimitiveIdOutputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PrimitiveId"), Values("Geometry"), Values("Output"),
            Values("%u32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    PrimitiveIdInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PrimitiveId"), Values("Vertex", "GLCompute"),
            Values("Input"), Values("%u32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be used only with Fragment, TessellationControl, "
                "TessellationEvaluation or Geometry execution models"))));

INSTANTIATE_TEST_SUITE_P(
    PrimitiveIdFragmentNotInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("PrimitiveId"), Values("Fragment"), Values("Output"),
        Values("%u32"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "Output storage class if execution model is Fragment",
                          "which is called with execution model Fragment"))));

INSTANTIATE_TEST_SUITE_P(
    PrimitiveIdGeometryNotInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PrimitiveId"),
            Values("TessellationControl", "TessellationEvaluation"),
            Values("Output"), Values("%u32"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "Output storage class if execution model is Tessellation",
                "which is called with execution model Tessellation"))));

INSTANTIATE_TEST_SUITE_P(
    PrimitiveIdNotIntScalar,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PrimitiveId"), Values("Fragment"), Values("Input"),
            Values("%f32", "%u32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "is not an int scalar"))));

INSTANTIATE_TEST_SUITE_P(
    PrimitiveIdNotInt32,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PrimitiveId"), Values("Fragment"), Values("Input"),
            Values("%u64"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "has bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    SampleIdSuccess, ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SampleId"), Values("Fragment"), Values("Input"),
            Values("%u32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    SampleIdInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("SampleId"),
        Values("Vertex", "GLCompute", "Geometry", "TessellationControl",
               "TessellationEvaluation"),
        Values("Input"), Values("%u32"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with Fragment execution model"))));

INSTANTIATE_TEST_SUITE_P(
    SampleIdNotInput, ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("SampleId"), Values("Fragment"), Values("Output"),
        Values("%u32"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "Vulkan spec allows BuiltIn SampleId to be only used "
                          "for variables with Input storage class"))));

INSTANTIATE_TEST_SUITE_P(
    SampleIdNotIntScalar,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SampleId"), Values("Fragment"), Values("Input"),
            Values("%f32", "%u32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "is not an int scalar"))));

INSTANTIATE_TEST_SUITE_P(
    SampleIdNotInt32, ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SampleId"), Values("Fragment"), Values("Input"),
            Values("%u64"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "has bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    SampleMaskSuccess, ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SampleMask"), Values("Fragment"), Values("Input", "Output"),
            Values("%u32arr2", "%u32arr4"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    SampleMaskInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("SampleMask"),
        Values("Vertex", "GLCompute", "Geometry", "TessellationControl",
               "TessellationEvaluation"),
        Values("Input"), Values("%u32arr2"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with Fragment execution model"))));

INSTANTIATE_TEST_SUITE_P(
    SampleMaskWrongStorageClass,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SampleMask"), Values("Fragment"), Values("Workgroup"),
            Values("%u32arr2"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "Vulkan spec allows BuiltIn SampleMask to be only used for "
                "variables with Input or Output storage class"))));

INSTANTIATE_TEST_SUITE_P(
    SampleMaskNotArray,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SampleMask"), Values("Fragment"), Values("Input"),
            Values("%f32", "%u32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int array",
                              "is not an array"))));

INSTANTIATE_TEST_SUITE_P(
    SampleMaskNotIntArray,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SampleMask"), Values("Fragment"), Values("Input"),
            Values("%f32arr2"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int array",
                              "components are not int scalar"))));

INSTANTIATE_TEST_SUITE_P(
    SampleMaskNotInt32Array,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SampleMask"), Values("Fragment"), Values("Input"),
            Values("%u64arr2"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int array",
                              "has components with bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    SamplePositionSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SamplePosition"), Values("Fragment"), Values("Input"),
            Values("%f32vec2"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    SamplePositionNotFragment,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("SamplePosition"),
        Values("Vertex", "GLCompute", "Geometry", "TessellationControl",
               "TessellationEvaluation"),
        Values("Input"), Values("%f32vec2"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with Fragment execution model"))));

INSTANTIATE_TEST_SUITE_P(
    SamplePositionNotInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SamplePosition"), Values("Fragment"), Values("Output"),
            Values("%f32vec2"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    SamplePositionNotFloatVector,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SamplePosition"), Values("Fragment"), Values("Input"),
            Values("%f32arr2", "%u32vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 2-component 32-bit float vector",
                              "is not a float vector"))));

INSTANTIATE_TEST_SUITE_P(
    SamplePositionNotFloatVec2,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SamplePosition"), Values("Fragment"), Values("Input"),
            Values("%f32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 2-component 32-bit float vector",
                              "has 3 components"))));

INSTANTIATE_TEST_SUITE_P(
    SamplePositionNotF32Vec2,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("SamplePosition"), Values("Fragment"), Values("Input"),
            Values("%f64vec2"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 2-component 32-bit float vector",
                              "has components with bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    TessCoordSuccess, ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessCoord"), Values("TessellationEvaluation"),
            Values("Input"), Values("%f32vec3"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    TessCoordNotFragment,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("TessCoord"),
        Values("Vertex", "GLCompute", "Geometry", "TessellationControl",
               "Fragment"),
        Values("Input"), Values("%f32vec3"),
        Values(TestResult(
            SPV_ERROR_INVALID_DATA,
            "to be used only with TessellationEvaluation execution model"))));

INSTANTIATE_TEST_SUITE_P(
    TessCoordNotInput, ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessCoord"), Values("Fragment"), Values("Output"),
            Values("%f32vec3"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "to be only used for variables with Input storage class",
                "uses storage class Output"))));

INSTANTIATE_TEST_SUITE_P(
    TessCoordNotFloatVector,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessCoord"), Values("Fragment"), Values("Input"),
            Values("%f32arr3", "%u32vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 3-component 32-bit float vector",
                              "is not a float vector"))));

INSTANTIATE_TEST_SUITE_P(
    TessCoordNotFloatVec3,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessCoord"), Values("Fragment"), Values("Input"),
            Values("%f32vec2"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 3-component 32-bit float vector",
                              "has 2 components"))));

INSTANTIATE_TEST_SUITE_P(
    TessCoordNotF32Vec3,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessCoord"), Values("Fragment"), Values("Input"),
            Values("%f64vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 3-component 32-bit float vector",
                              "has components with bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelOuterTeseInputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelOuter"), Values("TessellationEvaluation"),
            Values("Input"), Values("%f32arr4"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    TessLevelOuterTescOutputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelOuter"), Values("TessellationControl"),
            Values("Output"), Values("%f32arr4"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    TessLevelOuterInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelOuter"),
            Values("Vertex", "GLCompute", "Geometry", "Fragment"),
            Values("Input"), Values("%f32arr4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "to be used only with TessellationControl or "
                              "TessellationEvaluation execution models."))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelOuterOutputTese,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelOuter"), Values("TessellationEvaluation"),
            Values("Output"), Values("%f32arr4"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "Vulkan spec doesn't allow TessLevelOuter/TessLevelInner to be "
                "used for variables with Output storage class if execution "
                "model is TessellationEvaluation."))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelOuterInputTesc,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelOuter"), Values("TessellationControl"),
            Values("Input"), Values("%f32arr4"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "Vulkan spec doesn't allow TessLevelOuter/TessLevelInner to be "
                "used for variables with Input storage class if execution "
                "model is TessellationControl."))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelOuterNotArray,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelOuter"), Values("TessellationEvaluation"),
            Values("Input"), Values("%f32vec4", "%f32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float array",
                              "is not an array"))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelOuterNotFloatArray,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelOuter"), Values("TessellationEvaluation"),
            Values("Input"), Values("%u32arr4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float array",
                              "components are not float scalar"))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelOuterNotFloatArr4,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelOuter"), Values("TessellationEvaluation"),
            Values("Input"), Values("%f32arr3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float array",
                              "has 3 components"))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelOuterNotF32Arr4,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelOuter"), Values("TessellationEvaluation"),
            Values("Input"), Values("%f64arr4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float array",
                              "has components with bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelInnerTeseInputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelInner"), Values("TessellationEvaluation"),
            Values("Input"), Values("%f32arr2"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    TessLevelInnerTescOutputSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelInner"), Values("TessellationControl"),
            Values("Output"), Values("%f32arr2"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    TessLevelInnerInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelInner"),
            Values("Vertex", "GLCompute", "Geometry", "Fragment"),
            Values("Input"), Values("%f32arr2"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "to be used only with TessellationControl or "
                              "TessellationEvaluation execution models."))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelInnerOutputTese,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelInner"), Values("TessellationEvaluation"),
            Values("Output"), Values("%f32arr2"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "Vulkan spec doesn't allow TessLevelOuter/TessLevelInner to be "
                "used for variables with Output storage class if execution "
                "model is TessellationEvaluation."))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelInnerInputTesc,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelInner"), Values("TessellationControl"),
            Values("Input"), Values("%f32arr2"),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "Vulkan spec doesn't allow TessLevelOuter/TessLevelInner to be "
                "used for variables with Input storage class if execution "
                "model is TessellationControl."))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelInnerNotArray,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelInner"), Values("TessellationEvaluation"),
            Values("Input"), Values("%f32vec2", "%f32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 2-component 32-bit float array",
                              "is not an array"))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelInnerNotFloatArray,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelInner"), Values("TessellationEvaluation"),
            Values("Input"), Values("%u32arr2"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 2-component 32-bit float array",
                              "components are not float scalar"))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelInnerNotFloatArr2,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelInner"), Values("TessellationEvaluation"),
            Values("Input"), Values("%f32arr3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 2-component 32-bit float array",
                              "has 3 components"))));

INSTANTIATE_TEST_SUITE_P(
    TessLevelInnerNotF32Arr2,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("TessLevelInner"), Values("TessellationEvaluation"),
            Values("Input"), Values("%f64arr2"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 2-component 32-bit float array",
                              "has components with bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    VertexIndexSuccess,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("VertexIndex"), Values("Vertex"), Values("Input"),
            Values("%u32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    VertexIndexSuccess,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("VertexIndex"), Values("Vertex"), Values("Input"),
            Values("%u32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    VertexIndexInvalidExecutionModel,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("VertexIndex"),
            Values("Fragment", "GLCompute", "Geometry", "TessellationControl",
                   "TessellationEvaluation"),
            Values("Input"), Values("%u32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "to be used only with Vertex execution model"))));

INSTANTIATE_TEST_SUITE_P(
    VertexIndexInvalidExecutionModel,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("VertexIndex"), Values("Fragment", "GLCompute"),
            Values("Input"), Values("%u32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "to be used only with Vertex execution model"))));

INSTANTIATE_TEST_SUITE_P(
    VertexIndexNotInput,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("VertexIndex"), Values("Vertex"), Values("Output"),
        Values("%u32"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "Vulkan spec allows BuiltIn VertexIndex to be only "
                          "used for variables with Input storage class"))));

INSTANTIATE_TEST_SUITE_P(
    VertexIndexNotInput,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("VertexIndex"), Values("Vertex"), Values("Output"),
        Values("%u32"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "WebGPU spec allows BuiltIn VertexIndex to be only "
                          "used for variables with Input storage class"))));

INSTANTIATE_TEST_SUITE_P(
    VertexIndexNotIntScalar,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("VertexIndex"), Values("Vertex"), Values("Input"),
            Values("%f32", "%u32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "is not an int scalar"))));

INSTANTIATE_TEST_SUITE_P(
    VertexIndexNotIntScalar,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("VertexIndex"), Values("Vertex"), Values("Input"),
            Values("%f32", "%u32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "is not an int scalar"))));

INSTANTIATE_TEST_SUITE_P(
    VertexIndexNotInt32,
    ValidateVulkanCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("VertexIndex"), Values("Vertex"), Values("Input"),
            Values("%u64"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int scalar",
                              "has bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    LocalInvocationIndexSuccess,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("LocalInvocationIndex"), Values("GLCompute"),
            Values("Input"), Values("%u32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    LocalInvocationIndexInvalidExecutionModel,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("LocalInvocationIndex"), Values("Fragment", "Vertex"),
        Values("Input"), Values("%u32"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "to be used only with GLCompute execution model"))));

INSTANTIATE_TEST_SUITE_P(
    LocalInvocationIndexNotInput,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(
        Values("LocalInvocationIndex"), Values("GLCompute"), Values("Output"),
        Values("%u32"),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "WebGPU spec allows BuiltIn LocalInvocationIndex to "
                          "be only used for variables with Input storage "
                          "class"))));

INSTANTIATE_TEST_SUITE_P(
    LocalInvocationIndexNotIntScalar,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("LocalInvocationIndex"), Values("GLCompute"),
            Values("Input"), Values("%f32", "%u32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit int", "is not an int"))));

INSTANTIATE_TEST_SUITE_P(
    WhitelistRejection,
    ValidateWebGPUCombineBuiltInExecutionModelDataTypeResult,
    Combine(Values("PointSize", "ClipDistance", "CullDistance", "VertexId",
                   "InstanceId", "PointCoord", "SampleMask", "HelperInvocation",
                   "WorkgroupId"),
            Values("Vertex"), Values("Input"), Values("%u32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "WebGPU does not allow BuiltIn"))));

CodeGenerator GetArrayedVariableCodeGenerator(spv_target_env env,
                                              const char* const built_in,
                                              const char* const execution_model,
                                              const char* const storage_class,
                                              const char* const data_type) {
  CodeGenerator generator =
      spvIsWebGPUEnv(env) ? CodeGenerator::GetWebGPUShaderCodeGenerator()
                          : CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = "OpDecorate %built_in_var BuiltIn ";
  generator.before_types_ += built_in;
  generator.before_types_ += "\n";

  std::ostringstream after_types;
  after_types << "%built_in_array = OpTypeArray " << data_type << " %u32_3\n";
  if (InitializerRequired(env, storage_class)) {
    after_types << "%built_in_array_null = OpConstantNull %built_in_array\n";
  }

  after_types << "%built_in_ptr = OpTypePointer " << storage_class
              << " %built_in_array\n";
  after_types << "%built_in_var = OpVariable %built_in_ptr " << storage_class;
  if (InitializerRequired(env, storage_class)) {
    after_types << " %built_in_array_null";
  }
  after_types << "\n";
  generator.after_types_ = after_types.str();

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = execution_model;
  entry_point.interfaces = "%built_in_var";
  // Any kind of reference would do.
  entry_point.body = R"(
%val = OpBitcast %u32 %built_in_var
)";

  std::ostringstream execution_modes;
  if (0 == std::strcmp(execution_model, "Fragment")) {
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " OriginUpperLeft\n";
    if (0 == std::strcmp(built_in, "FragDepth")) {
      execution_modes << "OpExecutionMode %" << entry_point.name
                      << " DepthReplacing\n";
    }
  }
  if (0 == std::strcmp(execution_model, "Geometry")) {
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " InputPoints\n";
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " OutputPoints\n";
  }
  if (0 == std::strcmp(execution_model, "GLCompute")) {
    execution_modes << "OpExecutionMode %" << entry_point.name
                    << " LocalSize 1 1 1\n";
  }
  entry_point.execution_modes = execution_modes.str();

  generator.entry_points_.push_back(std::move(entry_point));

  return generator;
}

TEST_P(ValidateVulkanCombineBuiltInArrayedVariable, Variable) {
  const char* const built_in = std::get<0>(GetParam());
  const char* const execution_model = std::get<1>(GetParam());
  const char* const storage_class = std::get<2>(GetParam());
  const char* const data_type = std::get<3>(GetParam());
  const TestResult& test_result = std::get<4>(GetParam());

  CodeGenerator generator = GetArrayedVariableCodeGenerator(
      SPV_ENV_VULKAN_1_0, built_in, execution_model, storage_class, data_type);

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(test_result.validation_result,
            ValidateInstructions(SPV_ENV_VULKAN_1_0));
  if (test_result.error_str) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str));
  }
  if (test_result.error_str2) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str2));
  }
}

TEST_P(ValidateWebGPUCombineBuiltInArrayedVariable, Variable) {
  const char* const built_in = std::get<0>(GetParam());
  const char* const execution_model = std::get<1>(GetParam());
  const char* const storage_class = std::get<2>(GetParam());
  const char* const data_type = std::get<3>(GetParam());
  const TestResult& test_result = std::get<4>(GetParam());

  CodeGenerator generator = GetArrayedVariableCodeGenerator(
      SPV_ENV_WEBGPU_0, built_in, execution_model, storage_class, data_type);

  CompileSuccessfully(generator.Build(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(test_result.validation_result,
            ValidateInstructions(SPV_ENV_WEBGPU_0));
  if (test_result.error_str) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str));
  }
  if (test_result.error_str2) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(test_result.error_str2));
  }
}

INSTANTIATE_TEST_SUITE_P(PointSizeArrayedF32TessControl,
                         ValidateVulkanCombineBuiltInArrayedVariable,
                         Combine(Values("PointSize"),
                                 Values("TessellationControl"), Values("Input"),
                                 Values("%f32"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    PointSizeArrayedF64TessControl, ValidateVulkanCombineBuiltInArrayedVariable,
    Combine(Values("PointSize"), Values("TessellationControl"), Values("Input"),
            Values("%f64"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit float scalar",
                              "has bit width 64"))));

INSTANTIATE_TEST_SUITE_P(
    PointSizeArrayedF32Vertex, ValidateVulkanCombineBuiltInArrayedVariable,
    Combine(Values("PointSize"), Values("Vertex"), Values("Output"),
            Values("%f32"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit float scalar",
                              "is not a float scalar"))));

INSTANTIATE_TEST_SUITE_P(PositionArrayedF32Vec4TessControl,
                         ValidateVulkanCombineBuiltInArrayedVariable,
                         Combine(Values("Position"),
                                 Values("TessellationControl"), Values("Input"),
                                 Values("%f32vec4"), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    PositionArrayedF32Vec3TessControl,
    ValidateVulkanCombineBuiltInArrayedVariable,
    Combine(Values("Position"), Values("TessellationControl"), Values("Input"),
            Values("%f32vec3"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float vector",
                              "has 3 components"))));

INSTANTIATE_TEST_SUITE_P(
    PositionArrayedF32Vec4Vertex, ValidateVulkanCombineBuiltInArrayedVariable,
    Combine(Values("Position"), Values("Vertex"), Values("Output"),
            Values("%f32vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float vector",
                              "is not a float vector"))));

INSTANTIATE_TEST_SUITE_P(
    PositionArrayedF32Vec4Vertex, ValidateWebGPUCombineBuiltInArrayedVariable,
    Combine(Values("Position"), Values("Vertex"), Values("Output"),
            Values("%f32vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 4-component 32-bit float vector",
                              "is not a float vector"))));

INSTANTIATE_TEST_SUITE_P(
    ClipAndCullDistanceOutputSuccess,
    ValidateVulkanCombineBuiltInArrayedVariable,
    Combine(Values("ClipDistance", "CullDistance"),
            Values("Geometry", "TessellationControl", "TessellationEvaluation"),
            Values("Output"), Values("%f32arr2", "%f32arr4"),
            Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    ClipAndCullDistanceVertexInput, ValidateVulkanCombineBuiltInArrayedVariable,
    Combine(Values("ClipDistance", "CullDistance"), Values("Fragment"),
            Values("Input"), Values("%f32arr4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit float array",
                              "components are not float scalar"))));

INSTANTIATE_TEST_SUITE_P(
    ClipAndCullDistanceNotArray, ValidateVulkanCombineBuiltInArrayedVariable,
    Combine(Values("ClipDistance", "CullDistance"),
            Values("Geometry", "TessellationControl", "TessellationEvaluation"),
            Values("Input"), Values("%f32vec2", "%f32vec4"),
            Values(TestResult(SPV_ERROR_INVALID_DATA,
                              "needs to be a 32-bit float array",
                              "components are not float scalar"))));

CodeGenerator GetWorkgroupSizeSuccessGenerator(spv_target_env env) {
  CodeGenerator generator =
      env == SPV_ENV_WEBGPU_0 ? CodeGenerator::GetWebGPUShaderCodeGenerator()
                              : CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = R"(
OpDecorate %workgroup_size BuiltIn WorkgroupSize
)";

  generator.after_types_ = R"(
%workgroup_size = OpConstantComposite %u32vec3 %u32_1 %u32_1 %u32_1
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "GLCompute";
  entry_point.body = R"(
%copy = OpCopyObject %u32vec3 %workgroup_size
)";
  generator.entry_points_.push_back(std::move(entry_point));

  return generator;
}

TEST_F(ValidateBuiltIns, VulkanWorkgroupSizeSuccess) {
  CodeGenerator generator =
      GetWorkgroupSizeSuccessGenerator(SPV_ENV_VULKAN_1_0);
  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_0));
}

TEST_F(ValidateBuiltIns, WebGPUWorkgroupSizeSuccess) {
  CodeGenerator generator = GetWorkgroupSizeSuccessGenerator(SPV_ENV_WEBGPU_0);
  CompileSuccessfully(generator.Build(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

CodeGenerator GetWorkgroupSizeFragmentGenerator(spv_target_env env) {
  CodeGenerator generator =
      env == SPV_ENV_WEBGPU_0 ? CodeGenerator::GetWebGPUShaderCodeGenerator()
                              : CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = R"(
OpDecorate %workgroup_size BuiltIn WorkgroupSize
)";

  generator.after_types_ = R"(
%workgroup_size = OpConstantComposite %u32vec3 %u32_1 %u32_1 %u32_1
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "Fragment";
  entry_point.execution_modes = "OpExecutionMode %main OriginUpperLeft";
  entry_point.body = R"(
%copy = OpCopyObject %u32vec3 %workgroup_size
)";
  generator.entry_points_.push_back(std::move(entry_point));

  return generator;
}

TEST_F(ValidateBuiltIns, VulkanWorkgroupSizeFragment) {
  CodeGenerator generator =
      GetWorkgroupSizeFragmentGenerator(SPV_ENV_VULKAN_1_0);

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vulkan spec allows BuiltIn WorkgroupSize to be used "
                        "only with GLCompute execution model"));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("is referencing ID <2> (OpConstantComposite) which is "
                        "decorated with BuiltIn WorkgroupSize in function <1> "
                        "called with execution model Fragment"));
}

TEST_F(ValidateBuiltIns, WebGPUWorkgroupSizeFragment) {
  CodeGenerator generator = GetWorkgroupSizeFragmentGenerator(SPV_ENV_WEBGPU_0);

  CompileSuccessfully(generator.Build(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("WebGPU spec allows BuiltIn WorkgroupSize to be used "
                        "only with GLCompute execution model"));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("is referencing ID <2> (OpConstantComposite) which is "
                        "decorated with BuiltIn WorkgroupSize in function <1> "
                        "called with execution model Fragment"));
}

TEST_F(ValidateBuiltIns, WorkgroupSizeNotConstant) {
  CodeGenerator generator = CodeGenerator::GetDefaultShaderCodeGenerator();
  generator.before_types_ = R"(
OpDecorate %copy BuiltIn WorkgroupSize
)";

  generator.after_types_ = R"(
%workgroup_size = OpConstantComposite %u32vec3 %u32_1 %u32_1 %u32_1
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "GLCompute";
  entry_point.body = R"(
%copy = OpCopyObject %u32vec3 %workgroup_size
)";
  generator.entry_points_.push_back(std::move(entry_point));

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vulkan spec requires BuiltIn WorkgroupSize to be a "
                        "constant. ID <2> (OpCopyObject) is not a constant"));
}

CodeGenerator GetWorkgroupSizeNotVectorGenerator(spv_target_env env) {
  CodeGenerator generator =
      env == SPV_ENV_WEBGPU_0 ? CodeGenerator::GetWebGPUShaderCodeGenerator()
                              : CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = R"(
OpDecorate %workgroup_size BuiltIn WorkgroupSize
)";

  generator.after_types_ = R"(
%workgroup_size = OpConstant %u32 16
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "GLCompute";
  entry_point.body = R"(
%copy = OpCopyObject %u32 %workgroup_size
)";
  generator.entry_points_.push_back(std::move(entry_point));

  return generator;
}

TEST_F(ValidateBuiltIns, VulkanWorkgroupSizeNotVector) {
  CodeGenerator generator =
      GetWorkgroupSizeNotVectorGenerator(SPV_ENV_VULKAN_1_0);

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("According to the Vulkan spec BuiltIn WorkgroupSize "
                        "variable needs to be a 3-component 32-bit int vector. "
                        "ID <2> (OpConstant) is not an int vector."));
}

TEST_F(ValidateBuiltIns, WebGPUWorkgroupSizeNotVector) {
  CodeGenerator generator =
      GetWorkgroupSizeNotVectorGenerator(SPV_ENV_WEBGPU_0);

  CompileSuccessfully(generator.Build(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("According to the WebGPU spec BuiltIn WorkgroupSize "
                        "variable needs to be a 3-component 32-bit int vector. "
                        "ID <2> (OpConstant) is not an int vector."));
}

CodeGenerator GetWorkgroupSizeNotIntVectorGenerator(spv_target_env env) {
  CodeGenerator generator =
      env == SPV_ENV_WEBGPU_0 ? CodeGenerator::GetWebGPUShaderCodeGenerator()
                              : CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = R"(
OpDecorate %workgroup_size BuiltIn WorkgroupSize
)";

  generator.after_types_ = R"(
%workgroup_size = OpConstantComposite %f32vec3 %f32_1 %f32_1 %f32_1
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "GLCompute";
  entry_point.body = R"(
%copy = OpCopyObject %f32vec3 %workgroup_size
)";
  generator.entry_points_.push_back(std::move(entry_point));

  return generator;
}

TEST_F(ValidateBuiltIns, VulkanWorkgroupSizeNotIntVector) {
  CodeGenerator generator =
      GetWorkgroupSizeNotIntVectorGenerator(SPV_ENV_VULKAN_1_0);

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("According to the Vulkan spec BuiltIn WorkgroupSize "
                        "variable needs to be a 3-component 32-bit int vector. "
                        "ID <2> (OpConstantComposite) is not an int vector."));
}

TEST_F(ValidateBuiltIns, WebGPUWorkgroupSizeNotIntVector) {
  CodeGenerator generator =
      GetWorkgroupSizeNotIntVectorGenerator(SPV_ENV_WEBGPU_0);

  CompileSuccessfully(generator.Build(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("According to the WebGPU spec BuiltIn WorkgroupSize "
                        "variable needs to be a 3-component 32-bit int vector. "
                        "ID <2> (OpConstantComposite) is not an int vector."));
}

CodeGenerator GetWorkgroupSizeNotVec3Generator(spv_target_env env) {
  CodeGenerator generator =
      env == SPV_ENV_WEBGPU_0 ? CodeGenerator::GetWebGPUShaderCodeGenerator()
                              : CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = R"(
OpDecorate %workgroup_size BuiltIn WorkgroupSize
)";

  generator.after_types_ = R"(
%workgroup_size = OpConstantComposite %u32vec2 %u32_1 %u32_1
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "GLCompute";
  entry_point.body = R"(
%copy = OpCopyObject %u32vec2 %workgroup_size
)";
  generator.entry_points_.push_back(std::move(entry_point));

  return generator;
}

TEST_F(ValidateBuiltIns, VulkanWorkgroupSizeNotVec3) {
  CodeGenerator generator =
      GetWorkgroupSizeNotVec3Generator(SPV_ENV_VULKAN_1_0);

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("According to the Vulkan spec BuiltIn WorkgroupSize "
                        "variable needs to be a 3-component 32-bit int vector. "
                        "ID <2> (OpConstantComposite) has 2 components."));
}

TEST_F(ValidateBuiltIns, WebGPUWorkgroupSizeNotVec3) {
  CodeGenerator generator = GetWorkgroupSizeNotVec3Generator(SPV_ENV_WEBGPU_0);

  CompileSuccessfully(generator.Build(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("According to the WebGPU spec BuiltIn WorkgroupSize "
                        "variable needs to be a 3-component 32-bit int vector. "
                        "ID <2> (OpConstantComposite) has 2 components."));
}

TEST_F(ValidateBuiltIns, WorkgroupSizeNotInt32Vec) {
  CodeGenerator generator = CodeGenerator::GetDefaultShaderCodeGenerator();
  generator.before_types_ = R"(
OpDecorate %workgroup_size BuiltIn WorkgroupSize
)";

  generator.after_types_ = R"(
%workgroup_size = OpConstantComposite %u64vec3 %u64_1 %u64_1 %u64_1
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "GLCompute";
  entry_point.body = R"(
%copy = OpCopyObject %u64vec3 %workgroup_size
)";
  generator.entry_points_.push_back(std::move(entry_point));

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("According to the Vulkan spec BuiltIn WorkgroupSize variable "
                "needs to be a 3-component 32-bit int vector. ID <2> "
                "(OpConstantComposite) has components with bit width 64."));
}

TEST_F(ValidateBuiltIns, WorkgroupSizePrivateVar) {
  CodeGenerator generator = CodeGenerator::GetDefaultShaderCodeGenerator();
  generator.before_types_ = R"(
OpDecorate %workgroup_size BuiltIn WorkgroupSize
)";

  generator.after_types_ = R"(
%workgroup_size = OpConstantComposite %u32vec3 %u32_1 %u32_1 %u32_1
%private_ptr_u32vec3 = OpTypePointer Private %u32vec3
%var = OpVariable %private_ptr_u32vec3 Private %workgroup_size
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "GLCompute";
  entry_point.body = R"(
)";
  generator.entry_points_.push_back(std::move(entry_point));

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_0));
}

TEST_F(ValidateBuiltIns, GeometryPositionInOutSuccess) {
  CodeGenerator generator = CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = R"(
OpMemberDecorate %input_type 0 BuiltIn Position
OpMemberDecorate %output_type 0 BuiltIn Position
)";

  generator.after_types_ = R"(
%input_type = OpTypeStruct %f32vec4
%arrayed_input_type = OpTypeArray %input_type %u32_3
%input_ptr = OpTypePointer Input %arrayed_input_type
%input = OpVariable %input_ptr Input
%input_f32vec4_ptr = OpTypePointer Input %f32vec4
%output_type = OpTypeStruct %f32vec4
%arrayed_output_type = OpTypeArray %output_type %u32_3
%output_ptr = OpTypePointer Output %arrayed_output_type
%output = OpVariable %output_ptr Output
%output_f32vec4_ptr = OpTypePointer Output %f32vec4
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "Geometry";
  entry_point.interfaces = "%input %output";
  entry_point.body = R"(
%input_pos = OpAccessChain %input_f32vec4_ptr %input %u32_0 %u32_0
%output_pos = OpAccessChain %output_f32vec4_ptr %output %u32_0 %u32_0
%pos = OpLoad %f32vec4 %input_pos
OpStore %output_pos %pos
)";
  generator.entry_points_.push_back(std::move(entry_point));
  generator.entry_points_[0].execution_modes =
      "OpExecutionMode %main InputPoints\nOpExecutionMode %main OutputPoints\n";

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_0));
}

TEST_F(ValidateBuiltIns, WorkgroupIdNotVec3) {
  CodeGenerator generator = CodeGenerator::GetDefaultShaderCodeGenerator();
  generator.before_types_ = R"(
OpDecorate %workgroup_size BuiltIn WorkgroupSize
OpDecorate %workgroup_id BuiltIn WorkgroupId
)";

  generator.after_types_ = R"(
%workgroup_size = OpConstantComposite %u32vec3 %u32_1 %u32_1 %u32_1
     %input_ptr = OpTypePointer Input %u32vec2
  %workgroup_id = OpVariable %input_ptr Input
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "GLCompute";
  entry_point.interfaces = "%workgroup_id";
  entry_point.body = R"(
%copy_size = OpCopyObject %u32vec3 %workgroup_size
  %load_id = OpLoad %u32vec2 %workgroup_id
)";
  generator.entry_points_.push_back(std::move(entry_point));

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("According to the Vulkan spec BuiltIn WorkgroupId "
                        "variable needs to be a 3-component 32-bit int vector. "
                        "ID <2> (OpVariable) has 2 components."));
}

TEST_F(ValidateBuiltIns, TwoBuiltInsFirstFails) {
  CodeGenerator generator = CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = R"(
OpMemberDecorate %input_type 0 BuiltIn FragCoord
OpMemberDecorate %output_type 0 BuiltIn Position
)";

  generator.after_types_ = R"(
%input_type = OpTypeStruct %f32vec4
%input_ptr = OpTypePointer Input %input_type
%input = OpVariable %input_ptr Input
%input_f32vec4_ptr = OpTypePointer Input %f32vec4
%output_type = OpTypeStruct %f32vec4
%output_ptr = OpTypePointer Output %output_type
%output = OpVariable %output_ptr Output
%output_f32vec4_ptr = OpTypePointer Output %f32vec4
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "Geometry";
  entry_point.interfaces = "%input %output";
  entry_point.body = R"(
%input_pos = OpAccessChain %input_f32vec4_ptr %input %u32_0
%output_pos = OpAccessChain %output_f32vec4_ptr %output %u32_0
%pos = OpLoad %f32vec4 %input_pos
OpStore %output_pos %pos
)";
  generator.entry_points_.push_back(std::move(entry_point));
  generator.entry_points_[0].execution_modes =
      "OpExecutionMode %main InputPoints\nOpExecutionMode %main OutputPoints\n";

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vulkan spec allows BuiltIn FragCoord to be used only "
                        "with Fragment execution model"));
}

TEST_F(ValidateBuiltIns, TwoBuiltInsSecondFails) {
  CodeGenerator generator = CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = R"(
OpMemberDecorate %input_type 0 BuiltIn Position
OpMemberDecorate %output_type 0 BuiltIn FragCoord
)";

  generator.after_types_ = R"(
%input_type = OpTypeStruct %f32vec4
%input_ptr = OpTypePointer Input %input_type
%input = OpVariable %input_ptr Input
%input_f32vec4_ptr = OpTypePointer Input %f32vec4
%output_type = OpTypeStruct %f32vec4
%output_ptr = OpTypePointer Output %output_type
%output = OpVariable %output_ptr Output
%output_f32vec4_ptr = OpTypePointer Output %f32vec4
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "Geometry";
  entry_point.interfaces = "%input %output";
  entry_point.body = R"(
%input_pos = OpAccessChain %input_f32vec4_ptr %input %u32_0
%output_pos = OpAccessChain %output_f32vec4_ptr %output %u32_0
%pos = OpLoad %f32vec4 %input_pos
OpStore %output_pos %pos
)";
  generator.entry_points_.push_back(std::move(entry_point));
  generator.entry_points_[0].execution_modes =
      "OpExecutionMode %main InputPoints\nOpExecutionMode %main OutputPoints\n";

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vulkan spec allows BuiltIn FragCoord to be only used "
                        "for variables with Input storage class"));
}

TEST_F(ValidateBuiltIns, VertexPositionVariableSuccess) {
  CodeGenerator generator = CodeGenerator::GetDefaultShaderCodeGenerator();
  generator.before_types_ = R"(
OpDecorate %position BuiltIn Position
)";

  generator.after_types_ = R"(
%f32vec4_ptr_output = OpTypePointer Output %f32vec4
%position = OpVariable %f32vec4_ptr_output Output
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "Vertex";
  entry_point.interfaces = "%position";
  entry_point.body = R"(
OpStore %position %f32vec4_0123
)";
  generator.entry_points_.push_back(std::move(entry_point));

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_0));
}

TEST_F(ValidateBuiltIns, FragmentPositionTwoEntryPoints) {
  CodeGenerator generator = CodeGenerator::GetDefaultShaderCodeGenerator();
  generator.before_types_ = R"(
OpMemberDecorate %output_type 0 BuiltIn Position
)";

  generator.after_types_ = R"(
%output_type = OpTypeStruct %f32vec4
%output_ptr = OpTypePointer Output %output_type
%output = OpVariable %output_ptr Output
%output_f32vec4_ptr = OpTypePointer Output %f32vec4
)";

  EntryPoint entry_point;
  entry_point.name = "vmain";
  entry_point.execution_model = "Vertex";
  entry_point.interfaces = "%output";
  entry_point.body = R"(
%val1 = OpFunctionCall %void %foo
)";
  generator.entry_points_.push_back(std::move(entry_point));

  entry_point.name = "fmain";
  entry_point.execution_model = "Fragment";
  entry_point.interfaces = "%output";
  entry_point.execution_modes = "OpExecutionMode %fmain OriginUpperLeft";
  entry_point.body = R"(
%val2 = OpFunctionCall %void %foo
)";
  generator.entry_points_.push_back(std::move(entry_point));

  generator.add_at_the_end_ = R"(
%foo = OpFunction %void None %func
%foo_entry = OpLabel
%position = OpAccessChain %output_f32vec4_ptr %output %u32_0
OpStore %position %f32vec4_0123
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vulkan spec allows BuiltIn Position to be used only "
                        "with Vertex, TessellationControl, "
                        "TessellationEvaluation or Geometry execution models"));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("called with execution model Fragment"));
}

CodeGenerator GetNoDepthReplacingGenerator(spv_target_env env) {
  CodeGenerator generator =
      spvIsWebGPUEnv(env) ? CodeGenerator::GetWebGPUShaderCodeGenerator()
                          : CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = R"(
OpMemberDecorate %output_type 0 BuiltIn FragDepth
)";

  generator.after_types_ = R"(
%output_type = OpTypeStruct %f32
%output_null = OpConstantNull %output_type
%output_ptr = OpTypePointer Output %output_type
%output = OpVariable %output_ptr Output %output_null
%output_f32_ptr = OpTypePointer Output %f32
)";

  EntryPoint entry_point;
  entry_point.name = "main";
  entry_point.execution_model = "Fragment";
  entry_point.interfaces = "%output";
  entry_point.execution_modes = "OpExecutionMode %main OriginUpperLeft";
  entry_point.body = R"(
%val2 = OpFunctionCall %void %foo
)";
  generator.entry_points_.push_back(std::move(entry_point));

  const std::string function_body = R"(
%foo = OpFunction %void None %func
%foo_entry = OpLabel
%frag_depth = OpAccessChain %output_f32_ptr %output %u32_0
OpStore %frag_depth %f32_1
OpReturn
OpFunctionEnd
)";

  if (spvIsWebGPUEnv(env)) {
    generator.after_types_ += function_body;
  } else {
    generator.add_at_the_end_ = function_body;
  }

  return generator;
}

TEST_F(ValidateBuiltIns, VulkanFragmentFragDepthNoDepthReplacing) {
  CodeGenerator generator = GetNoDepthReplacingGenerator(SPV_ENV_VULKAN_1_0);

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vulkan spec requires DepthReplacing execution mode to "
                        "be declared when using BuiltIn FragDepth"));
}

TEST_F(ValidateBuiltIns, WebGPUFragmentFragDepthNoDepthReplacing) {
  CodeGenerator generator = GetNoDepthReplacingGenerator(SPV_ENV_WEBGPU_0);

  CompileSuccessfully(generator.Build(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("WebGPU spec requires DepthReplacing execution mode to "
                        "be declared when using BuiltIn FragDepth"));
}

CodeGenerator GetOneMainHasDepthReplacingOtherHasntGenerator(
    spv_target_env env) {
  CodeGenerator generator =
      spvIsWebGPUEnv(env) ? CodeGenerator::GetWebGPUShaderCodeGenerator()
                          : CodeGenerator::GetDefaultShaderCodeGenerator();

  generator.before_types_ = R"(
OpMemberDecorate %output_type 0 BuiltIn FragDepth
)";

  generator.after_types_ = R"(
%output_type = OpTypeStruct %f32
%output_null = OpConstantNull %output_type
%output_ptr = OpTypePointer Output %output_type
%output = OpVariable %output_ptr Output %output_null
%output_f32_ptr = OpTypePointer Output %f32
)";

  EntryPoint entry_point;
  entry_point.name = "main_d_r";
  entry_point.execution_model = "Fragment";
  entry_point.interfaces = "%output";
  entry_point.execution_modes =
      "OpExecutionMode %main_d_r OriginUpperLeft\n"
      "OpExecutionMode %main_d_r DepthReplacing";
  entry_point.body = R"(
%val2 = OpFunctionCall %void %foo
)";
  generator.entry_points_.push_back(std::move(entry_point));

  entry_point.name = "main_no_d_r";
  entry_point.execution_model = "Fragment";
  entry_point.interfaces = "%output";
  entry_point.execution_modes = "OpExecutionMode %main_no_d_r OriginUpperLeft";
  entry_point.body = R"(
%val3 = OpFunctionCall %void %foo
)";
  generator.entry_points_.push_back(std::move(entry_point));

  const std::string function_body = R"(
%foo = OpFunction %void None %func
%foo_entry = OpLabel
%frag_depth = OpAccessChain %output_f32_ptr %output %u32_0
OpStore %frag_depth %f32_1
OpReturn
OpFunctionEnd
)";

  if (spvIsWebGPUEnv(env)) {
    generator.after_types_ += function_body;
  } else {
    generator.add_at_the_end_ = function_body;
  }

  return generator;
}

TEST_F(ValidateBuiltIns,
       VulkanFragmentFragDepthOneMainHasDepthReplacingOtherHasnt) {
  CodeGenerator generator =
      GetOneMainHasDepthReplacingOtherHasntGenerator(SPV_ENV_VULKAN_1_0);

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vulkan spec requires DepthReplacing execution mode to "
                        "be declared when using BuiltIn FragDepth"));
}

TEST_F(ValidateBuiltIns,
       WebGPUFragmentFragDepthOneMainHasDepthReplacingOtherHasnt) {
  CodeGenerator generator =
      GetOneMainHasDepthReplacingOtherHasntGenerator(SPV_ENV_WEBGPU_0);

  CompileSuccessfully(generator.Build(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("WebGPU spec requires DepthReplacing execution mode to "
                        "be declared when using BuiltIn FragDepth"));
}

TEST_F(ValidateBuiltIns, AllowInstanceIdWithIntersectionShader) {
  CodeGenerator generator = CodeGenerator::GetDefaultShaderCodeGenerator();
  generator.capabilities_ += R"(
OpCapability RayTracingNV
)";

  generator.extensions_ = R"(
OpExtension "SPV_NV_ray_tracing"
)";

  generator.before_types_ = R"(
OpMemberDecorate %input_type 0 BuiltIn InstanceId
)";

  generator.after_types_ = R"(
%input_type = OpTypeStruct %u32
%input_ptr = OpTypePointer Input %input_type
%input = OpVariable %input_ptr Input
)";

  EntryPoint entry_point;
  entry_point.name = "main_d_r";
  entry_point.execution_model = "IntersectionNV";
  entry_point.interfaces = "%input";
  entry_point.body = R"(
%val2 = OpFunctionCall %void %foo
)";
  generator.entry_points_.push_back(std::move(entry_point));

  generator.add_at_the_end_ = R"(
%foo = OpFunction %void None %func
%foo_entry = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  EXPECT_THAT(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_0));
}

TEST_F(ValidateBuiltIns, DisallowInstanceIdWithRayGenShader) {
  CodeGenerator generator = CodeGenerator::GetDefaultShaderCodeGenerator();
  generator.capabilities_ += R"(
OpCapability RayTracingNV
)";

  generator.extensions_ = R"(
OpExtension "SPV_NV_ray_tracing"
)";

  generator.before_types_ = R"(
OpMemberDecorate %input_type 0 BuiltIn InstanceId
)";

  generator.after_types_ = R"(
%input_type = OpTypeStruct %u32
%input_ptr = OpTypePointer Input %input_type
%input_ptr_u32 = OpTypePointer Input %u32
%input = OpVariable %input_ptr Input
)";

  EntryPoint entry_point;
  entry_point.name = "main_d_r";
  entry_point.execution_model = "RayGenerationNV";
  entry_point.interfaces = "%input";
  entry_point.body = R"(
%input_member = OpAccessChain %input_ptr_u32 %input %u32_0
)";
  generator.entry_points_.push_back(std::move(entry_point));

  CompileSuccessfully(generator.Build(), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Vulkan spec allows BuiltIn InstanceId to be used "
                        "only with IntersectionNV, ClosestHitNV and "
                        "AnyHitNV execution models"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
