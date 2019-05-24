// Copyright (c) 2018 Google Inc.
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

// Validation tests for memory/storage

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;

using ValidateMemory = spvtest::ValidateBase<bool>;

TEST_F(ValidateMemory, VulkanUniformConstantOnNonOpaqueResourceBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer UniformConstant %float
%2 = OpVariable %float_ptr UniformConstant
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("From Vulkan spec, section 14.5.2:\n"
                "Variables identified with the UniformConstant storage class "
                "are used only as handles to refer to opaque resources. Such "
                "variables must be typed as OpTypeImage, OpTypeSampler, "
                "OpTypeSampledImage, OpTypeAccelerationStructureNV, or an "
                "array of one of these types."));
}

TEST_F(ValidateMemory, VulkanUniformConstantOnOpaqueResourceGood) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %2 DescriptorSet 0
OpDecorate %2 Binding 0
%sampler = OpTypeSampler
%sampler_ptr = OpTypePointer UniformConstant %sampler
%2 = OpVariable %sampler_ptr UniformConstant
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, VulkanUniformConstantOnNonOpaqueResourceArrayBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%array_size = OpConstant %uint 5
%array = OpTypeArray %float %array_size
%array_ptr = OpTypePointer UniformConstant %array
%2 = OpVariable %array_ptr UniformConstant
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("From Vulkan spec, section 14.5.2:\n"
                "Variables identified with the UniformConstant storage class "
                "are used only as handles to refer to opaque resources. Such "
                "variables must be typed as OpTypeImage, OpTypeSampler, "
                "OpTypeSampledImage, OpTypeAccelerationStructureNV, or an "
                "array of one of these types."));
}

TEST_F(ValidateMemory, VulkanUniformConstantOnOpaqueResourceArrayGood) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %2 DescriptorSet 0
OpDecorate %2 Binding 0
%sampler = OpTypeSampler
%uint = OpTypeInt 32 0
%array_size = OpConstant %uint 5
%array = OpTypeArray %sampler %array_size
%array_ptr = OpTypePointer UniformConstant %array
%2 = OpVariable %array_ptr UniformConstant
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, VulkanUniformConstantOnOpaqueResourceRuntimeArrayGood) {
  std::string spirv = R"(
OpCapability RuntimeDescriptorArrayEXT
OpCapability Shader
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %2 DescriptorSet 0
OpDecorate %2 Binding 0
%sampler = OpTypeSampler
%uint = OpTypeInt 32 0
%array = OpTypeRuntimeArray %sampler
%array_ptr = OpTypePointer UniformConstant %array
%2 = OpVariable %array_ptr UniformConstant
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, VulkanUniformOnIntBad) {
  char src[] = R"(
            OpCapability Shader
            OpMemoryModel Logical GLSL450
            OpEntryPoint GLCompute %kernel "main"
            OpExecutionMode %kernel LocalSize 1 1 1

            OpDecorate %var DescriptorSet 0
            OpDecorate %var Binding 0

  %voidty = OpTypeVoid
%kernelty = OpTypeFunction %voidty
   %intty = OpTypeInt 32 0
   %varty = OpTypePointer Uniform %intty
   %value = OpConstant %intty 42

     %var = OpVariable %varty Uniform

  %kernel = OpFunction %voidty None %kernelty
   %label = OpLabel
            OpStore %var %value
            OpReturn
            OpFunctionEnd
)";
  CompileSuccessfully(src, SPV_ENV_VULKAN_1_1);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("From Vulkan spec, section 14.5.2:\n"
                "Variables identified with the Uniform storage class are used "
                "to access transparent buffer backed resources. Such variables "
                "must be typed as OpTypeStruct, or an array of this type"));
}

// #version 440
// #extension GL_EXT_nonuniform_qualifier : enable
// layout(binding = 1) uniform sampler2D s2d[][2];
// layout(location = 0) in nonuniformEXT int i;
// void main()
// {
//     vec4 v = texture(s2d[i][i], vec2(0.3));
// }
TEST_F(ValidateMemory, VulkanUniformOnRuntimeArrayOfArrayBad) {
  char src[] = R"(
               OpCapability Shader
               OpCapability ShaderNonUniformEXT
               OpCapability RuntimeDescriptorArrayEXT
               OpCapability SampledImageArrayNonUniformIndexingEXT
               OpExtension "SPV_EXT_descriptor_indexing"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %i
               OpSource GLSL 440
               OpSourceExtension "GL_EXT_nonuniform_qualifier"
               OpName %main "main"
               OpName %v "v"
               OpName %s2d "s2d"
               OpName %i "i"
               OpDecorate %s2d DescriptorSet 0
               OpDecorate %s2d Binding 1
               OpDecorate %i Location 0
               OpDecorate %i NonUniformEXT
               OpDecorate %21 NonUniformEXT
               OpDecorate %22 NonUniformEXT
               OpDecorate %25 NonUniformEXT
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
         %11 = OpTypeSampledImage %10
       %uint = OpTypeInt 32 0
     %uint_2 = OpConstant %uint 2
%_arr_11_uint_2 = OpTypeArray %11 %uint_2
%_runtimearr__arr_11_uint_2 = OpTypeRuntimeArray %_arr_11_uint_2
%_ptr_Uniform__runtimearr__arr_11_uint_2 = OpTypePointer Uniform %_runtimearr__arr_11_uint_2
        %s2d = OpVariable %_ptr_Uniform__runtimearr__arr_11_uint_2 Uniform
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
          %i = OpVariable %_ptr_Input_int Input
%_ptr_Uniform_11 = OpTypePointer Uniform %11
    %v2float = OpTypeVector %float 2
%float_0_300000012 = OpConstant %float 0.300000012
         %28 = OpConstantComposite %v2float %float_0_300000012 %float_0_300000012
    %float_0 = OpConstant %float 0
       %main = OpFunction %void None %3
          %5 = OpLabel
          %v = OpVariable %_ptr_Function_v4float Function
         %21 = OpLoad %int %i
         %22 = OpLoad %int %i
         %24 = OpAccessChain %_ptr_Uniform_11 %s2d %21 %22
         %25 = OpLoad %11 %24
         %30 = OpImageSampleExplicitLod %v4float %25 %28 Lod %float_0
               OpStore %v %30
               OpReturn
               OpFunctionEnd
)";
  CompileSuccessfully(src, SPV_ENV_VULKAN_1_1);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("From Vulkan spec, section 14.5.2:\n"
                "Variables identified with the Uniform storage class are used "
                "to access transparent buffer backed resources. Such variables "
                "must be typed as OpTypeStruct, or an array of this type"));
}

// #version 440
// layout (set=1, binding=1) uniform sampler2D variableName[2][2];
// void main() {
// }
TEST_F(ValidateMemory, VulkanUniformOnArrayOfArrayBad) {
  char src[] = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource GLSL 440
               OpName %main "main"
               OpName %variableName "variableName"
               OpDecorate %variableName DescriptorSet 1
               OpDecorate %variableName Binding 1
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
          %7 = OpTypeImage %float 2D 0 0 0 1 Unknown
          %8 = OpTypeSampledImage %7
       %uint = OpTypeInt 32 0
     %uint_2 = OpConstant %uint 2
%_arr_8_uint_2 = OpTypeArray %8 %uint_2
%_arr__arr_8_uint_2_uint_2 = OpTypeArray %_arr_8_uint_2 %uint_2
%_ptr_Uniform__arr__arr_8_uint_2_uint_2 = OpTypePointer Uniform %_arr__arr_8_uint_2_uint_2
%variableName = OpVariable %_ptr_Uniform__arr__arr_8_uint_2_uint_2 Uniform
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  CompileSuccessfully(src, SPV_ENV_VULKAN_1_1);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("From Vulkan spec, section 14.5.2:\n"
                "Variables identified with the Uniform storage class are used "
                "to access transparent buffer backed resources. Such variables "
                "must be typed as OpTypeStruct, or an array of this type"));
}

TEST_F(ValidateMemory, MismatchingStorageClassesBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Uniform %float
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
%2 = OpVariable %float_ptr Function
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "From SPIR-V spec, section 3.32.8 on OpVariable:\n"
          "Its Storage Class operand must be the same as the Storage Class "
          "operand of the result type."));
}

TEST_F(ValidateMemory, MatchingStorageClassesGood) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Function %float
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
%2 = OpVariable %float_ptr Function
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateMemory, WebGPUInitializerWithOutputStorageClassesGood) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Output %float
%init_val = OpConstant %float 1.0
%1 = OpVariable %float_ptr Output %init_val
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%2 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateMemory, WebGPUInitializerWithFunctionStorageClassesGood) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Function %float
%init_val = OpConstant %float 1.0
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
%2 = OpVariable %float_ptr Function %init_val
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateMemory, WebGPUInitializerWithPrivateStorageClassesGood) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Private %float
%init_val = OpConstant %float 1.0
%1 = OpVariable %float_ptr Private %init_val
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%2 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateMemory, WebGPUInitializerWithDisallowedStorageClassesBad) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Uniform %float
%init_val = OpConstant %float 1.0
%1 = OpVariable %float_ptr Uniform %init_val
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%2 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpVariable, <id> '5[%5]', has a disallowed initializer & "
                "storage class combination.\nFrom WebGPU spec:\nVariable "
                "declarations that include initializers must have one of the "
                "following storage classes: Output, Private, or Function\n  %5 "
                "= OpVariable %_ptr_Uniform_float Uniform %float_1\n"));
}

TEST_F(ValidateMemory, WebGPUOutputStorageClassWithoutInitializerBad) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Output %float
%1 = OpVariable %float_ptr Output
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%2 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpVariable, <id> '4[%4]', must have an initializer.\n"
                "From WebGPU execution environment spec:\n"
                "All variables in the following storage classes must have an "
                "initializer: Output, Private, or Function\n"
                "  %4 = OpVariable %_ptr_Output_float Output\n"));
}

TEST_F(ValidateMemory, WebGPUFunctionStorageClassWithoutInitializerBad) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Function %float
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
%2 = OpVariable %float_ptr Function
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpVariable, <id> '7[%7]', must have an initializer.\n"
                "From WebGPU execution environment spec:\n"
                "All variables in the following storage classes must have an "
                "initializer: Output, Private, or Function\n"
                "  %7 = OpVariable %_ptr_Function_float Function\n"));
}

TEST_F(ValidateMemory, WebGPUPrivateStorageClassWithoutInitializerBad) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Private %float
%1 = OpVariable %float_ptr Private
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%2 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpVariable, <id> '4[%4]', must have an initializer.\n"
                "From WebGPU execution environment spec:\n"
                "All variables in the following storage classes must have an "
                "initializer: Output, Private, or Function\n"
                "  %4 = OpVariable %_ptr_Private_float Private\n"));
}

TEST_F(ValidateMemory, VulkanInitializerWithOutputStorageClassesGood) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Output %float
%init_val = OpConstant %float 1.0
%1 = OpVariable %float_ptr Output %init_val
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%2 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, VulkanInitializerWithFunctionStorageClassesGood) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Function %float
%init_val = OpConstant %float 1.0
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
%2 = OpVariable %float_ptr Function %init_val
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, VulkanInitializerWithPrivateStorageClassesGood) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Private %float
%init_val = OpConstant %float 1.0
%1 = OpVariable %float_ptr Private %init_val
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%2 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, VulkanInitializerWithDisallowedStorageClassesBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer Input %float
%init_val = OpConstant %float 1.0
%1 = OpVariable %float_ptr Input %init_val
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%2 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpVariable, <id> '5[%5]', has a disallowed initializer & "
                "storage class combination.\nFrom Vulkan spec:\nVariable "
                "declarations that include initializers must have one of the "
                "following storage classes: Output, Private, or Function\n  %5 "
                "= OpVariable %_ptr_Input_float Input %float_1\n"));
}

TEST_F(ValidateMemory, ArrayLenCorrectResultType) {
  std::string spirv = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpExecutionMode %1 OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
     %uint = OpTypeInt 32 0
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %_runtimearr_float
%_ptr_Function__struct_7 = OpTypePointer Function %_struct_7
          %1 = OpFunction %void None %3
          %9 = OpLabel
         %10 = OpVariable %_ptr_Function__struct_7 Function
         %11 = OpArrayLength %uint %10 0
               OpReturn
               OpFunctionEnd

)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateMemory, ArrayLenIndexCorrectWith2Members) {
  std::string spirv = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpExecutionMode %1 OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
     %uint = OpTypeInt 32 0
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %float %_runtimearr_float
%_ptr_Function__struct_7  = OpTypePointer Function %_struct_7
          %1 = OpFunction %void None %3
          %9 = OpLabel
         %10 = OpVariable %_ptr_Function__struct_7  Function
         %11 = OpArrayLength %uint %10 1
               OpReturn
               OpFunctionEnd

)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateMemory, ArrayLenResultNotIntType) {
  std::string spirv = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpExecutionMode %1 OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_6 = OpTypeStruct %_runtimearr_float
%_ptr_Function__struct_6 = OpTypePointer Function %_struct_6
          %1 = OpFunction %void None %3
          %8 = OpLabel
          %9 = OpVariable %_ptr_Function__struct_6 Function
         %10 = OpArrayLength %float %9 0
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "The Result Type of OpArrayLength <id> '10[%10]' must be OpTypeInt "
          "with width 32 and signedness 0.\n  %10 = OpArrayLength %float %9 "
          "0\n"));
}

TEST_F(ValidateMemory, ArrayLenResultNot32bits) {
  std::string spirv = R"(
               OpCapability Shader
               OpCapability Int16
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpExecutionMode %1 OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
     %ushort = OpTypeInt 16 0
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %_runtimearr_float
%_ptr_Function__struct_7 = OpTypePointer Function %_struct_7
          %1 = OpFunction %void None %3
          %9 = OpLabel
         %10 = OpVariable %_ptr_Function__struct_7 Function
         %11 = OpArrayLength %ushort %10 0
               OpReturn
               OpFunctionEnd

)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "The Result Type of OpArrayLength <id> '11[%11]' must be OpTypeInt "
          "with width 32 and signedness 0.\n  %11 = OpArrayLength %ushort %10 "
          "0\n"));
}

TEST_F(ValidateMemory, ArrayLenResultSigned) {
  std::string spirv = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpExecutionMode %1 OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
     %int = OpTypeInt 32 1
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %_runtimearr_float
%_ptr_Function__struct_7 = OpTypePointer Function %_struct_7
          %1 = OpFunction %void None %3
          %9 = OpLabel
         %10 = OpVariable %_ptr_Function__struct_7 Function
         %11 = OpArrayLength %int %10 0
               OpReturn
               OpFunctionEnd

)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "The Result Type of OpArrayLength <id> '11[%11]' must be OpTypeInt "
          "with width 32 and signedness 0.\n  %11 = OpArrayLength %int %10 "
          "0\n"));
}

TEST_F(ValidateMemory, ArrayLenInputNotStruct) {
  std::string spirv = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpExecutionMode %1 OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
     %uint = OpTypeInt 32 0
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %_runtimearr_float
%_ptr_Function_float = OpTypePointer Function %float
          %1 = OpFunction %void None %3
          %9 = OpLabel
         %10 = OpVariable %_ptr_Function_float Function
         %11 = OpArrayLength %uint %10 0
               OpReturn
               OpFunctionEnd

)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The Struture's type in OpArrayLength <id> '11[%11]' "
                        "must be a pointer to an OpTypeStruct."));
}

TEST_F(ValidateMemory, ArrayLenInputLastMemberNoRTA) {
  std::string spirv = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpExecutionMode %1 OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
     %uint = OpTypeInt 32 0
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %float
%_ptr_Function__struct_7  = OpTypePointer Function %_struct_7
          %1 = OpFunction %void None %3
          %9 = OpLabel
         %10 = OpVariable %_ptr_Function__struct_7  Function
         %11 = OpArrayLength %uint %10 0
               OpReturn
               OpFunctionEnd

)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("The Struture's last member in OpArrayLength <id> '11[%11]' "
                "must be an OpTypeRuntimeArray.\n  %11 = OpArrayLength %uint "
                "%10 0\n"));
}

TEST_F(ValidateMemory, ArrayLenInputLastMemberNoRTA2) {
  std::string spirv = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpExecutionMode %1 OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
     %uint = OpTypeInt 32 0
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %_runtimearr_float %float
%_ptr_Function__struct_7  = OpTypePointer Function %_struct_7
          %1 = OpFunction %void None %3
          %9 = OpLabel
         %10 = OpVariable %_ptr_Function__struct_7  Function
         %11 = OpArrayLength %uint %10 1
               OpReturn
               OpFunctionEnd

)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("The Struture's last member in OpArrayLength <id> '11[%11]' "
                "must be an OpTypeRuntimeArray.\n  %11 = OpArrayLength %uint "
                "%10 1\n"));
}

TEST_F(ValidateMemory, ArrayLenIndexNotLastMember) {
  std::string spirv = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpExecutionMode %1 OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
     %uint = OpTypeInt 32 0
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %float %_runtimearr_float
%_ptr_Function__struct_7  = OpTypePointer Function %_struct_7
          %1 = OpFunction %void None %3
          %9 = OpLabel
         %10 = OpVariable %_ptr_Function__struct_7  Function
         %11 = OpArrayLength %uint %10 0
               OpReturn
               OpFunctionEnd

)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "The array member in OpArrayLength <id> '11[%11]' must be an the "
          "last member of the struct.\n  %11 = OpArrayLength %uint %10 0\n"));
}

TEST_F(ValidateMemory, ArrayLenIndexNotPointerToStruct) {
  std::string spirv = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpExecutionMode %1 OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
     %uint = OpTypeInt 32 0
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %float %_runtimearr_float
%_ptr_Function__struct_7  = OpTypePointer Function %_struct_7
          %1 = OpFunction %void None %3
          %9 = OpLabel
         %10 = OpVariable %_ptr_Function__struct_7  Function
         %11 = OpLoad %_struct_7 %10
         %12 = OpArrayLength %uint %11 0
               OpReturn
               OpFunctionEnd

)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "The Struture's type in OpArrayLength <id> '12[%12]' must be a "
          "pointer to an OpTypeStruct.\n  %12 = OpArrayLength %uint %11 0\n"));
}

TEST_F(ValidateMemory, ArrayLenPointerIsAType) {
  std::string spirv = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpExecutionMode %1 OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
          %1 = OpFunction %void None %3
          %9 = OpLabel
         %12 = OpArrayLength %uint %float 0
               OpReturn
               OpFunctionEnd

)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Operand 4[%float] cannot be a "
                                               "type"));
}

TEST_F(ValidateMemory, PushConstantNotStructGood) {
  std::string spirv = R"(
            OpCapability Shader
            OpMemoryModel Logical GLSL450
            OpEntryPoint Fragment %1 "main"
            OpExecutionMode %1 OriginUpperLeft

    %void = OpTypeVoid
  %voidfn = OpTypeFunction %void
   %float = OpTypeFloat 32
     %ptr = OpTypePointer PushConstant %float
      %pc = OpVariable %ptr PushConstant

       %1 = OpFunction %void None %voidfn
   %label = OpLabel
            OpReturn
            OpFunctionEnd
)";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateMemory, VulkanPushConstantNotStructBad) {
  std::string spirv = R"(
            OpCapability Shader
            OpMemoryModel Logical GLSL450
            OpEntryPoint Fragment %1 "main"
            OpExecutionMode %1 OriginUpperLeft

    %void = OpTypeVoid
  %voidfn = OpTypeFunction %void
   %float = OpTypeFloat 32
     %ptr = OpTypePointer PushConstant %float
      %pc = OpVariable %ptr PushConstant

       %1 = OpFunction %void None %voidfn
   %label = OpLabel
            OpReturn
            OpFunctionEnd
)";
  CompileSuccessfully(spirv, SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("PushConstant OpVariable <id> '6[%6]' has illegal "
                        "type.\nFrom Vulkan spec, section 14.5.1:\n"
                        "Such variables must be typed as OpTypeStruct, "
                        "or an array of this type"));
}

TEST_F(ValidateMemory, VulkanPushConstant) {
  std::string spirv = R"(
            OpCapability Shader
            OpMemoryModel Logical GLSL450
            OpEntryPoint Fragment %1 "main"
            OpExecutionMode %1 OriginUpperLeft

            OpDecorate %struct Block
            OpMemberDecorate %struct 0 Offset 0

    %void = OpTypeVoid
  %voidfn = OpTypeFunction %void
   %float = OpTypeFloat 32
  %struct = OpTypeStruct %float
     %ptr = OpTypePointer PushConstant %struct
      %pc = OpVariable %ptr PushConstant

       %1 = OpFunction %void None %voidfn
   %label = OpLabel
            OpReturn
            OpFunctionEnd
)";
  CompileSuccessfully(spirv, SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeLoadBad1) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
%load = OpLoad %int %var MakePointerVisibleKHR|NonPrivatePointerKHR %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeLoadBad2) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
%load = OpLoad %int %var Aligned|MakePointerVisibleKHR|NonPrivatePointerKHR 4 %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeLoadGood1) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
%load = OpLoad %int %var MakePointerVisibleKHR|NonPrivatePointerKHR %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeLoadGood2) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
%load = OpLoad %int %var Aligned|MakePointerVisibleKHR|NonPrivatePointerKHR 4 %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeStoreBad1) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpStore %var %device MakePointerAvailableKHR|NonPrivatePointerKHR %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeStoreBad2) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpStore %var %device Aligned|MakePointerAvailableKHR|NonPrivatePointerKHR 4 %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeStoreGood1) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpStore %var %device MakePointerAvailableKHR|NonPrivatePointerKHR %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeStoreGood2) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpStore %var %device Aligned|MakePointerAvailableKHR|NonPrivatePointerKHR 4 %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeCopyMemoryBad1) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var1 = OpVariable %int_ptr_ssbo StorageBuffer
%var2 = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpCopyMemory %var1 %var2 MakePointerAvailableKHR|NonPrivatePointerKHR %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeCopyMemoryBad2) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%workgroup = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var1 = OpVariable %int_ptr_ssbo StorageBuffer
%var2 = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpCopyMemory %var1 %var2 Aligned|MakePointerVisibleKHR|MakePointerAvailableKHR|NonPrivatePointerKHR 4 %device %workgroup
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeCopyMemoryBad3) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%workgroup = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var1 = OpVariable %int_ptr_ssbo StorageBuffer
%var2 = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpCopyMemory %var1 %var2 Aligned|MakePointerVisibleKHR|MakePointerAvailableKHR|NonPrivatePointerKHR 4 %workgroup %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeCopyMemoryGood1) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var1 = OpVariable %int_ptr_ssbo StorageBuffer
%var2 = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpCopyMemory %var1 %var2 MakePointerAvailableKHR|NonPrivatePointerKHR %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeCopyMemoryGood2) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%workgroup = OpConstant %int 2
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var1 = OpVariable %int_ptr_ssbo StorageBuffer
%var2 = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpCopyMemory %var1 %var2 Aligned|MakePointerVisibleKHR|MakePointerAvailableKHR|NonPrivatePointerKHR 4 %device %workgroup
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeCopyMemoryGood3) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%workgroup = OpConstant %int 2
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var1 = OpVariable %int_ptr_ssbo StorageBuffer
%var2 = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpCopyMemory %var1 %var2 Aligned|MakePointerVisibleKHR|MakePointerAvailableKHR|NonPrivatePointerKHR 4 %workgroup %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeCopyMemorySizedBad1) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpCapability Addresses
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var1 = OpVariable %int_ptr_ssbo StorageBuffer
%var2 = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpCopyMemorySized %var1 %var2 %device MakePointerAvailableKHR|NonPrivatePointerKHR %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeCopyMemorySizedBad2) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpCapability Addresses
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%workgroup = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var1 = OpVariable %int_ptr_ssbo StorageBuffer
%var2 = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpCopyMemorySized %var1 %var2 %device Aligned|MakePointerVisibleKHR|MakePointerAvailableKHR|NonPrivatePointerKHR 4 %device %workgroup
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeCopyMemorySizedBad3) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpCapability Addresses
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%workgroup = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var1 = OpVariable %int_ptr_ssbo StorageBuffer
%var2 = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpCopyMemorySized %var1 %var2 %device Aligned|MakePointerVisibleKHR|MakePointerAvailableKHR|NonPrivatePointerKHR 4 %workgroup %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeCopyMemorySizedGood1) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpCapability Linkage
OpCapability Addresses
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var1 = OpVariable %int_ptr_ssbo StorageBuffer
%var2 = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpCopyMemorySized %var1 %var2 %device MakePointerAvailableKHR|NonPrivatePointerKHR %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeCopyMemorySizedGood2) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpCapability Linkage
OpCapability Addresses
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%workgroup = OpConstant %int 2
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var1 = OpVariable %int_ptr_ssbo StorageBuffer
%var2 = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpCopyMemorySized %var1 %var2 %device Aligned|MakePointerVisibleKHR|MakePointerAvailableKHR|NonPrivatePointerKHR 4 %device %workgroup
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateMemory, VulkanMemoryModelDeviceScopeCopyMemorySizedGood3) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpCapability Linkage
OpCapability Addresses
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%void = OpTypeVoid
%int = OpTypeInt 32 0
%device = OpConstant %int 1
%workgroup = OpConstant %int 2
%int_ptr_ssbo = OpTypePointer StorageBuffer %int
%var1 = OpVariable %int_ptr_ssbo StorageBuffer
%var2 = OpVariable %int_ptr_ssbo StorageBuffer
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
OpCopyMemorySized %var1 %var2 %device Aligned|MakePointerVisibleKHR|MakePointerAvailableKHR|NonPrivatePointerKHR 4 %workgroup %device
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateMemory, ArrayLengthStructIsLabel) {
  const std::string spirv = R"(
OpCapability Tessellation
OpMemoryModel Logical GLSL450
OpName %20 "incorrect"
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%4 = OpFunction %void None %3
%20 = OpLabel
%24 = OpArrayLength %uint %20 0
%25 = OpLoad %v4float %24
OpReturnValue %25
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Operand 1[%incorrect] requires a type"));
}

TEST_F(ValidateMemory, PSBLoadAlignedSuccess) {
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
%ptr = OpTypePointer PhysicalStorageBufferEXT %uint64
%pptr_f = OpTypePointer Function %ptr
%void = OpTypeVoid
%voidfn = OpTypeFunction %void
%main = OpFunction %void None %voidfn
%entry = OpLabel
%val1 = OpVariable %pptr_f Function
%val2 = OpLoad %ptr %val1
%val3 = OpLoad %uint64 %val2 Aligned 8
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateMemory, PSBLoadAlignedMissing) {
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
%ptr = OpTypePointer PhysicalStorageBufferEXT %uint64
%pptr_f = OpTypePointer Function %ptr
%void = OpTypeVoid
%voidfn = OpTypeFunction %void
%main = OpFunction %void None %voidfn
%entry = OpLabel
%val1 = OpVariable %pptr_f Function
%val2 = OpLoad %ptr %val1
%val3 = OpLoad %uint64 %val2
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Memory accesses with PhysicalStorageBufferEXT must use Aligned"));
}

TEST_F(ValidateMemory, PSBStoreAlignedSuccess) {
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
%u64_1 = OpConstant %uint64 1
%ptr = OpTypePointer PhysicalStorageBufferEXT %uint64
%pptr_f = OpTypePointer Function %ptr
%void = OpTypeVoid
%voidfn = OpTypeFunction %void
%main = OpFunction %void None %voidfn
%entry = OpLabel
%val1 = OpVariable %pptr_f Function
%val2 = OpLoad %ptr %val1
OpStore %val2 %u64_1 Aligned 8
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateMemory, PSBStoreAlignedMissing) {
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
%u64_1 = OpConstant %uint64 1
%ptr = OpTypePointer PhysicalStorageBufferEXT %uint64
%pptr_f = OpTypePointer Function %ptr
%void = OpTypeVoid
%voidfn = OpTypeFunction %void
%main = OpFunction %void None %voidfn
%entry = OpLabel
%val1 = OpVariable %pptr_f Function
%val2 = OpLoad %ptr %val1
OpStore %val2 %u64_1 None
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Memory accesses with PhysicalStorageBufferEXT must use Aligned"));
}

TEST_F(ValidateMemory, PSBVariable) {
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
%ptr = OpTypePointer PhysicalStorageBufferEXT %uint64
%val1 = OpVariable %ptr PhysicalStorageBufferEXT
%void = OpTypeVoid
%voidfn = OpTypeFunction %void
%main = OpFunction %void None %voidfn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("PhysicalStorageBufferEXT must not be used with OpVariable"));
}

std::string GenCoopMatLoadStoreShader(const std::string& storeMemoryAccess,
                                      const std::string& loadMemoryAccess) {
  std::string s = R"(
OpCapability Shader
OpCapability GroupNonUniform
OpCapability VulkanMemoryModelKHR
OpCapability CooperativeMatrixNV
OpExtension "SPV_KHR_vulkan_memory_model"
OpExtension "SPV_NV_cooperative_matrix"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical VulkanKHR
OpEntryPoint GLCompute %4 "main" %11 %21
OpExecutionMode %4 LocalSize 1 1 1
OpDecorate %11 BuiltIn SubgroupId
OpDecorate %21 BuiltIn WorkgroupId
OpDecorate %74 ArrayStride 4
OpMemberDecorate %75 0 Offset 0
OpDecorate %75 Block
OpDecorate %77 DescriptorSet 0
OpDecorate %77 Binding 0
OpDecorate %92 ArrayStride 4
OpMemberDecorate %93 0 Offset 0
OpDecorate %93 Block
OpDecorate %95 DescriptorSet 0
OpDecorate %95 Binding 1
OpDecorate %102 ArrayStride 4
OpMemberDecorate %103 0 Offset 0
OpDecorate %103 Block
OpDecorate %105 DescriptorSet 0
OpDecorate %105 Binding 2
OpDecorate %117 ArrayStride 4
OpMemberDecorate %118 0 Offset 0
OpDecorate %118 Block
OpDecorate %120 DescriptorSet 0
OpDecorate %120 Binding 3
OpDecorate %123 SpecId 2
OpDecorate %124 SpecId 3
OpDecorate %125 SpecId 4
OpDecorate %126 SpecId 5
OpDecorate %127 SpecId 0
OpDecorate %128 SpecId 1
OpDecorate %129 BuiltIn WorkgroupSize
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%6 = OpTypeInt 32 0
%7 = OpTypeVector %6 2
%8 = OpTypePointer Function %7
%10 = OpTypePointer Input %6
%11 = OpVariable %10 Input
%13 = OpConstant %6 2
%19 = OpTypeVector %6 3
%20 = OpTypePointer Input %19
%21 = OpVariable %20 Input
%27 = OpConstantComposite %7 %13 %13
%31 = OpTypePointer Function %6
%33 = OpConstant %6 1024
%34 = OpConstant %6 1
%38 = OpConstant %6 8
%39 = OpConstant %6 0
%68 = OpTypeFloat 32
%69 = OpConstant %6 16
%70 = OpConstant %6 3
%71 = OpTypeCooperativeMatrixNV %68 %70 %69 %38
%72 = OpTypePointer Function %71
%74 = OpTypeRuntimeArray %68
%75 = OpTypeStruct %74
%76 = OpTypePointer StorageBuffer %75
%77 = OpVariable %76 StorageBuffer
%78 = OpTypeInt 32 1
%79 = OpConstant %78 0
%81 = OpConstant %6 5
%82 = OpTypePointer StorageBuffer %68
%84 = OpConstant %6 64
%85 = OpTypeBool
%86 = OpConstantFalse %85
%88 = OpTypePointer Private %71
%89 = OpVariable %88 Private
%92 = OpTypeRuntimeArray %68
%93 = OpTypeStruct %92
%94 = OpTypePointer StorageBuffer %93
%95 = OpVariable %94 StorageBuffer
%99 = OpVariable %88 Private
%102 = OpTypeRuntimeArray %68
%103 = OpTypeStruct %102
%104 = OpTypePointer StorageBuffer %103
%105 = OpVariable %104 StorageBuffer
%109 = OpVariable %88 Private
%111 = OpVariable %88 Private
%112 = OpSpecConstantOp %6 CooperativeMatrixLengthNV %71
%113 = OpSpecConstantOp %78 IAdd %112 %79
%117 = OpTypeRuntimeArray %68
%118 = OpTypeStruct %117
%119 = OpTypePointer StorageBuffer %118
%120 = OpVariable %119 StorageBuffer
%123 = OpSpecConstant %78 1
%124 = OpSpecConstant %78 1
%125 = OpSpecConstant %78 1
%126 = OpSpecConstant %78 1
%127 = OpSpecConstant %6 1
%128 = OpSpecConstant %6 1
%129 = OpSpecConstantComposite %19 %127 %128 %34
%4 = OpFunction %2 None %3
%5 = OpLabel
%9 = OpVariable %8 Function
%18 = OpVariable %8 Function
%32 = OpVariable %31 Function
%44 = OpVariable %31 Function
%52 = OpVariable %31 Function
%60 = OpVariable %31 Function
%73 = OpVariable %72 Function
%91 = OpVariable %72 Function
%101 = OpVariable %72 Function
%12 = OpLoad %6 %11
%14 = OpUMod %6 %12 %13
%15 = OpLoad %6 %11
%16 = OpUDiv %6 %15 %13
%17 = OpCompositeConstruct %7 %14 %16
OpStore %9 %17
%22 = OpLoad %19 %21
%23 = OpVectorShuffle %7 %22 %22 0 1
%24 = OpCompositeExtract %6 %23 0
%25 = OpCompositeExtract %6 %23 1
%26 = OpCompositeConstruct %7 %24 %25
%28 = OpIMul %7 %26 %27
%29 = OpLoad %7 %9
%30 = OpIAdd %7 %28 %29
OpStore %18 %30
%35 = OpAccessChain %31 %18 %34
%36 = OpLoad %6 %35
%37 = OpIMul %6 %33 %36
%40 = OpAccessChain %31 %18 %39
%41 = OpLoad %6 %40
%42 = OpIMul %6 %38 %41
%43 = OpIAdd %6 %37 %42
OpStore %32 %43
%45 = OpAccessChain %31 %18 %34
%46 = OpLoad %6 %45
%47 = OpIMul %6 %33 %46
%48 = OpAccessChain %31 %18 %39
%49 = OpLoad %6 %48
%50 = OpIMul %6 %38 %49
%51 = OpIAdd %6 %47 %50
OpStore %44 %51
%53 = OpAccessChain %31 %18 %34
%54 = OpLoad %6 %53
%55 = OpIMul %6 %33 %54
%56 = OpAccessChain %31 %18 %39
%57 = OpLoad %6 %56
%58 = OpIMul %6 %38 %57
%59 = OpIAdd %6 %55 %58
OpStore %52 %59
%61 = OpAccessChain %31 %18 %34
%62 = OpLoad %6 %61
%63 = OpIMul %6 %33 %62
%64 = OpAccessChain %31 %18 %39
%65 = OpLoad %6 %64
%66 = OpIMul %6 %38 %65
%67 = OpIAdd %6 %63 %66
OpStore %60 %67
%80 = OpLoad %6 %32
%83 = OpAccessChain %82 %77 %79 %80
%87 = OpCooperativeMatrixLoadNV %71 %83 %84 %86 )" +
                  loadMemoryAccess + R"( %81
OpStore %73 %87
%90 = OpLoad %71 %73
OpStore %89 %90
%96 = OpLoad %6 %44
%97 = OpAccessChain %82 %95 %79 %96
%98 = OpCooperativeMatrixLoadNV %71 %97 %84 %86 MakePointerVisibleKHR|NonPrivatePointerKHR %81
OpStore %91 %98
%100 = OpLoad %71 %91
OpStore %99 %100
%106 = OpLoad %6 %52
%107 = OpAccessChain %82 %105 %79 %106
%108 = OpCooperativeMatrixLoadNV %71 %107 %84 %86 MakePointerVisibleKHR|NonPrivatePointerKHR %81
OpStore %101 %108
%110 = OpLoad %71 %101
OpStore %109 %110
%114 = OpConvertSToF %68 %113
%115 = OpCompositeConstruct %71 %114
OpStore %111 %115
%116 = OpLoad %71 %111
%121 = OpLoad %6 %60
%122 = OpAccessChain %82 %120 %79 %121
OpCooperativeMatrixStoreNV %122 %116 %84 %86 )" + storeMemoryAccess + R"( %81
OpReturn
OpFunctionEnd
)";

  return s;
}

TEST_F(ValidateMemory, CoopMatLoadStoreSuccess) {
  std::string spirv =
      GenCoopMatLoadStoreShader("MakePointerAvailableKHR|NonPrivatePointerKHR",
                                "MakePointerVisibleKHR|NonPrivatePointerKHR");

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, CoopMatStoreMemoryAccessFail) {
  std::string spirv =
      GenCoopMatLoadStoreShader("MakePointerVisibleKHR|NonPrivatePointerKHR",
                                "MakePointerVisibleKHR|NonPrivatePointerKHR");

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("MakePointerVisibleKHR cannot be used with OpStore"));
}

TEST_F(ValidateMemory, CoopMatLoadMemoryAccessFail) {
  std::string spirv =
      GenCoopMatLoadStoreShader("MakePointerAvailableKHR|NonPrivatePointerKHR",
                                "MakePointerAvailableKHR|NonPrivatePointerKHR");

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("MakePointerAvailableKHR cannot be used with OpLoad"));
}

TEST_F(ValidateMemory, CoopMatInvalidStorageClassFail) {
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
%f16 = OpTypeFloat 16
%u32 = OpTypeInt 32 0

%u32_8 = OpConstant %u32 8
%subgroup = OpConstant %u32 3

%f16mat = OpTypeCooperativeMatrixNV %f16 %subgroup %u32_8 %u32_8

%str = OpTypeStruct %f16mat
%str_ptr = OpTypePointer Workgroup %str
%sh = OpVariable %str_ptr Workgroup

%main = OpFunction %void None %func
%main_entry = OpLabel

OpReturn
OpFunctionEnd)";

  CompileSuccessfully(body.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Cooperative matrix types (or types containing them) can only be "
          "allocated in Function or Private storage classes or as function "
          "parameters"));
}

TEST_F(ValidateMemory, CoopMatMatrixLengthResultTypeBad) {
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
%f16 = OpTypeFloat 16
%u32 = OpTypeInt 32 0
%i32 = OpTypeInt 32 1

%u32_8 = OpConstant %u32 8
%subgroup = OpConstant %u32 3

%f16mat = OpTypeCooperativeMatrixNV %f16 %subgroup %u32_8 %u32_8

%main = OpFunction %void None %func
%main_entry = OpLabel

%1 = OpCooperativeMatrixLengthNV %i32 %f16mat

OpReturn
OpFunctionEnd)";

  CompileSuccessfully(body.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("The Result Type of OpCooperativeMatrixLengthNV <id> "
                "'11[%11]' must be OpTypeInt with width 32 and signedness 0"));
}

TEST_F(ValidateMemory, CoopMatMatrixLengthOperandTypeBad) {
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
%f16 = OpTypeFloat 16
%u32 = OpTypeInt 32 0
%i32 = OpTypeInt 32 1

%u32_8 = OpConstant %u32 8
%subgroup = OpConstant %u32 3

%f16mat = OpTypeCooperativeMatrixNV %f16 %subgroup %u32_8 %u32_8

%main = OpFunction %void None %func
%main_entry = OpLabel

%1 = OpCooperativeMatrixLengthNV %u32 %u32

OpReturn
OpFunctionEnd)";

  CompileSuccessfully(body.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("The type in OpCooperativeMatrixLengthNV <id> '5[%uint]' "
                "must be OpTypeCooperativeMatrixNV"));
}

TEST_F(ValidateMemory, CoopMatMatrixLengthGood) {
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
%f16 = OpTypeFloat 16
%u32 = OpTypeInt 32 0
%i32 = OpTypeInt 32 1

%u32_8 = OpConstant %u32 8
%subgroup = OpConstant %u32 3

%f16mat = OpTypeCooperativeMatrixNV %f16 %subgroup %u32_8 %u32_8

%main = OpFunction %void None %func
%main_entry = OpLabel

%1 = OpCooperativeMatrixLengthNV %u32 %f16mat

OpReturn
OpFunctionEnd)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateMemory, VulkanRTAOutsideOfStructBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%sampler_t = OpTypeSampler
%array_t = OpTypeRuntimeArray %sampler_t
%array_ptr = OpTypePointer UniformConstant %array_t
%2 = OpVariable %array_ptr UniformConstant
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpVariable, <id> '5[%5]', is attempting to create memory for an "
          "illegal type, OpTypeRuntimeArray.\nFor Vulkan OpTypeRuntimeArray "
          "can only appear as the final member of an OpTypeStruct, thus cannot "
          "be instantiated via OpVariable\n  %5 = OpVariable "
          "%_ptr_UniformConstant__runtimearr_2 UniformConstant\n"));
}

TEST_F(ValidateMemory, WebGPURTAOutsideOfStructBad) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%sampler_t = OpTypeSampler
%array_t = OpTypeRuntimeArray %sampler_t
%array_ptr = OpTypePointer UniformConstant %array_t
%2 = OpVariable %array_ptr UniformConstant
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpVariable, <id> '5[%5]', is attempting to create memory for an "
          "illegal type, OpTypeRuntimeArray.\nFor WebGPU OpTypeRuntimeArray "
          "can only appear as the final member of an OpTypeStruct, thus cannot "
          "be instantiated via OpVariable\n  %5 = OpVariable "
          "%_ptr_UniformConstant__runtimearr_2 UniformConstant\n"));
}

TEST_F(ValidateMemory, VulkanRTAOutsideOfStructWithRuntimeDescriptorArrayGood) {
  std::string spirv = R"(
OpCapability Shader
OpCapability RuntimeDescriptorArrayEXT
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%sampler_t = OpTypeSampler
%array_t = OpTypeRuntimeArray %sampler_t
%array_sb_ptr = OpTypePointer StorageBuffer %array_t
%2 = OpVariable %array_sb_ptr StorageBuffer
%array_uc_ptr = OpTypePointer UniformConstant %array_t
%3 = OpVariable %array_uc_ptr UniformConstant
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(
    ValidateMemory,
    VulkanRTAOutsideOfStructWithRuntimeDescriptorArrayAndWrongStorageClassBad) {
  std::string spirv = R"(
OpCapability Shader
OpCapability RuntimeDescriptorArrayEXT
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%uint_t = OpTypeInt 32 0
%array_t = OpTypeRuntimeArray %uint_t
%array_ptr = OpTypePointer Workgroup %array_t
%2 = OpVariable %array_ptr Workgroup
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("For Vulkan with RuntimeDescriptorArrayEXT, a variable "
                "containing OpTypeRuntimeArray must have storage class of "
                "StorageBuffer, Uniform, or UniformConstant.\n  %5 = "
                "OpVariable %_ptr_Workgroup__runtimearr_uint Workgroup\n"));
}

TEST_F(ValidateMemory, VulkanRTAInsideStorageBufferStructGood) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t ArrayStride 4
OpMemberDecorate %struct_t 0 Offset 0
OpDecorate %struct_t Block
%uint_t = OpTypeInt 32 0
%array_t = OpTypeRuntimeArray %uint_t
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer StorageBuffer %struct_t
%2 = OpVariable %struct_ptr StorageBuffer
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, WebGPURTAInsideStorageBufferStructGood) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t ArrayStride 4
OpMemberDecorate %struct_t 0 Offset 0
OpDecorate %struct_t Block
%uint_t = OpTypeInt 32 0
%array_t = OpTypeRuntimeArray %uint_t
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer StorageBuffer %struct_t
%2 = OpVariable %struct_ptr StorageBuffer
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateMemory, VulkanRTAInsideWrongStorageClassStructBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%uint_t = OpTypeInt 32 0
%array_t = OpTypeRuntimeArray %uint_t
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer Workgroup %struct_t
%2 = OpVariable %struct_ptr Workgroup
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "For Vulkan, OpTypeStruct variables containing OpTypeRuntimeArray "
          "must have storage class of StorageBuffer or Uniform.\n  %6 = "
          "OpVariable %_ptr_Workgroup__struct_4 Workgroup\n"));
}

TEST_F(ValidateMemory, WebGPURTAInsideWrongStorageClassStructBad) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%uint_t = OpTypeInt 32 0
%array_t = OpTypeRuntimeArray %uint_t
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer Workgroup %struct_t
%2 = OpVariable %struct_ptr Workgroup
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("For WebGPU, OpTypeStruct variables containing "
                "OpTypeRuntimeArray must have storage class of StorageBuffer\n "
                " %6 = OpVariable %_ptr_Workgroup__struct_4 Workgroup\n"));
}

TEST_F(ValidateMemory, VulkanRTAInsideStorageBufferStructWithoutBlockBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%uint_t = OpTypeInt 32 0
%array_t = OpTypeRuntimeArray %uint_t
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer StorageBuffer %struct_t
%2 = OpVariable %struct_ptr StorageBuffer
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("For Vulkan, an OpTypeStruct variable containing an "
                        "OpTypeRuntimeArray must be decorated with Block if it "
                        "has storage class StorageBuffer.\n  %6 = OpVariable "
                        "%_ptr_StorageBuffer__struct_4 StorageBuffer\n"));
}

TEST_F(ValidateMemory, WebGPURTAInsideStorageBufferStructWithoutBlockBad) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%uint_t = OpTypeInt 32 0
%array_t = OpTypeRuntimeArray %uint_t
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer StorageBuffer %struct_t
%2 = OpVariable %struct_ptr StorageBuffer
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("For WebGPU, an OpTypeStruct variable containing an "
                        "OpTypeRuntimeArray must be decorated with Block if it "
                        "has storage class StorageBuffer.\n  %6 = OpVariable "
                        "%_ptr_StorageBuffer__struct_4 StorageBuffer\n"));
}

TEST_F(ValidateMemory, VulkanRTAInsideUniformStructGood) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t ArrayStride 4
OpMemberDecorate %struct_t 0 Offset 0
OpDecorate %struct_t BufferBlock
%uint_t = OpTypeInt 32 0
%array_t = OpTypeRuntimeArray %uint_t
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer Uniform %struct_t
%2 = OpVariable %struct_ptr Uniform
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, WebGPURTAInsideUniformStructBad) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t ArrayStride 4
OpMemberDecorate %struct_t 0 Offset 0
OpDecorate %struct_t Block
%uint_t = OpTypeInt 32 0
%array_t = OpTypeRuntimeArray %uint_t
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer Uniform %struct_t
%2 = OpVariable %struct_ptr Uniform
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("For WebGPU, OpTypeStruct variables containing "
                "OpTypeRuntimeArray must have storage class of StorageBuffer\n "
                " %6 = OpVariable %_ptr_Uniform__struct_3 Uniform\n"));
}

TEST_F(ValidateMemory, VulkanRTAInsideUniformStructWithoutBufferBlockBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%uint_t = OpTypeInt 32 0
%array_t = OpTypeRuntimeArray %uint_t
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer Uniform %struct_t
%2 = OpVariable %struct_ptr Uniform
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("For Vulkan, an OpTypeStruct variable containing an "
                        "OpTypeRuntimeArray must be decorated with BufferBlock "
                        "if it has storage class Uniform.\n  %6 = OpVariable "
                        "%_ptr_Uniform__struct_4 Uniform\n"));
}

TEST_F(ValidateMemory, VulkanRTAInsideRTABad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%sampler_t = OpTypeSampler
%inner_array_t = OpTypeRuntimeArray %sampler_t
%array_t = OpTypeRuntimeArray %inner_array_t
%array_ptr = OpTypePointer UniformConstant %array_t
%2 = OpVariable %array_ptr UniformConstant
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpTypeRuntimeArray Element Type <id> '3[%_runtimearr_2]' is not "
          "valid in Vulkan environments.\n  %_runtimearr__runtimearr_2 = "
          "OpTypeRuntimeArray %_runtimearr_2\n"));
}

TEST_F(ValidateMemory, WebGPURTAInsideRTABad) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%sampler_t = OpTypeSampler
%inner_array_t = OpTypeRuntimeArray %sampler_t
%array_t = OpTypeRuntimeArray %inner_array_t
%array_ptr = OpTypePointer UniformConstant %array_t
%2 = OpVariable %array_ptr UniformConstant
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpTypeRuntimeArray Element Type <id> '3[%_runtimearr_2]' is not "
          "valid in WebGPU environments.\n  %_runtimearr__runtimearr_2 = "
          "OpTypeRuntimeArray %_runtimearr_2\n"));
}

TEST_F(ValidateMemory, VulkanRTAInsideRTAWithRuntimeDescriptorArrayBad) {
  std::string spirv = R"(
OpCapability RuntimeDescriptorArrayEXT
OpCapability Shader
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t Block
%uint_t = OpTypeInt 32 0
%inner_array_t = OpTypeRuntimeArray %uint_t
%array_t = OpTypeRuntimeArray %inner_array_t
%array_ptr = OpTypePointer StorageBuffer %array_t
%2 = OpVariable %array_ptr StorageBuffer
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpTypeRuntimeArray Element Type <id> '4[%_runtimearr_uint]' is not "
          "valid in Vulkan environments.\n  %_runtimearr__runtimearr_uint = "
          "OpTypeRuntimeArray %_runtimearr_uint\n"));
}

TEST_F(ValidateMemory,
       VulkanUniformStructInsideRTAWithRuntimeDescriptorArrayGood) {
  std::string spirv = R"(
OpCapability RuntimeDescriptorArrayEXT
OpCapability Shader
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t ArrayStride 4
OpMemberDecorate %struct_t 0 Offset 0
OpDecorate %struct_t Block
%uint_t = OpTypeInt 32 0
%struct_t = OpTypeStruct %uint_t
%array_t = OpTypeRuntimeArray %struct_t
%array_ptr = OpTypePointer Uniform %array_t
%2 = OpVariable %array_ptr Uniform
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, VulkanRTAInsideRTAInsideStructBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t ArrayStride 4
OpMemberDecorate %struct_t 0 Offset 0
OpDecorate %struct_t Block
%uint_t = OpTypeInt 32 0
%inner_array_t = OpTypeRuntimeArray %uint_t
%array_t = OpTypeRuntimeArray %inner_array_t
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer StorageBuffer %struct_t
%2 = OpVariable %struct_ptr StorageBuffer
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpTypeRuntimeArray Element Type <id> '5[%_runtimearr_uint]' is not "
          "valid in Vulkan environments.\n  %_runtimearr__runtimearr_uint = "
          "OpTypeRuntimeArray %_runtimearr_uint\n"));
}

TEST_F(ValidateMemory,
       VulkanRTAInsideRTAInsideStructWithRuntimeDescriptorArrayBad) {
  std::string spirv = R"(
OpCapability RuntimeDescriptorArrayEXT
OpCapability Shader
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t ArrayStride 4
OpMemberDecorate %struct_t 0 Offset 0
OpDecorate %struct_t Block
%uint_t = OpTypeInt 32 0
%inner_array_t = OpTypeRuntimeArray %uint_t
%array_t = OpTypeRuntimeArray %inner_array_t
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer StorageBuffer %struct_t
%2 = OpVariable %struct_ptr StorageBuffer
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpTypeRuntimeArray Element Type <id> '5[%_runtimearr_uint]' is not "
          "valid in Vulkan environments.\n  %_runtimearr__runtimearr_uint = "
          "OpTypeRuntimeArray %_runtimearr_uint\n"));
}

TEST_F(ValidateMemory, VulkanRTAInsideArrayBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%uint_t = OpTypeInt 32 0
%dim = OpConstant %uint_t 1
%sampler_t = OpTypeSampler
%inner_array_t = OpTypeRuntimeArray %sampler_t
%array_t = OpTypeArray %inner_array_t %dim
%array_ptr = OpTypePointer UniformConstant %array_t
%2 = OpVariable %array_ptr UniformConstant
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpTypeArray Element Type <id> '5[%_runtimearr_4]' is not "
                "valid in Vulkan environments.\n  %_arr__runtimearr_4_uint_1 = "
                "OpTypeArray %_runtimearr_4 %uint_1\n"));
}

TEST_F(ValidateMemory, WebGPURTAInsideArrayBad) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%uint_t = OpTypeInt 32 0
%dim = OpConstant %uint_t 1
%sampler_t = OpTypeSampler
%inner_array_t = OpTypeRuntimeArray %sampler_t
%array_t = OpTypeArray %inner_array_t %dim
%array_ptr = OpTypePointer UniformConstant %array_t
%2 = OpVariable %array_ptr UniformConstant
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpTypeArray Element Type <id> '5[%_runtimearr_4]' is not "
                "valid in WebGPU environments.\n  %_arr__runtimearr_4_uint_1 = "
                "OpTypeArray %_runtimearr_4 %uint_1\n"));
}

TEST_F(ValidateMemory, VulkanRTAInsideArrayWithRuntimeDescriptorArrayBad) {
  std::string spirv = R"(
OpCapability RuntimeDescriptorArrayEXT
OpCapability Shader
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t Block
%uint_t = OpTypeInt 32 0
%dim = OpConstant %uint_t 1
%sampler_t = OpTypeSampler
%inner_array_t = OpTypeRuntimeArray %uint_t
%array_t = OpTypeRuntimeArray %inner_array_t
%array_ptr = OpTypePointer StorageBuffer %array_t
%2 = OpVariable %array_ptr StorageBuffer
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpTypeRuntimeArray Element Type <id> '6[%_runtimearr_uint]' is not "
          "valid in Vulkan environments.\n  %_runtimearr__runtimearr_uint = "
          "OpTypeRuntimeArray %_runtimearr_uint\n"));
}

TEST_F(ValidateMemory, VulkanRTAInsideArrayInsideStructBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t ArrayStride 4
OpMemberDecorate %struct_t 0 Offset 0
OpDecorate %struct_t Block
%uint_t = OpTypeInt 32 0
%dim = OpConstant %uint_t 1
%inner_array_t = OpTypeRuntimeArray %uint_t
%array_t = OpTypeArray %inner_array_t %dim
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer StorageBuffer %struct_t
%2 = OpVariable %struct_ptr StorageBuffer
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpTypeArray Element Type <id> '6[%_runtimearr_uint]' is not "
          "valid in Vulkan environments.\n  %_arr__runtimearr_uint_uint_1 "
          "= OpTypeArray %_runtimearr_uint %uint_1\n"));
}

TEST_F(ValidateMemory,
       VulkanRTAInsideArrayInsideStructWithRuntimeDescriptorArrayBad) {
  std::string spirv = R"(
OpCapability RuntimeDescriptorArrayEXT
OpCapability Shader
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t ArrayStride 4
OpMemberDecorate %struct_t 0 Offset 0
OpDecorate %struct_t Block
%uint_t = OpTypeInt 32 0
%dim = OpConstant %uint_t 1
%inner_array_t = OpTypeRuntimeArray %uint_t
%array_t = OpTypeArray %inner_array_t %dim
%struct_t = OpTypeStruct %array_t
%struct_ptr = OpTypePointer StorageBuffer %struct_t
%2 = OpVariable %struct_ptr StorageBuffer
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpTypeArray Element Type <id> '6[%_runtimearr_uint]' is not "
          "valid in Vulkan environments.\n  %_arr__runtimearr_uint_uint_1 "
          "= OpTypeArray %_runtimearr_uint %uint_1\n"));
}

TEST_F(ValidateMemory, VulkanRTAStructInsideRTAWithRuntimeDescriptorArrayGood) {
  std::string spirv = R"(
OpCapability RuntimeDescriptorArrayEXT
OpCapability Shader
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t ArrayStride 4
OpMemberDecorate %struct_t 0 Offset 0
OpDecorate %struct_t Block
%uint_t = OpTypeInt 32 0
%inner_array_t = OpTypeRuntimeArray %uint_t
%struct_t = OpTypeStruct %inner_array_t
%array_t = OpTypeRuntimeArray %struct_t
%array_ptr = OpTypePointer StorageBuffer %array_t
%2 = OpVariable %array_ptr StorageBuffer
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, VulkanRTAStructInsideArrayGood) {
  std::string spirv = R"(
OpCapability RuntimeDescriptorArrayEXT
OpCapability Shader
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
OpDecorate %array_t ArrayStride 4
OpMemberDecorate %struct_t 0 Offset 0
OpDecorate %struct_t Block
%uint_t = OpTypeInt 32 0
%inner_array_t = OpTypeRuntimeArray %uint_t
%struct_t = OpTypeStruct %inner_array_t
%array_size = OpConstant %uint_t 5
%array_t = OpTypeArray %struct_t %array_size
%array_ptr = OpTypePointer StorageBuffer %array_t
%2 = OpVariable %array_ptr StorageBuffer
%void = OpTypeVoid
%func_t = OpTypeFunction %void
%func = OpFunction %void None %func_t
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
