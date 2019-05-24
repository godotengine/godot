// Copyright (c) 2019 Google LLC.
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

#include "test/val/val_code_generator.h"

#include <sstream>

namespace spvtools {
namespace val {
namespace {

std::string GetDefaultShaderCapabilities() {
  return R"(
OpCapability Shader
OpCapability Geometry
OpCapability Tessellation
OpCapability Float64
OpCapability Int64
OpCapability MultiViewport
OpCapability SampleRateShading
)";
}

std::string GetWebGPUShaderCapabilities() {
  return R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
)";
}

std::string GetDefaultShaderTypes() {
  return R"(
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%f64 = OpTypeFloat 64
%u32 = OpTypeInt 32 0
%u64 = OpTypeInt 64 0
%f32vec2 = OpTypeVector %f32 2
%f32vec3 = OpTypeVector %f32 3
%f32vec4 = OpTypeVector %f32 4
%f64vec2 = OpTypeVector %f64 2
%f64vec3 = OpTypeVector %f64 3
%f64vec4 = OpTypeVector %f64 4
%u32vec2 = OpTypeVector %u32 2
%u32vec3 = OpTypeVector %u32 3
%u64vec3 = OpTypeVector %u64 3
%u32vec4 = OpTypeVector %u32 4
%u64vec2 = OpTypeVector %u64 2

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%f32_2 = OpConstant %f32 2
%f32_3 = OpConstant %f32 3
%f32_4 = OpConstant %f32 4
%f32_h = OpConstant %f32 0.5
%f32vec2_01 = OpConstantComposite %f32vec2 %f32_0 %f32_1
%f32vec2_12 = OpConstantComposite %f32vec2 %f32_1 %f32_2
%f32vec3_012 = OpConstantComposite %f32vec3 %f32_0 %f32_1 %f32_2
%f32vec3_123 = OpConstantComposite %f32vec3 %f32_1 %f32_2 %f32_3
%f32vec4_0123 = OpConstantComposite %f32vec4 %f32_0 %f32_1 %f32_2 %f32_3
%f32vec4_1234 = OpConstantComposite %f32vec4 %f32_1 %f32_2 %f32_3 %f32_4

%f64_0 = OpConstant %f64 0
%f64_1 = OpConstant %f64 1
%f64_2 = OpConstant %f64 2
%f64_3 = OpConstant %f64 3
%f64vec2_01 = OpConstantComposite %f64vec2 %f64_0 %f64_1
%f64vec3_012 = OpConstantComposite %f64vec3 %f64_0 %f64_1 %f64_2
%f64vec4_0123 = OpConstantComposite %f64vec4 %f64_0 %f64_1 %f64_2 %f64_3

%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u32_2 = OpConstant %u32 2
%u32_3 = OpConstant %u32 3
%u32_4 = OpConstant %u32 4

%u64_0 = OpConstant %u64 0
%u64_1 = OpConstant %u64 1
%u64_2 = OpConstant %u64 2
%u64_3 = OpConstant %u64 3

%u32vec2_01 = OpConstantComposite %u32vec2 %u32_0 %u32_1
%u32vec2_12 = OpConstantComposite %u32vec2 %u32_1 %u32_2
%u32vec4_0123 = OpConstantComposite %u32vec4 %u32_0 %u32_1 %u32_2 %u32_3
%u64vec2_01 = OpConstantComposite %u64vec2 %u64_0 %u64_1

%u32arr2 = OpTypeArray %u32 %u32_2
%u32arr3 = OpTypeArray %u32 %u32_3
%u32arr4 = OpTypeArray %u32 %u32_4
%u64arr2 = OpTypeArray %u64 %u32_2
%u64arr3 = OpTypeArray %u64 %u32_3
%u64arr4 = OpTypeArray %u64 %u32_4
%f32arr2 = OpTypeArray %f32 %u32_2
%f32arr3 = OpTypeArray %f32 %u32_3
%f32arr4 = OpTypeArray %f32 %u32_4
%f64arr2 = OpTypeArray %f64 %u32_2
%f64arr3 = OpTypeArray %f64 %u32_3
%f64arr4 = OpTypeArray %f64 %u32_4

%f32vec3arr3 = OpTypeArray %f32vec3 %u32_3
%f32vec4arr3 = OpTypeArray %f32vec4 %u32_3
%f64vec4arr3 = OpTypeArray %f64vec4 %u32_3
)";
}

std::string GetWebGPUShaderTypes() {
  return R"(
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%f32vec2 = OpTypeVector %f32 2
%f32vec3 = OpTypeVector %f32 3
%f32vec4 = OpTypeVector %f32 4
%u32vec2 = OpTypeVector %u32 2
%u32vec3 = OpTypeVector %u32 3
%u32vec4 = OpTypeVector %u32 4

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%f32_2 = OpConstant %f32 2
%f32_3 = OpConstant %f32 3
%f32_4 = OpConstant %f32 4
%f32_h = OpConstant %f32 0.5
%f32vec2_01 = OpConstantComposite %f32vec2 %f32_0 %f32_1
%f32vec2_12 = OpConstantComposite %f32vec2 %f32_1 %f32_2
%f32vec3_012 = OpConstantComposite %f32vec3 %f32_0 %f32_1 %f32_2
%f32vec3_123 = OpConstantComposite %f32vec3 %f32_1 %f32_2 %f32_3
%f32vec4_0123 = OpConstantComposite %f32vec4 %f32_0 %f32_1 %f32_2 %f32_3
%f32vec4_1234 = OpConstantComposite %f32vec4 %f32_1 %f32_2 %f32_3 %f32_4

%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u32_2 = OpConstant %u32 2
%u32_3 = OpConstant %u32 3
%u32_4 = OpConstant %u32 4

%u32vec2_01 = OpConstantComposite %u32vec2 %u32_0 %u32_1
%u32vec2_12 = OpConstantComposite %u32vec2 %u32_1 %u32_2
%u32vec4_0123 = OpConstantComposite %u32vec4 %u32_0 %u32_1 %u32_2 %u32_3

%u32arr2 = OpTypeArray %u32 %u32_2
%u32arr3 = OpTypeArray %u32 %u32_3
%u32arr4 = OpTypeArray %u32 %u32_4
%f32arr2 = OpTypeArray %f32 %u32_2
%f32arr3 = OpTypeArray %f32 %u32_3
%f32arr4 = OpTypeArray %f32 %u32_4

%f32vec3arr3 = OpTypeArray %f32vec3 %u32_3
%f32vec4arr3 = OpTypeArray %f32vec4 %u32_3
)";
}

}  // namespace

CodeGenerator CodeGenerator::GetDefaultShaderCodeGenerator() {
  CodeGenerator generator;
  generator.capabilities_ = GetDefaultShaderCapabilities();
  generator.memory_model_ = "OpMemoryModel Logical GLSL450\n";
  generator.types_ = GetDefaultShaderTypes();
  return generator;
}

CodeGenerator CodeGenerator::GetWebGPUShaderCodeGenerator() {
  CodeGenerator generator;
  generator.capabilities_ = GetWebGPUShaderCapabilities();
  generator.memory_model_ = "OpMemoryModel Logical VulkanKHR\n";
  generator.extensions_ = "OpExtension \"SPV_KHR_vulkan_memory_model\"\n";
  generator.types_ = GetWebGPUShaderTypes();
  return generator;
}

std::string CodeGenerator::Build() const {
  std::ostringstream ss;

  ss << capabilities_;
  ss << extensions_;
  ss << memory_model_;

  for (const EntryPoint& entry_point : entry_points_) {
    ss << "OpEntryPoint " << entry_point.execution_model << " %"
       << entry_point.name << " \"" << entry_point.name << "\" "
       << entry_point.interfaces << "\n";
  }

  for (const EntryPoint& entry_point : entry_points_) {
    ss << entry_point.execution_modes << "\n";
  }

  ss << before_types_;
  ss << types_;
  ss << after_types_;

  for (const EntryPoint& entry_point : entry_points_) {
    ss << "\n";
    ss << "%" << entry_point.name << " = OpFunction %void None %func\n";
    ss << "%" << entry_point.name << "_entry = OpLabel\n";
    ss << entry_point.body;
    ss << "\nOpReturn\nOpFunctionEnd\n";
  }

  ss << add_at_the_end_;

  return ss.str();
}

}  // namespace val
}  // namespace spvtools
