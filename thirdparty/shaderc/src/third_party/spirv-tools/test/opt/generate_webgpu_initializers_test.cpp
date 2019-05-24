// Copyright (c) 2019 Google Inc.
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

#include <vector>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

typedef std::tuple<std::string, bool> GenerateWebGPUInitializersParam;

using GlobalVariableTest =
    PassTest<::testing::TestWithParam<GenerateWebGPUInitializersParam>>;
using LocalVariableTest =
    PassTest<::testing::TestWithParam<GenerateWebGPUInitializersParam>>;

using GenerateWebGPUInitializersTest = PassTest<::testing::Test>;

void operator+=(std::vector<const char*>& lhs, const char* rhs) {
  lhs.push_back(rhs);
}

void operator+=(std::vector<const char*>& lhs,
                const std::vector<const char*>& rhs) {
  lhs.reserve(lhs.size() + rhs.size());
  for (auto* c : rhs) lhs.push_back(c);
}

std::string GetGlobalVariableTestString(std::string ptr_str,
                                        std::string var_str,
                                        std::string const_str = "") {
  std::vector<const char*> result = {
      // clang-format off
               "OpCapability Shader",
               "OpCapability VulkanMemoryModelKHR",
               "OpExtension \"SPV_KHR_vulkan_memory_model\"",
               "OpMemoryModel Logical VulkanKHR",
               "OpEntryPoint Vertex %1 \"shader\"",
       "%uint = OpTypeInt 32 0",
                ptr_str.c_str()};
  // clang-format on

  if (!const_str.empty()) result += const_str.c_str();

  result += {
      // clang-format off
                var_str.c_str(),
     "%uint_0 = OpConstant %uint 0",
       "%void = OpTypeVoid",
          "%7 = OpTypeFunction %void",
          "%1 = OpFunction %void None %7",
          "%8 = OpLabel",
               "OpStore %4 %uint_0",
               "OpReturn",
               "OpFunctionEnd"
      // clang-format on
  };
  return JoinAllInsts(result);
}

std::string GetPointerString(std::string storage_type) {
  std::string result = "%_ptr_";
  result += storage_type + "_uint = OpTypePointer ";
  result += storage_type + " %uint";
  return result;
}

std::string GetGlobalVariableString(std::string storage_type,
                                    bool initialized) {
  std::string result = "%4 = OpVariable %_ptr_";
  result += storage_type + "_uint ";
  result += storage_type;
  if (initialized) result += " %9";
  return result;
}

std::string GetUninitializedGlobalVariableTestString(std::string storage_type) {
  return GetGlobalVariableTestString(
      GetPointerString(storage_type),
      GetGlobalVariableString(storage_type, false));
}

std::string GetNullConstantString() { return "%9 = OpConstantNull %uint"; }

std::string GetInitializedGlobalVariableTestString(std::string storage_type) {
  return GetGlobalVariableTestString(
      GetPointerString(storage_type),
      GetGlobalVariableString(storage_type, true), GetNullConstantString());
}

TEST_P(GlobalVariableTest, Check) {
  std::string storage_class = std::get<0>(GetParam());
  bool changed = std::get<1>(GetParam());
  std::string input = GetUninitializedGlobalVariableTestString(storage_class);
  std::string expected =
      changed ? GetInitializedGlobalVariableTestString(storage_class) : input;

  SinglePassRunAndCheck<GenerateWebGPUInitializersPass>(input, expected,
                                                        /* skip_nop = */ false);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    GenerateWebGPUInitializers, GlobalVariableTest,
    ::testing::ValuesIn(std::vector<GenerateWebGPUInitializersParam>({
       std::make_tuple("Private", true),
       std::make_tuple("Output", true),
       std::make_tuple("Function", true),
       std::make_tuple("UniformConstant", false),
       std::make_tuple("Input", false),
       std::make_tuple("Uniform", false),
       std::make_tuple("Workgroup", false)
    })));
// clang-format on

std::string GetLocalVariableTestString(std::string ptr_str, std::string var_str,
                                       std::string const_str = "") {
  std::vector<const char*> result = {
      // clang-format off
               "OpCapability Shader",
               "OpCapability VulkanMemoryModelKHR",
               "OpExtension \"SPV_KHR_vulkan_memory_model\"",
               "OpMemoryModel Logical VulkanKHR",
               "OpEntryPoint Vertex %1 \"shader\"",
       "%uint = OpTypeInt 32 0",
                ptr_str.c_str(),
     "%uint_0 = OpConstant %uint 0",
       "%void = OpTypeVoid",
          "%6 = OpTypeFunction %void"};
  // clang-format on

  if (!const_str.empty()) result += const_str.c_str();

  result += {
      // clang-format off
          "%1 = OpFunction %void None %6",
          "%7 = OpLabel",
                var_str.c_str(),
               "OpStore %8 %uint_0"
      // clang-format on
  };
  return JoinAllInsts(result);
}

std::string GetLocalVariableString(std::string storage_type, bool initialized) {
  std::string result = "%8 = OpVariable %_ptr_";
  result += storage_type + "_uint ";
  result += storage_type;
  if (initialized) result += " %9";
  return result;
}

std::string GetUninitializedLocalVariableTestString(std::string storage_type) {
  return GetLocalVariableTestString(
      GetPointerString(storage_type),
      GetLocalVariableString(storage_type, false));
}

std::string GetInitializedLocalVariableTestString(std::string storage_type) {
  return GetLocalVariableTestString(GetPointerString(storage_type),
                                    GetLocalVariableString(storage_type, true),
                                    GetNullConstantString());
}

TEST_P(LocalVariableTest, Check) {
  std::string storage_class = std::get<0>(GetParam());
  bool changed = std::get<1>(GetParam());

  std::string input = GetUninitializedLocalVariableTestString(storage_class);
  std::string expected =
      changed ? GetInitializedLocalVariableTestString(storage_class) : input;

  SinglePassRunAndCheck<GenerateWebGPUInitializersPass>(input, expected,
                                                        /* skip_nop = */ false);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    GenerateWebGPUInitializers, LocalVariableTest,
    ::testing::ValuesIn(std::vector<GenerateWebGPUInitializersParam>({
       std::make_tuple("Private", true),
       std::make_tuple("Output", true),
       std::make_tuple("Function", true),
       std::make_tuple("UniformConstant", false),
       std::make_tuple("Input", false),
       std::make_tuple("Uniform", false),
       std::make_tuple("Workgroup", false)
    })));
// clang-format on

TEST_F(GenerateWebGPUInitializersTest, AlreadyInitializedUnchanged) {
  std::vector<const char*> spirv = {
      // clang-format off
                       "OpCapability Shader",
                       "OpCapability VulkanMemoryModelKHR",
                       "OpExtension \"SPV_KHR_vulkan_memory_model\"",
                       "OpMemoryModel Logical VulkanKHR",
                       "OpEntryPoint Vertex %1 \"shader\"",
               "%uint = OpTypeInt 32 0",
  "%_ptr_Private_uint = OpTypePointer Private %uint",
             "%uint_0 = OpConstant %uint 0",
                  "%5 = OpVariable %_ptr_Private_uint Private %uint_0",
               "%void = OpTypeVoid",
                  "%7 = OpTypeFunction %void",
                  "%1 = OpFunction %void None %7",
                  "%8 = OpLabel",
                       "OpReturn",
                       "OpFunctionEnd"
      // clang-format on
  };
  std::string str = JoinAllInsts(spirv);

  SinglePassRunAndCheck<GenerateWebGPUInitializersPass>(str, str,
                                                        /* skip_nop = */ false);
}

TEST_F(GenerateWebGPUInitializersTest, AmbigiousArrays) {
  std::vector<const char*> input_spirv = {
      // clang-format off
                                   "OpCapability Shader",
                                   "OpCapability VulkanMemoryModelKHR",
                                   "OpExtension \"SPV_KHR_vulkan_memory_model\"",
                                   "OpMemoryModel Logical VulkanKHR",
                                   "OpEntryPoint Vertex %1 \"shader\"",
                           "%uint = OpTypeInt 32 0",
                         "%uint_2 = OpConstant %uint 2",
               "%_arr_uint_uint_2 = OpTypeArray %uint %uint_2",
             "%_arr_uint_uint_2_0 = OpTypeArray %uint %uint_2",
  "%_ptr_Private__arr_uint_uint_2 = OpTypePointer Private %_arr_uint_uint_2",
"%_ptr_Private__arr_uint_uint_2_0 = OpTypePointer Private %_arr_uint_uint_2_0",
                              "%8 = OpConstantNull %_arr_uint_uint_2_0",
                              "%9 = OpVariable %_ptr_Private__arr_uint_uint_2 Private",
                             "%10 = OpVariable %_ptr_Private__arr_uint_uint_2_0 Private %8",
                           "%void = OpTypeVoid",
                             "%12 = OpTypeFunction %void",
                              "%1 = OpFunction %void None %12",
                             "%13 = OpLabel",
                                   "OpReturn",
                                   "OpFunctionEnd"
      // clang-format on
  };
  std::string input_str = JoinAllInsts(input_spirv);

  std::vector<const char*> expected_spirv = {
      // clang-format off
                                   "OpCapability Shader",
                                   "OpCapability VulkanMemoryModelKHR",
                                   "OpExtension \"SPV_KHR_vulkan_memory_model\"",
                                   "OpMemoryModel Logical VulkanKHR",
                                   "OpEntryPoint Vertex %1 \"shader\"",
                           "%uint = OpTypeInt 32 0",
                         "%uint_2 = OpConstant %uint 2",
               "%_arr_uint_uint_2 = OpTypeArray %uint %uint_2",
             "%_arr_uint_uint_2_0 = OpTypeArray %uint %uint_2",
  "%_ptr_Private__arr_uint_uint_2 = OpTypePointer Private %_arr_uint_uint_2",
"%_ptr_Private__arr_uint_uint_2_0 = OpTypePointer Private %_arr_uint_uint_2_0",
                              "%8 = OpConstantNull %_arr_uint_uint_2_0",
                             "%14 = OpConstantNull %_arr_uint_uint_2",
                              "%9 = OpVariable %_ptr_Private__arr_uint_uint_2 Private %14",
                             "%10 = OpVariable %_ptr_Private__arr_uint_uint_2_0 Private %8",
                           "%void = OpTypeVoid",
                             "%12 = OpTypeFunction %void",
                              "%1 = OpFunction %void None %12",
                             "%13 = OpLabel",
                                   "OpReturn",
                                   "OpFunctionEnd"
      // clang-format on
  };
  std::string expected_str = JoinAllInsts(expected_spirv);

  SinglePassRunAndCheck<GenerateWebGPUInitializersPass>(input_str, expected_str,
                                                        /* skip_nop = */ false);
}

TEST_F(GenerateWebGPUInitializersTest, AmbigiousStructs) {
  std::vector<const char*> input_spirv = {
      // clang-format off
                          "OpCapability Shader",
                          "OpCapability VulkanMemoryModelKHR",
                          "OpExtension \"SPV_KHR_vulkan_memory_model\"",
                          "OpMemoryModel Logical VulkanKHR",
                          "OpEntryPoint Vertex %1 \"shader\"",
                  "%uint = OpTypeInt 32 0",
             "%_struct_3 = OpTypeStruct %uint",
             "%_struct_4 = OpTypeStruct %uint",
"%_ptr_Private__struct_3 = OpTypePointer Private %_struct_3",
"%_ptr_Private__struct_4 = OpTypePointer Private %_struct_4",
                     "%7 = OpConstantNull %_struct_3",
                     "%8 = OpVariable %_ptr_Private__struct_3 Private %7",
                     "%9 = OpVariable %_ptr_Private__struct_4 Private",
                  "%void = OpTypeVoid",
                    "%11 = OpTypeFunction %void",
                     "%1 = OpFunction %void None %11",
                    "%12 = OpLabel",
                          "OpReturn",
                          "OpFunctionEnd"
      // clang-format on
  };
  std::string input_str = JoinAllInsts(input_spirv);

  std::vector<const char*> expected_spirv = {
      // clang-format off
                          "OpCapability Shader",
                          "OpCapability VulkanMemoryModelKHR",
                          "OpExtension \"SPV_KHR_vulkan_memory_model\"",
                          "OpMemoryModel Logical VulkanKHR",
                          "OpEntryPoint Vertex %1 \"shader\"",
                  "%uint = OpTypeInt 32 0",
             "%_struct_3 = OpTypeStruct %uint",
             "%_struct_4 = OpTypeStruct %uint",
"%_ptr_Private__struct_3 = OpTypePointer Private %_struct_3",
"%_ptr_Private__struct_4 = OpTypePointer Private %_struct_4",
                     "%7 = OpConstantNull %_struct_3",
                     "%8 = OpVariable %_ptr_Private__struct_3 Private %7",
                    "%13 = OpConstantNull %_struct_4",
                     "%9 = OpVariable %_ptr_Private__struct_4 Private %13",
                  "%void = OpTypeVoid",
                    "%11 = OpTypeFunction %void",
                     "%1 = OpFunction %void None %11",
                    "%12 = OpLabel",
                          "OpReturn",
                          "OpFunctionEnd"
      // clang-format on
  };
  std::string expected_str = JoinAllInsts(expected_spirv);

  SinglePassRunAndCheck<GenerateWebGPUInitializersPass>(input_str, expected_str,
                                                        /* skip_nop = */ false);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
