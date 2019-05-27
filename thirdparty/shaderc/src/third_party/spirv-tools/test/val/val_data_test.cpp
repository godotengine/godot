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

// Validation tests for Data Rules.

#include <sstream>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;
using ::testing::MatchesRegex;

using ValidateData = spvtest::ValidateBase<std::pair<std::string, bool>>;

std::string HeaderWith(std::string cap) {
  return std::string("OpCapability Shader OpCapability Linkage OpCapability ") +
         cap + " OpMemoryModel Logical GLSL450 ";
}

std::string WebGPUHeaderWith(std::string cap) {
  return R"(
OpCapability Shader
OpCapability )" +
         cap + R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
)";
}

std::string webgpu_header = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
)";

std::string header = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
)";
std::string header_with_addresses = R"(
     OpCapability Addresses
     OpCapability Kernel
     OpCapability GenericPointer
     OpCapability Linkage
     OpMemoryModel Physical32 OpenCL
)";
std::string header_with_vec16_cap = R"(
     OpCapability Shader
     OpCapability Vector16
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
)";
std::string header_with_int8 = R"(
     OpCapability Shader
     OpCapability Linkage
     OpCapability Int8
     OpMemoryModel Logical GLSL450
)";
std::string header_with_int16 = R"(
     OpCapability Shader
     OpCapability Linkage
     OpCapability Int16
     OpMemoryModel Logical GLSL450
)";
std::string header_with_int64 = R"(
     OpCapability Shader
     OpCapability Linkage
     OpCapability Int64
     OpMemoryModel Logical GLSL450
)";
std::string header_with_float16 = R"(
     OpCapability Shader
     OpCapability Linkage
     OpCapability Float16
     OpMemoryModel Logical GLSL450
)";
std::string header_with_float16_buffer = R"(
     OpCapability Shader
     OpCapability Linkage
     OpCapability Float16Buffer
     OpMemoryModel Logical GLSL450
)";
std::string header_with_float64 = R"(
     OpCapability Shader
     OpCapability Linkage
     OpCapability Float64
     OpMemoryModel Logical GLSL450
)";

std::string invalid_comp_error = "Illegal number of components";
std::string missing_cap_error = "requires the Vector16 capability";
std::string missing_int8_cap_error = "requires the Int8 capability";
std::string missing_int16_cap_error =
    "requires the Int16 capability,"
    " or an extension that explicitly enables 16-bit integers.";
std::string missing_int64_cap_error = "requires the Int64 capability";
std::string missing_float16_cap_error =
    "requires the Float16 or Float16Buffer capability,"
    " or an extension that explicitly enables 16-bit floating point.";
std::string missing_float64_cap_error = "requires the Float64 capability";
std::string invalid_num_bits_error = "Invalid number of bits";

TEST_F(ValidateData, vec0) {
  std::string str = header + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 0
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(invalid_comp_error));
}

TEST_F(ValidateData, vec1) {
  std::string str = header + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 1
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(invalid_comp_error));
}

TEST_F(ValidateData, vec2) {
  std::string str = header + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 2
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, vec3) {
  std::string str = header + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 3
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, vec4) {
  std::string str = header + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, vec5) {
  std::string str = header + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 5
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(invalid_comp_error));
}

TEST_F(ValidateData, vec8) {
  std::string str = header + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 8
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(missing_cap_error));
}

TEST_F(ValidateData, vec8_with_capability) {
  std::string str = header_with_vec16_cap + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 8
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, vec16) {
  std::string str = header + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 8
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(missing_cap_error));
}

TEST_F(ValidateData, vec16_with_capability) {
  std::string str = header_with_vec16_cap + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 16
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, vec15) {
  std::string str = header + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 15
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(invalid_comp_error));
}

TEST_F(ValidateData, int8_good) {
  std::string str = header_with_int8 + "%2 = OpTypeInt 8 0";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, int8_bad) {
  std::string str = header + "%2 = OpTypeInt 8 1";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(missing_int8_cap_error));
}

TEST_F(ValidateData, int8_with_storage_buffer_8bit_access_good) {
  std::string str = HeaderWith(
                        "StorageBuffer8BitAccess "
                        "OpExtension \"SPV_KHR_8bit_storage\"") +
                    " %2 = OpTypeInt 8 0";
  CompileSuccessfully(str.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions()) << getDiagnosticString();
}

TEST_F(ValidateData, int8_with_uniform_and_storage_buffer_8bit_access_good) {
  std::string str = HeaderWith(
                        "UniformAndStorageBuffer8BitAccess "
                        "OpExtension \"SPV_KHR_8bit_storage\"") +
                    " %2 = OpTypeInt 8 0";
  CompileSuccessfully(str.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions()) << getDiagnosticString();
}

TEST_F(ValidateData, int8_with_storage_push_constant_8_good) {
  std::string str = HeaderWith(
                        "StoragePushConstant8 "
                        "OpExtension \"SPV_KHR_8bit_storage\"") +
                    " %2 = OpTypeInt 8 0";
  CompileSuccessfully(str.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions()) << getDiagnosticString();
}

TEST_F(ValidateData, webgpu_int8_bad) {
  std::string str = WebGPUHeaderWith("Int8") + "%2 = OpTypeInt 8 0";
  CompileSuccessfully(str.c_str(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_CAPABILITY,
            ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Capability Int8 is not allowed by WebGPU specification (or "
                "requires extension)\n"
                "  OpCapability Int8\n"));
}

TEST_F(ValidateData, int16_good) {
  std::string str = header_with_int16 + "%2 = OpTypeInt 16 1";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, storage_uniform_buffer_block_16_good) {
  std::string str = HeaderWith(
                        "StorageUniformBufferBlock16 "
                        "OpExtension \"SPV_KHR_16bit_storage\"") +
                    "%2 = OpTypeInt 16 1 %3 = OpTypeFloat 16";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, storage_uniform_16_good) {
  std::string str =
      HeaderWith("StorageUniform16 OpExtension \"SPV_KHR_16bit_storage\"") +
      "%2 = OpTypeInt 16 1 %3 = OpTypeFloat 16";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, storage_push_constant_16_good) {
  std::string str = HeaderWith(
                        "StoragePushConstant16 "
                        "OpExtension \"SPV_KHR_16bit_storage\"") +
                    "%2 = OpTypeInt 16 1 %3 = OpTypeFloat 16";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, storage_input_output_16_good) {
  std::string str = HeaderWith(
                        "StorageInputOutput16 "
                        "OpExtension \"SPV_KHR_16bit_storage\"") +
                    "%2 = OpTypeInt 16 1 %3 = OpTypeFloat 16";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, amd_gpu_shader_half_float_fetch_16_good) {
  std::string str = R"(
     OpCapability Shader
     OpCapability Linkage
     OpExtension "SPV_AMD_gpu_shader_half_float_fetch"
     OpMemoryModel Logical GLSL450
     %2 = OpTypeFloat 16)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, int16_bad) {
  std::string str = header + "%2 = OpTypeInt 16 1";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(missing_int16_cap_error));
}

TEST_F(ValidateData, webgpu_int16_bad) {
  std::string str = WebGPUHeaderWith("Int16") + "%2 = OpTypeInt 16 1";
  CompileSuccessfully(str.c_str(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_CAPABILITY,
            ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Capability Int16 is not allowed by WebGPU specification (or "
                "requires extension)\n"
                "  OpCapability Int16\n"));
}

TEST_F(ValidateData, webgpu_int32_good) {
  std::string str = webgpu_header + R"(
          OpEntryPoint Fragment %func "func"
          OpExecutionMode %func OriginUpperLeft
%uint_t = OpTypeInt 32 0
  %void = OpTypeVoid
%func_t = OpTypeFunction %void
  %func = OpFunction %void None %func_t
     %1 = OpLabel
          OpReturn
          OpFunctionEnd
)";
  CompileSuccessfully(str.c_str(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateData, int64_good) {
  std::string str = header_with_int64 + "%2 = OpTypeInt 64 1";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, int64_bad) {
  std::string str = header + "%2 = OpTypeInt 64 1";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(missing_int64_cap_error));
}

TEST_F(ValidateData, webgpu_int64_bad) {
  std::string str = WebGPUHeaderWith("Int64") + "%2 = OpTypeInt 64 1";
  CompileSuccessfully(str.c_str(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_CAPABILITY,
            ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Capability Int64 is not allowed by WebGPU specification (or "
                "requires extension)\n"
                "  OpCapability Int64\n"));
}

// Number of bits in an integer may be only one of: {8,16,32,64}
TEST_F(ValidateData, int_invalid_num_bits) {
  std::string str = header + "%2 = OpTypeInt 48 1";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(invalid_num_bits_error));
}

TEST_F(ValidateData, float16_good) {
  std::string str = header_with_float16 + "%2 = OpTypeFloat 16";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, float16_buffer_good) {
  std::string str = header_with_float16_buffer + "%2 = OpTypeFloat 16";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, float16_bad) {
  std::string str = header + "%2 = OpTypeFloat 16";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(missing_float16_cap_error));
}

TEST_F(ValidateData, webgpu_float16_bad) {
  std::string str = WebGPUHeaderWith("Float16") + "%2 = OpTypeFloat 16";
  CompileSuccessfully(str.c_str(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_CAPABILITY,
            ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Capability Float16 is not allowed by WebGPU specification (or "
                "requires extension)\n"
                "  OpCapability Float16\n"));
}

TEST_F(ValidateData, webgpu_float32_good) {
  std::string str = webgpu_header + R"(
           OpEntryPoint Fragment %func "func"
           OpExecutionMode %func OriginUpperLeft
%float_t = OpTypeFloat 32
   %void = OpTypeVoid
 %func_t = OpTypeFunction %void
   %func = OpFunction %void None %func_t
      %1 = OpLabel
           OpReturn
           OpFunctionEnd
)";
  CompileSuccessfully(str.c_str(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateData, float64_good) {
  std::string str = header_with_float64 + "%2 = OpTypeFloat 64";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, float64_bad) {
  std::string str = header + "%2 = OpTypeFloat 64";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(missing_float64_cap_error));
}

TEST_F(ValidateData, webgpu_float64_bad) {
  std::string str = WebGPUHeaderWith("Float64") + "%2 = OpTypeFloat 64";
  CompileSuccessfully(str.c_str(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_CAPABILITY,
            ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Capability Float64 is not allowed by WebGPU specification (or "
                "requires extension)\n"
                "  OpCapability Float64\n"));
}

// Number of bits in a float may be only one of: {16,32,64}
TEST_F(ValidateData, float_invalid_num_bits) {
  std::string str = header + "%2 = OpTypeFloat 48";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(invalid_num_bits_error));
}

TEST_F(ValidateData, matrix_data_type_float) {
  std::string str = header + R"(
%f32    =  OpTypeFloat 32
%vec3   =  OpTypeVector %f32 3
%mat33  =  OpTypeMatrix %vec3 3
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, ids_should_be_validated_before_data) {
  std::string str = header + R"(
%f32    =  OpTypeFloat 32
%mat33  =  OpTypeMatrix %vec3 3
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("ID 3[%3] has not been defined"));
}

TEST_F(ValidateData, matrix_bad_column_type) {
  std::string str = header + R"(
%f32    =  OpTypeFloat 32
%mat33  =  OpTypeMatrix %f32 3
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Columns in a matrix must be of type vector"));
}

TEST_F(ValidateData, matrix_data_type_int) {
  std::string str = header + R"(
%int32  =  OpTypeInt 32 1
%vec3   =  OpTypeVector %int32 3
%mat33  =  OpTypeMatrix %vec3 3
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("can only be parameterized with floating-point types"));
}

TEST_F(ValidateData, matrix_data_type_bool) {
  std::string str = header + R"(
%boolt  =  OpTypeBool
%vec3   =  OpTypeVector %boolt 3
%mat33  =  OpTypeMatrix %vec3 3
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("can only be parameterized with floating-point types"));
}

TEST_F(ValidateData, matrix_with_0_columns) {
  std::string str = header + R"(
%f32    =  OpTypeFloat 32
%vec3   =  OpTypeVector %f32 3
%mat33  =  OpTypeMatrix %vec3 0
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("can only be parameterized as having only 2, 3, or 4 columns"));
}

TEST_F(ValidateData, matrix_with_1_column) {
  std::string str = header + R"(
%f32    =  OpTypeFloat 32
%vec3   =  OpTypeVector %f32 3
%mat33  =  OpTypeMatrix %vec3 1
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("can only be parameterized as having only 2, 3, or 4 columns"));
}

TEST_F(ValidateData, matrix_with_2_columns) {
  std::string str = header + R"(
%f32    =  OpTypeFloat 32
%vec3   =  OpTypeVector %f32 3
%mat33  =  OpTypeMatrix %vec3 2
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, matrix_with_3_columns) {
  std::string str = header + R"(
%f32    =  OpTypeFloat 32
%vec3   =  OpTypeVector %f32 3
%mat33  =  OpTypeMatrix %vec3 3
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, matrix_with_4_columns) {
  std::string str = header + R"(
%f32    =  OpTypeFloat 32
%vec3   =  OpTypeVector %f32 3
%mat33  =  OpTypeMatrix %vec3 4
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, matrix_with_5_column) {
  std::string str = header + R"(
%f32    =  OpTypeFloat 32
%vec3   =  OpTypeVector %f32 3
%mat33  =  OpTypeMatrix %vec3 5
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("can only be parameterized as having only 2, 3, or 4 columns"));
}

TEST_F(ValidateData, specialize_int) {
  std::string str = header + R"(
%i32 = OpTypeInt 32 1
%len = OpSpecConstant %i32 2)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, specialize_float) {
  std::string str = header + R"(
%f32 = OpTypeFloat 32
%len = OpSpecConstant %f32 2)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, specialize_boolean) {
  std::string str = header + R"(
%2 = OpTypeBool
%3 = OpSpecConstantTrue %2
%4 = OpSpecConstantFalse %2)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, specialize_boolean_to_int) {
  std::string str = header + R"(
%2 = OpTypeInt 32 1
%3 = OpSpecConstantTrue %2
%4 = OpSpecConstantFalse %2)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Specialization constant must be a boolean"));
}

TEST_F(ValidateData, missing_forward_pointer_decl) {
  std::string str = header_with_addresses + R"(
%uintt = OpTypeInt 32 0
%3 = OpTypeStruct %fwd_ptrt %uintt
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("must first be declared using OpTypeForwardPointer"));
}

TEST_F(ValidateData, missing_forward_pointer_decl_self_reference) {
  std::string str = header_with_addresses + R"(
%uintt = OpTypeInt 32 0
%3 = OpTypeStruct %3 %uintt
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("must first be declared using OpTypeForwardPointer"));
}

TEST_F(ValidateData, forward_pointer_missing_definition) {
  std::string str = header_with_addresses + R"(
OpTypeForwardPointer %_ptr_Generic_struct_A Generic
%uintt = OpTypeInt 32 0
%struct_B = OpTypeStruct %uintt %_ptr_Generic_struct_A
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("forward referenced IDs have not been defined"));
}

TEST_F(ValidateData, forward_ref_bad_type) {
  std::string str = header_with_addresses + R"(
OpTypeForwardPointer %_ptr_Generic_struct_A Generic
%uintt = OpTypeInt 32 0
%struct_B = OpTypeStruct %uintt %_ptr_Generic_struct_A
%_ptr_Generic_struct_A = OpTypeFloat 32
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Pointer type in OpTypeForwardPointer is not a pointer "
                        "type.\n  OpTypeForwardPointer %float Generic\n"));
}

TEST_F(ValidateData, forward_ref_points_to_non_struct) {
  std::string str = header_with_addresses + R"(
OpTypeForwardPointer %_ptr_Generic_struct_A Generic
%uintt = OpTypeInt 32 0
%struct_B = OpTypeStruct %uintt %_ptr_Generic_struct_A
%_ptr_Generic_struct_A = OpTypePointer Generic %uintt
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("A forward reference operand in an OpTypeStruct must "
                        "be an OpTypePointer that points to an OpTypeStruct. "
                        "Found OpTypePointer that points to OpTypeInt."));
}

TEST_F(ValidateData, struct_forward_pointer_good) {
  std::string str = header_with_addresses + R"(
OpTypeForwardPointer %_ptr_Generic_struct_A Generic
%uintt = OpTypeInt 32 0
%struct_B = OpTypeStruct %uintt %_ptr_Generic_struct_A
%struct_C = OpTypeStruct %uintt %struct_B
%struct_A = OpTypeStruct %uintt %struct_C
%_ptr_Generic_struct_A = OpTypePointer Generic %struct_C
)";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, ext_16bit_storage_caps_allow_free_fp_rounding_mode) {
  for (const char* cap : {"StorageUniform16", "StorageUniformBufferBlock16",
                          "StoragePushConstant16", "StorageInputOutput16"}) {
    for (const char* mode : {"RTE", "RTZ", "RTP", "RTN"}) {
      std::string str = std::string(R"(
        OpCapability Shader
        OpCapability Linkage
        OpCapability )") +
                        cap + R"(
        OpExtension "SPV_KHR_storage_buffer_storage_class"
        OpExtension "SPV_KHR_variable_pointers"
        OpExtension "SPV_KHR_16bit_storage"
        OpMemoryModel Logical GLSL450
        OpDecorate %_ FPRoundingMode )" + mode + R"(
        %half = OpTypeFloat 16
        %float = OpTypeFloat 32
        %float_1_25 = OpConstant %float 1.25
        %half_ptr = OpTypePointer StorageBuffer %half
        %half_ptr_var = OpVariable %half_ptr StorageBuffer
        %void = OpTypeVoid
        %func = OpTypeFunction %void
        %main = OpFunction %void None %func
        %main_entry = OpLabel
        %_ = OpFConvert %half %float_1_25
        OpStore %half_ptr_var %_
        OpReturn
        OpFunctionEnd
      )";
      CompileSuccessfully(str.c_str());
      ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
    }
  }
}

TEST_F(ValidateData, vulkan_disallow_free_fp_rounding_mode) {
  for (const char* mode : {"RTE", "RTZ"}) {
    for (const auto env : {SPV_ENV_VULKAN_1_0, SPV_ENV_VULKAN_1_1}) {
      std::string str = std::string(R"(
        OpCapability Shader
        OpExtension "SPV_KHR_storage_buffer_storage_class"
        OpExtension "SPV_KHR_variable_pointers"
        OpMemoryModel Logical GLSL450
        OpDecorate %_ FPRoundingMode )") +
                        mode + R"(
        %half = OpTypeFloat 16
        %float = OpTypeFloat 32
        %float_1_25 = OpConstant %float 1.25
        %half_ptr = OpTypePointer StorageBuffer %half
        %half_ptr_var = OpVariable %half_ptr StorageBuffer
        %void = OpTypeVoid
        %func = OpTypeFunction %void
        %main = OpFunction %void None %func
        %main_entry = OpLabel
        %_ = OpFConvert %half %float_1_25
        OpStore %half_ptr_var %_
        OpReturn
        OpFunctionEnd
      )";
      CompileSuccessfully(str.c_str());
      ASSERT_EQ(SPV_ERROR_INVALID_CAPABILITY, ValidateInstructions(env));
      EXPECT_THAT(
          getDiagnosticString(),
          HasSubstr("Operand 2 of Decorate requires one of these capabilities: "
                    "StorageBuffer16BitAccess StorageUniform16 "
                    "StoragePushConstant16 StorageInputOutput16"));
    }
  }
}

TEST_F(ValidateData, void_array) {
  std::string str = header + R"(
   %void = OpTypeVoid
    %int = OpTypeInt 32 0
  %int_5 = OpConstant %int 5
  %array = OpTypeArray %void %int_5
  )";

  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpTypeArray Element Type <id> '1[%void]' is a void type."));
}

TEST_F(ValidateData, void_runtime_array) {
  std::string str = header + R"(
   %void = OpTypeVoid
  %array = OpTypeRuntimeArray %void
  )";

  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpTypeRuntimeArray Element Type <id> '1[%void]' is a void type."));
}

TEST_F(ValidateData, vulkan_RTA_array_at_end_of_struct) {
  std::string str = R"(
              OpCapability Shader
              OpMemoryModel Logical GLSL450
              OpEntryPoint Fragment %func "func"
              OpExecutionMode %func OriginUpperLeft
              OpDecorate %array_t ArrayStride 4
              OpMemberDecorate %struct_t 0 Offset 0
              OpMemberDecorate %struct_t 1 Offset 4
              OpDecorate %struct_t Block
     %uint_t = OpTypeInt 32 0
   %array_t = OpTypeRuntimeArray %uint_t
  %struct_t = OpTypeStruct %uint_t %array_t
%struct_ptr = OpTypePointer StorageBuffer %struct_t
         %2 = OpVariable %struct_ptr StorageBuffer
      %void = OpTypeVoid
    %func_t = OpTypeFunction %void
      %func = OpFunction %void None %func_t
         %1 = OpLabel
              OpReturn
              OpFunctionEnd
)";

  CompileSuccessfully(str.c_str(), SPV_ENV_VULKAN_1_1);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateData, vulkan_RTA_not_at_end_of_struct) {
  std::string str = R"(
              OpCapability Shader
              OpMemoryModel Logical GLSL450
              OpEntryPoint Fragment %func "func"
              OpExecutionMode %func OriginUpperLeft
              OpDecorate %array_t ArrayStride 4
              OpMemberDecorate %struct_t 0 Offset 0
              OpMemberDecorate %struct_t 1 Offset 4
              OpDecorate %struct_t Block
     %uint_t = OpTypeInt 32 0
   %array_t = OpTypeRuntimeArray %uint_t
  %struct_t = OpTypeStruct %array_t %uint_t
%struct_ptr = OpTypePointer StorageBuffer %struct_t
         %2 = OpVariable %struct_ptr StorageBuffer
      %void = OpTypeVoid
    %func_t = OpTypeFunction %void
      %func = OpFunction %void None %func_t
         %1 = OpLabel
              OpReturn
              OpFunctionEnd
)";

  CompileSuccessfully(str.c_str(), SPV_ENV_VULKAN_1_1);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("In Vulkan, OpTypeRuntimeArray must only be used for "
                        "the last member of an OpTypeStruct\n  %_struct_3 = "
                        "OpTypeStruct %_runtimearr_uint %uint\n"));
}

TEST_F(ValidateData, webgpu_RTA_array_at_end_of_struct) {
  std::string str = R"(
              OpCapability Shader
              OpCapability VulkanMemoryModelKHR
              OpExtension "SPV_KHR_vulkan_memory_model"
              OpMemoryModel Logical VulkanKHR
              OpEntryPoint Fragment %func "func"
              OpExecutionMode %func OriginUpperLeft
              OpDecorate %array_t ArrayStride 4
              OpMemberDecorate %struct_t 0 Offset 0
              OpMemberDecorate %struct_t 1 Offset 4
              OpDecorate %struct_t Block
     %uint_t = OpTypeInt 32 0
   %array_t = OpTypeRuntimeArray %uint_t
  %struct_t = OpTypeStruct %uint_t %array_t
%struct_ptr = OpTypePointer StorageBuffer %struct_t
         %2 = OpVariable %struct_ptr StorageBuffer
      %void = OpTypeVoid
    %func_t = OpTypeFunction %void
      %func = OpFunction %void None %func_t
         %1 = OpLabel
              OpReturn
              OpFunctionEnd
)";

  CompileSuccessfully(str.c_str(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateData, webgpu_RTA_not_at_end_of_struct) {
  std::string str = R"(
              OpCapability Shader
              OpCapability VulkanMemoryModelKHR
              OpExtension "SPV_KHR_vulkan_memory_model"
              OpMemoryModel Logical VulkanKHR
              OpEntryPoint Fragment %func "func"
              OpExecutionMode %func OriginUpperLeft
              OpDecorate %array_t ArrayStride 4
              OpMemberDecorate %struct_t 0 Offset 0
              OpMemberDecorate %struct_t 1 Offset 4
              OpDecorate %struct_t Block
     %uint_t = OpTypeInt 32 0
   %array_t = OpTypeRuntimeArray %uint_t
  %struct_t = OpTypeStruct %array_t %uint_t
%struct_ptr = OpTypePointer StorageBuffer %struct_t
         %2 = OpVariable %struct_ptr StorageBuffer
      %void = OpTypeVoid
    %func_t = OpTypeFunction %void
      %func = OpFunction %void None %func_t
         %1 = OpLabel
              OpReturn
              OpFunctionEnd
)";

  CompileSuccessfully(str.c_str(), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("In WebGPU, OpTypeRuntimeArray must only be used for "
                        "the last member of an OpTypeStruct\n  %_struct_3 = "
                        "OpTypeStruct %_runtimearr_uint %uint\n"));
}

TEST_F(ValidateData, invalid_forward_reference_in_array) {
  std::string str = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
       %uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%_arr_3_uint_1 = OpTypeArray %_arr_3_uint_1 %uint_1
)";

  CompileSuccessfully(str.c_str(), SPV_ENV_UNIVERSAL_1_3);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Forward reference operands in an OpTypeArray must "
                        "first be declared using OpTypeForwardPointer."));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
