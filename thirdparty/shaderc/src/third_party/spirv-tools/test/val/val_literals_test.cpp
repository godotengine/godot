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

// Validation tests for ilegal literals

#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;

using ValidateLiterals = spvtest::ValidateBase<std::string>;
using ValidateLiteralsShader = spvtest::ValidateBase<std::string>;
using ValidateLiteralsKernel = spvtest::ValidateBase<std::string>;

std::string GenerateShaderCode() {
  std::string str = R"(
          OpCapability Shader
          OpCapability Linkage
          OpCapability Int16
          OpCapability Int64
          OpCapability Float16
          OpCapability Float64
          OpMemoryModel Logical GLSL450
%int16  = OpTypeInt 16 1
%uint16 = OpTypeInt 16 0
%int32  = OpTypeInt 32 1
%uint32 = OpTypeInt 32 0
%int64  = OpTypeInt 64 1
%uint64 = OpTypeInt 64 0
%half   = OpTypeFloat 16
%float  = OpTypeFloat 32
%double = OpTypeFloat 64
%10     = OpTypeVoid
    )";
  return str;
}

std::string GenerateKernelCode() {
  std::string str = R"(
          OpCapability Kernel
          OpCapability Addresses
          OpCapability Linkage
          OpCapability Int8
          OpMemoryModel Physical64 OpenCL
%uint8  = OpTypeInt 8 0
    )";
  return str;
}

TEST_F(ValidateLiterals, LiteralsShaderGood) {
  std::string str = GenerateShaderCode() + R"(
%11 = OpConstant %int16   !0x00007FFF
%12 = OpConstant %int16   !0xFFFF8000
%13 = OpConstant %int16   !0xFFFFABCD
%14 = OpConstant %uint16  !0x0000ABCD
%15 = OpConstant %int16  -32768
%16 = OpConstant %uint16  65535
%17 = OpConstant %int32  -2147483648
%18 = OpConstant %uint32  4294967295
%19 = OpConstant %int64  -9223372036854775808
%20 = OpConstant %uint64  18446744073709551615
%21 = OpConstant %half    !0x0000FFFF
%22 = OpConstant %float   !0xFFFFFFFF
%23 = OpConstant %double  !0xFFFFFFFF !0xFFFFFFFF
  )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateLiteralsShader, LiteralsShaderBad) {
  std::string str = GenerateShaderCode() + GetParam();
  std::string inst_id = "11";
  CompileSuccessfully(str);
  EXPECT_EQ(SPV_ERROR_INVALID_VALUE, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("The high-order bits of a literal number in instruction <id> " +
                inst_id +
                " must be 0 for a floating-point type, "
                "or 0 for an integer type with Signedness of 0, "
                "or sign extended when Signedness is 1"));
}

INSTANTIATE_TEST_SUITE_P(
    LiteralsShaderCases, ValidateLiteralsShader,
    ::testing::Values("%11 = OpConstant %int16  !0xFFFF0000",  // Sign bit is 0
                      "%11 = OpConstant %int16  !0x00008000",  // Sign bit is 1
                      "%11 = OpConstant %int16  !0xABCD8000",  // Sign bit is 1
                      "%11 = OpConstant %int16  !0xABCD0000",
                      "%11 = OpConstant %uint16 !0xABCD0000",
                      "%11 = OpConstant %half   !0xABCD0000",
                      "%11 = OpConstant %half   !0x00010000"));

TEST_F(ValidateLiterals, LiteralsKernelGood) {
  std::string str = GenerateKernelCode() + R"(
%4  = OpConstant %uint8  !0x000000AB
%6  = OpConstant %uint8  255
  )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateLiteralsKernel, LiteralsKernelBad) {
  std::string str = GenerateKernelCode() + GetParam();
  std::string inst_id = "2";
  CompileSuccessfully(str);
  EXPECT_EQ(SPV_ERROR_INVALID_VALUE, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("The high-order bits of a literal number in instruction <id> " +
                inst_id +
                " must be 0 for a floating-point type, "
                "or 0 for an integer type with Signedness of 0, "
                "or sign extended when Signedness is 1"));
}

INSTANTIATE_TEST_SUITE_P(
    LiteralsKernelCases, ValidateLiteralsKernel,
    ::testing::Values("%2 = OpConstant %uint8  !0xABCDEF00",
                      "%2 = OpConstant %uint8  !0xABCDEFFF"));

}  // namespace
}  // namespace val
}  // namespace spvtools
