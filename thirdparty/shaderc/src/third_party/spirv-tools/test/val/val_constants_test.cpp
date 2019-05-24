// Copyright (c) 2019 Google LLC
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

// Test validation of constants.
//
// This file contains newer tests.  Older tests may be in other files such as
// val_id_test.cpp.

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
using ::testing::ValuesIn;

using ValidateConstant = spvtest::ValidateBase<bool>;

#define kBasicTypes                             \
  "%bool = OpTypeBool "                         \
  "%uint = OpTypeInt 32 0 "                     \
  "%uint2 = OpTypeVector %uint 2 "              \
  "%float = OpTypeFloat 32 "                    \
  "%_ptr_uint = OpTypePointer Workgroup %uint " \
  "%uint_0 = OpConstantNull %uint "             \
  "%uint2_0 = OpConstantNull %uint "            \
  "%float_0 = OpConstantNull %float "           \
  "%false = OpConstantFalse %bool "             \
  "%true = OpConstantTrue %bool "               \
  "%null = OpConstantNull %_ptr_uint "

#define kShaderPreamble    \
  "OpCapability Shader\n"  \
  "OpCapability Linkage\n" \
  "OpMemoryModel Logical Simple\n"

#define kKernelPreamble      \
  "OpCapability Kernel\n"    \
  "OpCapability Linkage\n"   \
  "OpCapability Addresses\n" \
  "OpMemoryModel Physical32 OpenCL\n"

struct ConstantOpCase {
  spv_target_env env;
  std::string assembly;
  bool expect_success;
  std::string expect_err;
};

using ValidateConstantOp = spvtest::ValidateBase<ConstantOpCase>;

TEST_P(ValidateConstantOp, Samples) {
  const auto env = GetParam().env;
  CompileSuccessfully(GetParam().assembly, env);
  const auto result = ValidateInstructions(env);
  if (GetParam().expect_success) {
    EXPECT_EQ(SPV_SUCCESS, result);
    EXPECT_THAT(getDiagnosticString(), Eq(""));
  } else {
    EXPECT_EQ(SPV_ERROR_INVALID_ID, result);
    EXPECT_THAT(getDiagnosticString(), HasSubstr(GetParam().expect_err));
  }
}

#define GOOD_SHADER_10(STR) \
  { SPV_ENV_UNIVERSAL_1_0, kShaderPreamble kBasicTypes STR, true, "" }
#define GOOD_KERNEL_10(STR) \
  { SPV_ENV_UNIVERSAL_1_0, kKernelPreamble kBasicTypes STR, true, "" }
INSTANTIATE_TEST_SUITE_P(
    UniversalInShader, ValidateConstantOp,
    ValuesIn(std::vector<ConstantOpCase>{
        // TODO(dneto): Conversions must change width.
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint SConvert %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %float FConvert %float_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint SNegate %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint Not %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint IAdd %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint ISub %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint IMul %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint UDiv %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint SDiv %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint UMod %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint SRem %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint SMod %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint ShiftRightLogical %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint ShiftRightArithmetic %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint ShiftLeftLogical %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint BitwiseOr %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint BitwiseXor %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint2 VectorShuffle %uint2_0 %uint2_0 1 3"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint CompositeExtract %uint2_0 1"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint2 CompositeInsert %uint_0 %uint2_0 1"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool LogicalOr %true %false"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool LogicalNot %true"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool LogicalAnd %true %false"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool LogicalEqual %true %false"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool LogicalNotEqual %true %false"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint Select %true %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool IEqual %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool INotEqual %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool ULessThan %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool SLessThan %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool ULessThanEqual %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool SLessThanEqual %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool UGreaterThan %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool UGreaterThanEqual %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool SGreaterThan %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool SGreaterThanEqual %uint_0 %uint_0"),
    }));

INSTANTIATE_TEST_SUITE_P(
    UniversalInKernel, ValidateConstantOp,
    ValuesIn(std::vector<ConstantOpCase>{
        // TODO(dneto): Conversions must change width.
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint SConvert %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FConvert %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint SNegate %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint Not %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint IAdd %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint ISub %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint IMul %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint UDiv %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint SDiv %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint UMod %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint SRem %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint SMod %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint ShiftRightLogical %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint ShiftRightArithmetic %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint ShiftLeftLogical %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint BitwiseOr %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint BitwiseXor %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint2 VectorShuffle %uint2_0 %uint2_0 1 3"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint CompositeExtract %uint2_0 1"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint2 CompositeInsert %uint_0 %uint2_0 1"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool LogicalOr %true %false"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool LogicalNot %true"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool LogicalAnd %true %false"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool LogicalEqual %true %false"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool LogicalNotEqual %true %false"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint Select %true %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool IEqual %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool INotEqual %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool ULessThan %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool SLessThan %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool ULessThanEqual %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool SLessThanEqual %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool UGreaterThan %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool UGreaterThanEqual %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool SGreaterThan %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool SGreaterThanEqual %uint_0 %uint_0"),
    }));

INSTANTIATE_TEST_SUITE_P(
    KernelInKernel, ValidateConstantOp,
    ValuesIn(std::vector<ConstantOpCase>{
        // TODO(dneto): Conversions must change width.
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint ConvertFToS %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float ConvertSToF %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint ConvertFToU %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float ConvertUToF %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint UConvert %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %_ptr_uint GenericCastToPtr %null"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %_ptr_uint PtrCastToGeneric %null"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint Bitcast %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FNegate %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FAdd %float_0 %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FSub %float_0 %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FMul %float_0 %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FDiv %float_0 %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FRem %float_0 %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FMod %float_0 %float_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %_ptr_uint AccessChain %null %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %_ptr_uint InBoundsAccessChain "
                       "%null %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %_ptr_uint PtrAccessChain %null %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %_ptr_uint "
                       "InBoundsPtrAccessChain %null %uint_0"),
    }));

#define BAD_SHADER_10(STR, NAME)                                   \
  {                                                                \
    SPV_ENV_UNIVERSAL_1_0, kShaderPreamble kBasicTypes STR, false, \
        "Specialization constant operation " NAME                  \
        " requires Kernel capability"                              \
  }
INSTANTIATE_TEST_SUITE_P(
    KernelInShader, ValidateConstantOp,
    ValuesIn(std::vector<ConstantOpCase>{
        // TODO(dneto): Conversions must change width.
        BAD_SHADER_10("%v = OpSpecConstantOp %uint ConvertFToS %float_0",
                      "ConvertFToS"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float ConvertSToF %uint_0",
                      "ConvertSToF"),
        BAD_SHADER_10("%v = OpSpecConstantOp %uint ConvertFToU %float_0",
                      "ConvertFToU"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float ConvertUToF %uint_0",
                      "ConvertUToF"),
        BAD_SHADER_10("%v = OpSpecConstantOp %_ptr_uint GenericCastToPtr %null",
                      "GenericCastToPtr"),
        BAD_SHADER_10("%v = OpSpecConstantOp %_ptr_uint PtrCastToGeneric %null",
                      "PtrCastToGeneric"),
        BAD_SHADER_10("%v = OpSpecConstantOp %uint Bitcast %uint_0", "Bitcast"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FNegate %float_0",
                      "FNegate"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FAdd %float_0 %float_0",
                      "FAdd"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FSub %float_0 %float_0",
                      "FSub"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FMul %float_0 %float_0",
                      "FMul"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FDiv %float_0 %float_0",
                      "FDiv"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FRem %float_0 %float_0",
                      "FRem"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FMod %float_0 %float_0",
                      "FMod"),
        BAD_SHADER_10(
            "%v = OpSpecConstantOp %_ptr_uint AccessChain %null %uint_0",
            "AccessChain"),
        BAD_SHADER_10("%v = OpSpecConstantOp %_ptr_uint InBoundsAccessChain "
                      "%null %uint_0",
                      "InBoundsAccessChain"),
        BAD_SHADER_10(
            "%v = OpSpecConstantOp %_ptr_uint PtrAccessChain %null %uint_0",
            "PtrAccessChain"),
        BAD_SHADER_10("%v = OpSpecConstantOp %_ptr_uint "
                      "InBoundsPtrAccessChain %null %uint_0",
                      "InBoundsPtrAccessChain"),
    }));

INSTANTIATE_TEST_SUITE_P(
    UConvertInAMD_gpu_shader_int16, ValidateConstantOp,
    ValuesIn(std::vector<ConstantOpCase>{
        // SPV_AMD_gpu_shader_int16 should enable UConvert for OpSpecConstantOp
        // https://github.com/KhronosGroup/glslang/issues/848
        {SPV_ENV_UNIVERSAL_1_0,
         "OpCapability Shader "
         "OpCapability Linkage ; So we don't need to define a function\n"
         "OpExtension \"SPV_AMD_gpu_shader_int16\" "
         "OpMemoryModel Logical Simple " kBasicTypes
         "%v = OpSpecConstantOp %uint UConvert %uint_0",
         true, ""},
    }));

}  // namespace
}  // namespace val
}  // namespace spvtools
