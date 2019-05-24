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

// Validation tests for Universal Limits. (Section 2.17 of the SPIR-V Spec)

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

using ValidateLimits = spvtest::ValidateBase<bool>;

std::string header = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
)";

TEST_F(ValidateLimits, IdLargerThanBoundBad) {
  std::string str = header + R"(
;  %i32 has ID 1
%i32    = OpTypeInt 32 1
%c      = OpConstant %i32 100

; Fake an instruction with 64 as the result id.
; !64 = OpConstantNull %i32
!0x3002e !1 !64
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Result <id> '64' must be less than the ID bound '3'."));
}

TEST_F(ValidateLimits, IdEqualToBoundBad) {
  std::string str = header + R"(
;  %i32 has ID 1
%i32    = OpTypeInt 32 1
%c      = OpConstant %i32 100

; Fake an instruction with 64 as the result id.
; !64 = OpConstantNull %i32
!0x3002e !1 !64
)";

  CompileSuccessfully(str);

  // The largest ID used in this program is 64. Let's overwrite the ID bound in
  // the header to be 64. This should result in an error because all IDs must
  // satisfy: 0 < id < bound.
  OverwriteAssembledBinary(3, 64);

  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Result <id> '64' must be less than the ID bound '64'."));
}

TEST_F(ValidateLimits, IdBoundTooBigDeaultLimit) {
  std::string str = header;

  CompileSuccessfully(str);

  // The largest ID used in this program is 64. Let's overwrite the ID bound in
  // the header to be 64. This should result in an error because all IDs must
  // satisfy: 0 < id < bound.
  OverwriteAssembledBinary(3, 0x4FFFFF);

  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Invalid SPIR-V.  The id bound is larger than the max "
                        "id bound 4194303."));
}

TEST_F(ValidateLimits, IdBoundAtSetLimit) {
  std::string str = header;

  CompileSuccessfully(str);

  // The largest ID used in this program is 64. Let's overwrite the ID bound in
  // the header to be 64. This should result in an error because all IDs must
  // satisfy: 0 < id < bound.
  uint32_t id_bound = 0x4FFFFF;

  OverwriteAssembledBinary(3, id_bound);
  getValidatorOptions()->universal_limits_.max_id_bound = id_bound;

  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLimits, IdBoundJustAboveSetLimit) {
  std::string str = header;

  CompileSuccessfully(str);

  // The largest ID used in this program is 64. Let's overwrite the ID bound in
  // the header to be 64. This should result in an error because all IDs must
  // satisfy: 0 < id < bound.
  uint32_t id_bound = 5242878;

  OverwriteAssembledBinary(3, id_bound);
  getValidatorOptions()->universal_limits_.max_id_bound = id_bound - 1;

  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Invalid SPIR-V.  The id bound is larger than the max "
                        "id bound 5242877."));
}

TEST_F(ValidateLimits, IdBoundAtInMaxLimit) {
  std::string str = header;

  CompileSuccessfully(str);

  uint32_t id_bound = std::numeric_limits<uint32_t>::max();

  OverwriteAssembledBinary(3, id_bound);
  getValidatorOptions()->universal_limits_.max_id_bound = id_bound;

  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLimits, StructNumMembersGood) {
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeInt 32 0
%2 = OpTypeStruct)";
  for (int i = 0; i < 16383; ++i) {
    spirv << " %1";
  }
  CompileSuccessfully(spirv.str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLimits, StructNumMembersExceededBad) {
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeInt 32 0
%2 = OpTypeStruct)";
  for (int i = 0; i < 16384; ++i) {
    spirv << " %1";
  }
  CompileSuccessfully(spirv.str());
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of OpTypeStruct members (16384) has exceeded "
                        "the limit (16383)."));
}

TEST_F(ValidateLimits, CustomizedStructNumMembersGood) {
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeInt 32 0
%2 = OpTypeStruct)";
  for (int i = 0; i < 32000; ++i) {
    spirv << " %1";
  }
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_struct_members, 32000u);
  CompileSuccessfully(spirv.str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLimits, CustomizedStructNumMembersBad) {
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeInt 32 0
%2 = OpTypeStruct)";
  for (int i = 0; i < 32001; ++i) {
    spirv << " %1";
  }
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_struct_members, 32000u);
  CompileSuccessfully(spirv.str());
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of OpTypeStruct members (32001) has exceeded "
                        "the limit (32000)."));
}

// Valid: Switch statement has 16,383 branches.
TEST_F(ValidateLimits, SwitchNumBranchesGood) {
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeInt 32 0
%4 = OpConstant %3 1234
%5 = OpFunction %1 None %2
%7 = OpLabel
%8 = OpIAdd %3 %4 %4
%9 = OpSwitch %4 %10)";

  // Now add the (literal, label) pairs
  for (int i = 0; i < 16383; ++i) {
    spirv << " 1 %10";
  }

  spirv << R"(
%10 = OpLabel
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Switch statement has 16,384 branches.
TEST_F(ValidateLimits, SwitchNumBranchesBad) {
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeInt 32 0
%4 = OpConstant %3 1234
%5 = OpFunction %1 None %2
%7 = OpLabel
%8 = OpIAdd %3 %4 %4
%9 = OpSwitch %4 %10)";

  // Now add the (literal, label) pairs
  for (int i = 0; i < 16384; ++i) {
    spirv << " 1 %10";
  }

  spirv << R"(
%10 = OpLabel
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of (literal, label) pairs in OpSwitch (16384) "
                        "exceeds the limit (16383)."));
}

// Valid: Switch statement has 10 branches (limit is 10)
TEST_F(ValidateLimits, CustomizedSwitchNumBranchesGood) {
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeInt 32 0
%4 = OpConstant %3 1234
%5 = OpFunction %1 None %2
%7 = OpLabel
%8 = OpIAdd %3 %4 %4
%9 = OpSwitch %4 %10)";

  // Now add the (literal, label) pairs
  for (int i = 0; i < 10; ++i) {
    spirv << " 1 %10";
  }

  spirv << R"(
%10 = OpLabel
OpReturn
OpFunctionEnd
  )";

  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_switch_branches, 10u);
  CompileSuccessfully(spirv.str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Switch statement has 11 branches (limit is 10)
TEST_F(ValidateLimits, CustomizedSwitchNumBranchesBad) {
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeInt 32 0
%4 = OpConstant %3 1234
%5 = OpFunction %1 None %2
%7 = OpLabel
%8 = OpIAdd %3 %4 %4
%9 = OpSwitch %4 %10)";

  // Now add the (literal, label) pairs
  for (int i = 0; i < 11; ++i) {
    spirv << " 1 %10";
  }

  spirv << R"(
%10 = OpLabel
OpReturn
OpFunctionEnd
  )";

  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_switch_branches, 10u);
  CompileSuccessfully(spirv.str());
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of (literal, label) pairs in OpSwitch (11) "
                        "exceeds the limit (10)."));
}

// Valid: OpTypeFunction with 255 arguments.
TEST_F(ValidateLimits, OpTypeFunctionGood) {
  int num_args = 255;
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeInt 32 0
%2 = OpTypeFunction %1)";
  // add parameters
  for (int i = 0; i < num_args; ++i) {
    spirv << " %1";
  }
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: OpTypeFunction with 256 arguments. (limit is 255 according to the
// spec Universal Limits (2.17).
TEST_F(ValidateLimits, OpTypeFunctionBad) {
  int num_args = 256;
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeInt 32 0
%2 = OpTypeFunction %1)";
  for (int i = 0; i < num_args; ++i) {
    spirv << " %1";
  }
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpTypeFunction may not take more than 255 arguments. "
                        "OpTypeFunction <id> '2[%2]' has 256 arguments."));
}

// Valid: OpTypeFunction with 100 arguments (Custom limit: 100)
TEST_F(ValidateLimits, CustomizedOpTypeFunctionGood) {
  int num_args = 100;
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeInt 32 0
%2 = OpTypeFunction %1)";
  // add parameters
  for (int i = 0; i < num_args; ++i) {
    spirv << " %1";
  }
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_function_args, 100u);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: OpTypeFunction with 101 arguments. (Custom limit: 100)
TEST_F(ValidateLimits, CustomizedOpTypeFunctionBad) {
  int num_args = 101;
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeInt 32 0
%2 = OpTypeFunction %1)";
  for (int i = 0; i < num_args; ++i) {
    spirv << " %1";
  }
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_function_args, 100u);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpTypeFunction may not take more than 100 arguments. "
                        "OpTypeFunction <id> '2[%2]' has 101 arguments."));
}

// Valid: module has 65,535 global variables.
TEST_F(ValidateLimits, NumGlobalVarsGood) {
  int num_globals = 65535;
  std::ostringstream spirv;
  spirv << header << R"(
     %int = OpTypeInt 32 0
%_ptr_int = OpTypePointer Input %int
  )";

  for (int i = 0; i < num_globals; ++i) {
    spirv << "%var_" << i << " = OpVariable %_ptr_int Input\n";
  }

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: module has 65,536 global variables (limit is 65,535).
TEST_F(ValidateLimits, NumGlobalVarsBad) {
  int num_globals = 65536;
  std::ostringstream spirv;
  spirv << header << R"(
     %int = OpTypeInt 32 0
%_ptr_int = OpTypePointer Input %int
  )";

  for (int i = 0; i < num_globals; ++i) {
    spirv << "%var_" << i << " = OpVariable %_ptr_int Input\n";
  }

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of Global Variables (Storage Class other than "
                        "'Function') exceeded the valid limit (65535)."));
}

// Valid: module has 50 global variables (limit is 50)
TEST_F(ValidateLimits, CustomizedNumGlobalVarsGood) {
  int num_globals = 50;
  std::ostringstream spirv;
  spirv << header << R"(
     %int = OpTypeInt 32 0
%_ptr_int = OpTypePointer Input %int
  )";

  for (int i = 0; i < num_globals; ++i) {
    spirv << "%var_" << i << " = OpVariable %_ptr_int Input\n";
  }

  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_global_variables, 50u);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: module has 51 global variables (limit is 50).
TEST_F(ValidateLimits, CustomizedNumGlobalVarsBad) {
  int num_globals = 51;
  std::ostringstream spirv;
  spirv << header << R"(
     %int = OpTypeInt 32 0
%_ptr_int = OpTypePointer Input %int
  )";

  for (int i = 0; i < num_globals; ++i) {
    spirv << "%var_" << i << " = OpVariable %_ptr_int Input\n";
  }

  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_global_variables, 50u);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of Global Variables (Storage Class other than "
                        "'Function') exceeded the valid limit (50)."));
}

// Valid: module has 524,287 local variables.
// Note: AppVeyor limits process time to 300s.  For a VisualStudio Debug
// build, going up to 524287 local variables gets too close to that
// limit.  So test with an artificially lowered limit.
TEST_F(ValidateLimits, NumLocalVarsGoodArtificiallyLowLimit5K) {
  int num_locals = 5000;
  std::ostringstream spirv;
  spirv << header << R"(
 %int      = OpTypeInt 32 0
 %_ptr_int = OpTypePointer Function %int
 %voidt    = OpTypeVoid
 %funct    = OpTypeFunction %voidt
 %main     = OpFunction %voidt None %funct
 %entry    = OpLabel
  )";

  for (int i = 0; i < num_locals; ++i) {
    spirv << "%var_" << i << " = OpVariable %_ptr_int Function\n";
  }

  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  // Artificially limit it.
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_local_variables, num_locals);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: module has 524,288 local variables (limit is 524,287).
// Artificially limit the check to 5001.
TEST_F(ValidateLimits, NumLocalVarsBadArtificiallyLowLimit5K) {
  int num_locals = 5001;
  std::ostringstream spirv;
  spirv << header << R"(
 %int      = OpTypeInt 32 0
 %_ptr_int = OpTypePointer Function %int
 %voidt    = OpTypeVoid
 %funct    = OpTypeFunction %voidt
 %main     = OpFunction %voidt None %funct
 %entry    = OpLabel
  )";

  for (int i = 0; i < num_locals; ++i) {
    spirv << "%var_" << i << " = OpVariable %_ptr_int Function\n";
  }

  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_local_variables, 5000u);
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of local variables ('Function' Storage Class) "
                        "exceeded the valid limit (5000)."));
}

// Valid: module has 100 local variables (limit is 100).
TEST_F(ValidateLimits, CustomizedNumLocalVarsGood) {
  int num_locals = 100;
  std::ostringstream spirv;
  spirv << header << R"(
 %int      = OpTypeInt 32 0
 %_ptr_int = OpTypePointer Function %int
 %voidt    = OpTypeVoid
 %funct    = OpTypeFunction %voidt
 %main     = OpFunction %voidt None %funct
 %entry    = OpLabel
  )";

  for (int i = 0; i < num_locals; ++i) {
    spirv << "%var_" << i << " = OpVariable %_ptr_int Function\n";
  }

  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";

  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_local_variables, 100u);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: module has 101 local variables (limit is 100).
TEST_F(ValidateLimits, CustomizedNumLocalVarsBad) {
  int num_locals = 101;
  std::ostringstream spirv;
  spirv << header << R"(
 %int      = OpTypeInt 32 0
 %_ptr_int = OpTypePointer Function %int
 %voidt    = OpTypeVoid
 %funct    = OpTypeFunction %voidt
 %main     = OpFunction %voidt None %funct
 %entry    = OpLabel
  )";

  for (int i = 0; i < num_locals; ++i) {
    spirv << "%var_" << i << " = OpVariable %_ptr_int Function\n";
  }

  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";

  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_local_variables, 100u);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of local variables ('Function' Storage Class) "
                        "exceeded the valid limit (100)."));
}

// Valid: Structure nesting depth of 255.
TEST_F(ValidateLimits, StructNestingDepthGood) {
  std::ostringstream spirv;
  spirv << header << R"(
    %int = OpTypeInt 32 0
    %s_depth_1  = OpTypeStruct %int
  )";
  for (auto i = 2; i <= 255; ++i) {
    spirv << "%s_depth_" << i << " = OpTypeStruct %int %s_depth_" << i - 1;
    spirv << "\n";
  }
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Structure nesting depth of 256.
TEST_F(ValidateLimits, StructNestingDepthBad) {
  std::ostringstream spirv;
  spirv << header << R"(
    %int = OpTypeInt 32 0
    %s_depth_1  = OpTypeStruct %int
  )";
  for (auto i = 2; i <= 256; ++i) {
    spirv << "%s_depth_" << i << " = OpTypeStruct %int %s_depth_" << i - 1;
    spirv << "\n";
  }
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Structure Nesting Depth may not be larger than 255. Found 256."));
}

// Valid: Structure nesting depth of 100 (limit is 100).
TEST_F(ValidateLimits, CustomizedStructNestingDepthGood) {
  std::ostringstream spirv;
  spirv << header << R"(
    %int = OpTypeInt 32 0
    %s_depth_1  = OpTypeStruct %int
  )";
  for (auto i = 2; i <= 100; ++i) {
    spirv << "%s_depth_" << i << " = OpTypeStruct %int %s_depth_" << i - 1;
    spirv << "\n";
  }
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_struct_depth, 100u);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Structure nesting depth of 101 (limit is 100).
TEST_F(ValidateLimits, CustomizedStructNestingDepthBad) {
  std::ostringstream spirv;
  spirv << header << R"(
    %int = OpTypeInt 32 0
    %s_depth_1  = OpTypeStruct %int
  )";
  for (auto i = 2; i <= 101; ++i) {
    spirv << "%s_depth_" << i << " = OpTypeStruct %int %s_depth_" << i - 1;
    spirv << "\n";
  }
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_struct_depth, 100u);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Structure Nesting Depth may not be larger than 100. Found 101."));
}

// clang-format off
// Generates an SPIRV program with the given control flow nesting depth
void GenerateSpirvProgramWithCfgNestingDepth(std::string& str, int depth) {
  std::ostringstream spirv;
  spirv << header << R"(
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %bool = OpTypeBool
         %12 = OpConstantTrue %bool
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpLoopMerge %8 %9 None
               OpBranch %10
         %10 = OpLabel
               OpBranchConditional %12 %7 %8
          %7 = OpLabel
  )";
  int first_id = 13;
  int last_id = 14;
  // We already have 1 level of nesting due to the Loop.
  int num_if_conditions = depth-1;
  int largest_index = first_id + 2*num_if_conditions - 2;
  for (int i = first_id; i <= largest_index; i = i + 2) {
    spirv << "OpSelectionMerge %" << i+1 << " None" << "\n";
    spirv << "OpBranchConditional %12 " << "%" << i << " %" << i+1 << "\n";
    spirv << "%" << i << " = OpLabel" << "\n";
  }
  spirv << "OpBranch %9" << "\n";

  for (int i = largest_index+1; i > last_id; i = i - 2) {
    spirv << "%" << i << " = OpLabel" << "\n";
    spirv << "OpBranch %" << i-2 << "\n";
  }
  spirv << "%" << last_id << " = OpLabel" << "\n";
  spirv << "OpBranch %9" << "\n";
  spirv << R"(
    %9 = OpLabel
    OpBranch %6
    %8 = OpLabel
    OpReturn
    OpFunctionEnd
  )";
  str = spirv.str();
}
// clang-format on

// Invalid: Control Flow Nesting depth is 1024. (limit is 1023).
TEST_F(ValidateLimits, ControlFlowDepthBad) {
  std::string spirv;
  GenerateSpirvProgramWithCfgNestingDepth(spirv, 1024);
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Maximum Control Flow nesting depth exceeded."));
}

// Valid: Control Flow Nesting depth is 10 (custom limit: 10).
TEST_F(ValidateLimits, CustomizedControlFlowDepthGood) {
  std::string spirv;
  GenerateSpirvProgramWithCfgNestingDepth(spirv, 10);
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_control_flow_nesting_depth, 10u);
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Control Flow Nesting depth is 11. (custom limit: 10).
TEST_F(ValidateLimits, CustomizedControlFlowDepthBad) {
  std::string spirv;
  GenerateSpirvProgramWithCfgNestingDepth(spirv, 11);
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_control_flow_nesting_depth, 10u);
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Maximum Control Flow nesting depth exceeded."));
}

// Valid. The purpose here is to test the CFG depth calculation code when a loop
// continue target is the loop iteself. It also exercises the case where a loop
// is unreachable.
TEST_F(ValidateLimits, ControlFlowNoEntryToLoopGood) {
  std::string str = header + R"(
           OpName %entry "entry"
           OpName %loop "loop"
           OpName %exit "exit"
%voidt   = OpTypeVoid
%funct   = OpTypeFunction %voidt
%main    = OpFunction %voidt None %funct
%entry   = OpLabel
           OpBranch %exit
%loop    = OpLabel
           OpLoopMerge %loop %loop None
           OpBranch %loop
%exit    = OpLabel
           OpReturn
           OpFunctionEnd
  )";
  CompileSuccessfully(str);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

}  // namespace
}  // namespace val
}  // namespace spvtools
