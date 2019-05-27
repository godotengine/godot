// Copyright (c) 2018 LunarG Inc.
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

#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

using ValidateAdjacency = spvtest::ValidateBase<bool>;

TEST_F(ValidateAdjacency, OpPhiBeginsModuleFail) {
  const std::string module = R"(
%result = OpPhi %bool %true %true_label %false %false_label
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%func = OpTypeFunction %void
%main = OpFunction %void None %func
%main_entry = OpLabel
OpBranch %true_label
%true_label = OpLabel
OpBranch %false_label
%false_label = OpLabel
OpBranch %end_label
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(module);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("ID 1[%bool] has not been defined"));
}

TEST_F(ValidateAdjacency, OpLoopMergeEndsModuleFail) {
  const std::string module = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%func = OpTypeFunction %void
%main = OpFunction %void None %func
%main_entry = OpLabel
OpBranch %loop
%loop = OpLabel
OpLoopMerge %end %loop None
)";

  CompileSuccessfully(module);
  EXPECT_EQ(SPV_ERROR_INVALID_LAYOUT, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Missing OpFunctionEnd at end of module"));
}

TEST_F(ValidateAdjacency, OpSelectionMergeEndsModuleFail) {
  const std::string module = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%func = OpTypeFunction %void
%main = OpFunction %void None %func
%main_entry = OpLabel
OpBranch %merge
%merge = OpLabel
OpSelectionMerge %merge None
)";

  CompileSuccessfully(module);
  EXPECT_EQ(SPV_ERROR_INVALID_LAYOUT, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Missing OpFunctionEnd at end of module"));
}

std::string GenerateShaderCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "OpCapability Shader",
    const std::string& execution_model = "Fragment") {
  std::ostringstream ss;
  ss << capabilities_and_extensions << "\n";
  ss << "OpMemoryModel Logical GLSL450\n";
  ss << "OpEntryPoint " << execution_model << " %main \"main\"\n";
  if (execution_model == "Fragment") {
    ss << "OpExecutionMode %main OriginUpperLeft\n";
  }

  ss << R"(
%string = OpString ""
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 0
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%zero = OpConstant %int 0
%int_1 = OpConstant %int 1
%func = OpTypeFunction %void
%func_int = OpTypePointer Function %int
%main = OpFunction %void None %func
%main_entry = OpLabel
)";

  ss << body;

  ss << R"(
OpReturn
OpFunctionEnd)";

  return ss.str();
}

TEST_F(ValidateAdjacency, OpPhiPreceededByOpLabelSuccess) {
  const std::string body = R"(
OpSelectionMerge %end_label None
OpBranchConditional %true %true_label %false_label
%true_label = OpLabel
OpBranch %end_label
%false_label = OpLabel
OpBranch %end_label
%end_label = OpLabel
%line = OpLine %string 0 0
%result = OpPhi %bool %true %true_label %false %false_label
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAdjacency, OpPhiPreceededByOpPhiSuccess) {
  const std::string body = R"(
OpSelectionMerge %end_label None
OpBranchConditional %true %true_label %false_label
%true_label = OpLabel
OpBranch %end_label
%false_label = OpLabel
OpBranch %end_label
%end_label = OpLabel
%1 = OpPhi %bool %true %true_label %false %false_label
%2 = OpPhi %bool %true %true_label %false %false_label
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAdjacency, OpPhiPreceededByOpLineSuccess) {
  const std::string body = R"(
OpSelectionMerge %end_label None
OpBranchConditional %true %true_label %false_label
%true_label = OpLabel
OpBranch %end_label
%false_label = OpLabel
OpBranch %end_label
%end_label = OpLabel
%line = OpLine %string 0 0
%result = OpPhi %bool %true %true_label %false %false_label
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAdjacency, OpPhiPreceededByBadOpFail) {
  const std::string body = R"(
OpSelectionMerge %end_label None
OpBranchConditional %true %true_label %false_label
%true_label = OpLabel
OpBranch %end_label
%false_label = OpLabel
OpBranch %end_label
%end_label = OpLabel
OpNop
%result = OpPhi %bool %true %true_label %false %false_label
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpPhi must appear within a non-entry block before all "
                        "non-OpPhi instructions"));
}

TEST_F(ValidateAdjacency, OpPhiPreceededByOpLineAndBadOpFail) {
  const std::string body = R"(
OpSelectionMerge %end_label None
OpBranchConditional %true %true_label %false_label
%true_label = OpLabel
OpBranch %end_label
%false_label = OpLabel
OpBranch %end_label
%end_label = OpLabel
OpNop
OpLine %string 1 1
%result = OpPhi %bool %true %true_label %false %false_label
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpPhi must appear within a non-entry block before all "
                        "non-OpPhi instructions"));
}

TEST_F(ValidateAdjacency, OpPhiFollowedByOpLineGood) {
  const std::string body = R"(
OpSelectionMerge %end_label None
OpBranchConditional %true %true_label %false_label
%true_label = OpLabel
OpBranch %end_label
%false_label = OpLabel
OpBranch %end_label
%end_label = OpLabel
%result = OpPhi %bool %true %true_label %false %false_label
OpLine %string 1 1
OpNop
OpNop
OpLine %string 2 1
OpNop
OpLine %string 3 1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAdjacency, OpPhiMultipleOpLineAndOpPhiFail) {
  const std::string body = R"(
OpSelectionMerge %end_label None
OpBranchConditional %true %true_label %false_label
%true_label = OpLabel
OpBranch %end_label
%false_label = OpLabel
OpBranch %end_label
%end_label = OpLabel
OpLine %string 1 1
%value = OpPhi %int %zero %true_label %int_1 %false_label
OpNop
OpLine %string 2 1
OpNop
OpLine %string 3 1
%result = OpPhi %bool %true %true_label %false %false_label
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpPhi must appear within a non-entry block before all "
                        "non-OpPhi instructions"));
}

TEST_F(ValidateAdjacency, OpPhiMultipleOpLineAndOpPhiGood) {
  const std::string body = R"(
OpSelectionMerge %end_label None
OpBranchConditional %true %true_label %false_label
%true_label = OpLabel
OpBranch %end_label
%false_label = OpLabel
OpBranch %end_label
%end_label = OpLabel
OpLine %string 1 1
%value = OpPhi %int %zero %true_label %int_1 %false_label
OpLine %string 2 1
%result = OpPhi %bool %true %true_label %false %false_label
OpLine %string 3 1
OpNop
OpNop
OpLine %string 4 1
OpNop
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAdjacency, OpPhiInEntryBlockBad) {
  const std::string body = R"(
OpLine %string 1 1
%value = OpPhi %int
OpLine %string 2 1
OpNop
OpLine %string 3 1
OpNop
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpPhi must appear within a non-entry block before all "
                        "non-OpPhi instructions"));
}

TEST_F(ValidateAdjacency, OpVariableInFunctionGood) {
  const std::string body = R"(
OpLine %string 1 1
%var = OpVariable %func_int Function
OpLine %string 2 1
OpNop
OpLine %string 3 1
OpNop
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAdjacency, OpVariableInFunctionMultipleGood) {
  const std::string body = R"(
OpLine %string 1 1
%1 = OpVariable %func_int Function
OpLine %string 2 1
%2 = OpVariable %func_int Function
%3 = OpVariable %func_int Function
OpNop
OpLine %string 3 1
OpNop
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAdjacency, OpVariableInFunctionBad) {
  const std::string body = R"(
%1 = OpUndef %int
%2 = OpVariable %func_int Function
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("All OpVariable instructions in a function must be the "
                        "first instructions"));
}

TEST_F(ValidateAdjacency, OpVariableInFunctionMultipleBad) {
  const std::string body = R"(
OpNop
%1 = OpVariable %func_int Function
OpLine %string 1 1
%2 = OpVariable %func_int Function
OpNop
OpNop
OpLine %string 2 1
%3 = OpVariable %func_int Function
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("All OpVariable instructions in a function must be the "
                        "first instructions"));
}

TEST_F(ValidateAdjacency, OpLoopMergePreceedsOpBranchSuccess) {
  const std::string body = R"(
OpBranch %loop
%loop = OpLabel
OpLoopMerge %end %loop None
OpBranch %loop
%end = OpLabel
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAdjacency, OpLoopMergePreceedsOpBranchConditionalSuccess) {
  const std::string body = R"(
OpBranch %loop
%loop = OpLabel
OpLoopMerge %end %loop None
OpBranchConditional %true %loop %end
%end = OpLabel
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAdjacency, OpLoopMergePreceedsBadOpFail) {
  const std::string body = R"(
OpBranch %loop
%loop = OpLabel
OpLoopMerge %end %loop None
OpNop
OpBranchConditional %true %loop %end
%end = OpLabel
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpLoopMerge must immediately precede either an "
                        "OpBranch or OpBranchConditional instruction."));
}

TEST_F(ValidateAdjacency, OpSelectionMergePreceedsOpBranchConditionalSuccess) {
  const std::string body = R"(
OpSelectionMerge %end_label None
OpBranchConditional %true %true_label %false_label
%true_label = OpLabel
OpBranch %end_label
%false_label = OpLabel
OpBranch %end_label
%end_label = OpLabel
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAdjacency, OpSelectionMergePreceedsOpSwitchSuccess) {
  const std::string body = R"(
OpSelectionMerge %merge None
OpSwitch %zero %merge 0 %label
%label = OpLabel
OpBranch %merge
%merge = OpLabel
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAdjacency, OpSelectionMergePreceedsBadOpFail) {
  const std::string body = R"(
OpSelectionMerge %merge None
OpNop
OpSwitch %zero %merge 0 %label
%label = OpLabel
OpBranch %merge
%merge = OpLabel
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSelectionMerge must immediately precede either an "
                        "OpBranchConditional or OpSwitch instruction"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
