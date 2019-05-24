// Copyright (c) 2015-2016 The Khronos Group Inc.
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

// Validation tests for SSA

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

using ValidateSSA = spvtest::ValidateBase<std::pair<std::string, bool>>;

TEST_F(ValidateSSA, Default) {
  char str[] = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
     OpEntryPoint GLCompute %3 ""
     OpExecutionMode %3 LocalSize 1 1 1
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, IdUndefinedBad) {
  char str[] = R"(
          OpCapability Shader
          OpCapability Linkage
          OpMemoryModel Logical GLSL450
          OpName %missing "missing"
%voidt  = OpTypeVoid
%vfunct = OpTypeFunction %voidt
%func   = OpFunction %vfunct None %missing
%flabel = OpLabel
          OpReturn
          OpFunctionEnd
    )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

TEST_F(ValidateSSA, IdRedefinedBad) {
  char str[] = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
     OpName %2 "redefined"
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%2 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(ValidateSSA, DominateUsageBad) {
  char str[] = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
     OpName %1 "not_dominant"
%2 = OpTypeFunction %1              ; uses %1 before it's definition
%1 = OpTypeVoid
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("not_dominant"));
}

TEST_F(ValidateSSA, DominateUsageWithinBlockBad) {
  char str[] = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
     OpName %bad "bad"
%voidt = OpTypeVoid
%funct = OpTypeFunction %voidt
%uintt = OpTypeInt 32 0
%one   = OpConstant %uintt 1
%func  = OpFunction %voidt None %funct
%entry = OpLabel
%sum   = OpIAdd %uintt %one %bad
%bad   = OpCopyObject %uintt %sum
         OpReturn
         OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("ID .\\[%bad\\] has not been defined\n"
                           "  %8 = OpIAdd %uint %uint_1 %bad\n"));
}

TEST_F(ValidateSSA, DominateUsageSameInstructionBad) {
  char str[] = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
     OpName %sum "sum"
%voidt = OpTypeVoid
%funct = OpTypeFunction %voidt
%uintt = OpTypeInt 32 0
%one   = OpConstant %uintt 1
%func  = OpFunction %voidt None %funct
%entry = OpLabel
%sum   = OpIAdd %uintt %one %sum
         OpReturn
         OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("ID .\\[%sum\\] has not been defined\n"
                           "  %sum = OpIAdd %uint %uint_1 %sum\n"));
}

TEST_F(ValidateSSA, ForwardNameGood) {
  char str[] = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
     OpName %3 "main"
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, ForwardNameMissingTargetBad) {
  char str[] = R"(
      OpCapability Shader
      OpCapability Linkage
      OpMemoryModel Logical GLSL450
      OpName %5 "main"              ; Target never defined
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("main"));
}

TEST_F(ValidateSSA, ForwardMemberNameGood) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
           OpMemberName %struct 0 "value"
           OpMemberName %struct 1 "size"
%intt   =  OpTypeInt 32 1
%uintt  =  OpTypeInt 32 0
%struct =  OpTypeStruct %intt %uintt
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, ForwardMemberNameMissingTargetBad) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
           OpMemberName %struct 0 "value"
           OpMemberName %bad 1 "size"     ; Target is not defined
%intt   =  OpTypeInt 32 1
%uintt  =  OpTypeInt 32 0
%struct =  OpTypeStruct %intt %uintt
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The following forward referenced IDs have not been "
                        "defined:\n2[%2]"));
}

TEST_F(ValidateSSA, ForwardDecorateGood) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
           OpDecorate %var Restrict
%intt   =  OpTypeInt 32 1
%ptrt   =  OpTypePointer UniformConstant %intt
%var    =  OpVariable %ptrt UniformConstant
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, ForwardDecorateInvalidIDBad) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
           OpName %missing "missing"
           OpDecorate %missing Restrict        ;Missing ID
%voidt  =  OpTypeVoid
%intt   =  OpTypeInt 32 1
%ptrt   =  OpTypePointer UniformConstant %intt
%var    =  OpVariable %ptrt UniformConstant
%2      =  OpTypeFunction %voidt
%3      =  OpFunction %voidt None %2
%4      =  OpLabel
           OpReturn
           OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

TEST_F(ValidateSSA, ForwardMemberDecorateGood) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
           OpMemberDecorate %struct 1 RowMajor
%intt   =  OpTypeInt 32 1
%f32    =  OpTypeFloat 32
%vec3   =  OpTypeVector %f32 3
%mat33  =  OpTypeMatrix %vec3 3
%struct =  OpTypeStruct %intt %mat33
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, ForwardMemberDecorateInvalidIdBad) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
           OpName %missing "missing"
           OpMemberDecorate %missing 1 RowMajor ; Target not defined
%intt   =  OpTypeInt 32 1
%f32    =  OpTypeFloat 32
%vec3   =  OpTypeVector %f32 3
%mat33  =  OpTypeMatrix %vec3 3
%struct =  OpTypeStruct %intt %mat33
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

TEST_F(ValidateSSA, ForwardGroupDecorateGood) {
  char str[] = R"(
          OpCapability Shader
          OpCapability Linkage
          OpMemoryModel Logical GLSL450
          OpDecorate %dgrp RowMajor
%dgrp   = OpDecorationGroup
          OpGroupDecorate %dgrp %mat33 %mat44
%f32    =  OpTypeFloat 32
%vec3   = OpTypeVector %f32 3
%vec4   = OpTypeVector %f32 4
%mat33  = OpTypeMatrix %vec3 3
%mat44  = OpTypeMatrix %vec4 4
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, ForwardGroupDecorateMissingGroupBad) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
           OpName %missing "missing"
           OpDecorate %dgrp RowMajor
%dgrp   =  OpDecorationGroup
           OpGroupDecorate %missing %mat33 %mat44 ; Target not defined
%intt   =  OpTypeInt 32 1
%vec3   =  OpTypeVector %intt 3
%vec4   =  OpTypeVector %intt 4
%mat33  =  OpTypeMatrix %vec3 3
%mat44  =  OpTypeMatrix %vec4 4
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

TEST_F(ValidateSSA, ForwardGroupDecorateMissingTargetBad) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
           OpName %missing "missing"
           OpDecorate %dgrp RowMajor
%dgrp   =  OpDecorationGroup
           OpGroupDecorate %dgrp %missing %mat44 ; Target not defined
%f32    =  OpTypeFloat 32
%vec3   =  OpTypeVector %f32 3
%vec4   =  OpTypeVector %f32 4
%mat33  =  OpTypeMatrix %vec3 3
%mat44  =  OpTypeMatrix %vec4 4
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

TEST_F(ValidateSSA, ForwardGroupDecorateDecorationGroupDominateBad) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
           OpName %dgrp "group"
           OpDecorate %dgrp RowMajor
           OpGroupDecorate %dgrp %mat33 %mat44 ; Decoration group does not dominate usage
%dgrp   =  OpDecorationGroup
%intt   =  OpTypeInt 32 1
%vec3   =  OpTypeVector %intt 3
%vec4   =  OpTypeVector %intt 4
%mat33  =  OpTypeMatrix %vec3 3
%mat44  =  OpTypeMatrix %vec4 4
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("group"));
}

TEST_F(ValidateSSA, ForwardDecorateInvalidIdBad) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
           OpName %missing "missing"
           OpDecorate %missing Restrict        ; Missing target
%voidt  =  OpTypeVoid
%intt   =  OpTypeInt 32 1
%ptrt   =  OpTypePointer UniformConstant %intt
%var    =  OpVariable %ptrt UniformConstant
%2      =  OpTypeFunction %voidt
%3      =  OpFunction %voidt None %2
%4      =  OpLabel
           OpReturn
           OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

TEST_F(ValidateSSA, FunctionCallGood) {
  char str[] = R"(
         OpCapability Shader
         OpCapability Linkage
         OpMemoryModel Logical GLSL450
%1    =  OpTypeVoid
%2    =  OpTypeInt 32 1
%3    =  OpTypeInt 32 0
%4    =  OpTypeFunction %1
%8    =  OpTypeFunction %1 %2 %3
%four =  OpConstant %2 4
%five =  OpConstant %3 5
%9    =  OpFunction %1 None %8
%10   =  OpFunctionParameter %2
%11   =  OpFunctionParameter %3
%12   =  OpLabel
         OpReturn
         OpFunctionEnd
%5    =  OpFunction %1 None %4
%6    =  OpLabel
%7    =  OpFunctionCall %1 %9 %four %five
         OpReturn
         OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, ForwardFunctionCallGood) {
  char str[] = R"(
         OpCapability Shader
         OpCapability Linkage
         OpMemoryModel Logical GLSL450
%1    =  OpTypeVoid
%2    =  OpTypeInt 32 1
%3    =  OpTypeInt 32 0
%four =  OpConstant %2 4
%five =  OpConstant %3 5
%8    =  OpTypeFunction %1 %2 %3
%4    =  OpTypeFunction %1
%5    =  OpFunction %1 None %4
%6    =  OpLabel
%7    =  OpFunctionCall %1 %9 %four %five
         OpReturn
         OpFunctionEnd
%9    =  OpFunction %1 None %8
%10   =  OpFunctionParameter %2
%11   =  OpFunctionParameter %3
%12   =  OpLabel
         OpReturn
         OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, ForwardBranchConditionalGood) {
  char str[] = R"(
            OpCapability Shader
            OpCapability Linkage
            OpMemoryModel Logical GLSL450
%voidt  =   OpTypeVoid
%boolt  =   OpTypeBool
%vfunct =   OpTypeFunction %voidt
%true   =   OpConstantTrue %boolt
%main   =   OpFunction %voidt None %vfunct
%mainl  =   OpLabel
            OpSelectionMerge %endl None
            OpBranchConditional %true %truel %falsel
%truel  =   OpLabel
            OpNop
            OpBranch %endl
%falsel =   OpLabel
            OpNop
            OpBranch %endl
%endl    =  OpLabel
            OpReturn
            OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, ForwardBranchConditionalWithWeightsGood) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
%voidt  =  OpTypeVoid
%boolt  =  OpTypeBool
%vfunct =  OpTypeFunction %voidt
%true   =  OpConstantTrue %boolt
%main   =  OpFunction %voidt None %vfunct
%mainl  =  OpLabel
           OpSelectionMerge %endl None
           OpBranchConditional %true %truel %falsel 1 9
%truel  =  OpLabel
           OpNop
           OpBranch %endl
%falsel =  OpLabel
           OpNop
           OpBranch %endl
%endl   =  OpLabel
           OpReturn
           OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, ForwardBranchConditionalNonDominantConditionBad) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
           OpName %tcpy "conditional"
%voidt  =  OpTypeVoid
%boolt  =  OpTypeBool
%vfunct =  OpTypeFunction %voidt
%true   =  OpConstantTrue %boolt
%main   =  OpFunction %voidt None %vfunct
%mainl  =  OpLabel
           OpSelectionMerge %endl None
           OpBranchConditional %tcpy %truel %falsel ;
%truel  =  OpLabel
           OpNop
           OpBranch %endl
%falsel =  OpLabel
           OpNop
           OpBranch %endl
%endl   =  OpLabel
%tcpy   =  OpCopyObject %boolt %true
           OpReturn
           OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("conditional"));
}

TEST_F(ValidateSSA, ForwardBranchConditionalMissingTargetBad) {
  char str[] = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
           OpName %missing "missing"
%voidt  =  OpTypeVoid
%boolt  =  OpTypeBool
%vfunct =  OpTypeFunction %voidt
%true   =  OpConstantTrue %boolt
%main   =  OpFunction %voidt None %vfunct
%mainl  =  OpLabel
           OpSelectionMerge %endl None
           OpBranchConditional %true %missing %falsel
%truel  =  OpLabel
           OpNop
           OpBranch %endl
%falsel =  OpLabel
           OpNop
           OpBranch %endl
%endl   =  OpLabel
           OpReturn
           OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

// Since Int8 requires the Kernel capability, the signedness of int types may
// not be "1".
const std::string kHeader = R"(
OpCapability Int8
OpCapability DeviceEnqueue
OpCapability Linkage
OpMemoryModel Logical OpenCL
)";

const std::string kBasicTypes = R"(
%voidt  =  OpTypeVoid
%boolt  =  OpTypeBool
%int8t  =  OpTypeInt 8 0
%uintt  =  OpTypeInt 32 0
%vfunct =  OpTypeFunction %voidt
%intptrt = OpTypePointer UniformConstant %uintt
%zero      = OpConstant %uintt 0
%one       = OpConstant %uintt 1
%ten       = OpConstant %uintt 10
%false     = OpConstantFalse %boolt
)";

const std::string kKernelTypesAndConstants = R"(
%queuet  = OpTypeQueue

%three   = OpConstant %uintt 3
%arr3t   = OpTypeArray %uintt %three
%ndt     = OpTypeStruct %uintt %arr3t %arr3t %arr3t

%eventt  = OpTypeEvent

%offset = OpConstant %uintt 0
%local  = OpConstant %uintt 1
%gl     = OpConstant %uintt 1

%nevent = OpConstant %uintt 0
%event  = OpConstantNull %eventt

%firstp = OpConstant %int8t 0
%psize  = OpConstant %uintt 0
%palign = OpConstant %uintt 32
%lsize  = OpConstant %uintt 1
%flags  = OpConstant %uintt 0 ; NoWait

%kfunct = OpTypeFunction %voidt %intptrt
)";

const std::string kKernelSetup = R"(
%dqueue = OpGetDefaultQueue %queuet
%ndval  = OpBuildNDRange %ndt %gl %local %offset
%revent = OpUndef %eventt

)";

const std::string kKernelDefinition = R"(
%kfunc  = OpFunction %voidt None %kfunct
%iparam = OpFunctionParameter %intptrt
%kfuncl = OpLabel
          OpNop
          OpReturn
          OpFunctionEnd
)";

TEST_F(ValidateSSA, EnqueueKernelGood) {
  std::string str = kHeader + kBasicTypes + kKernelTypesAndConstants +
                    kKernelDefinition + R"(
                %main   = OpFunction %voidt None %vfunct
                %mainl  = OpLabel
                )" + kKernelSetup + R"(
                %err    = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp %psize
                                        %palign %lsize
                          OpReturn
                          OpFunctionEnd
                 )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, ForwardEnqueueKernelGood) {
  std::string str = kHeader + kBasicTypes + kKernelTypesAndConstants + R"(
                %main   = OpFunction %voidt None %vfunct
                %mainl  = OpLabel
                )" +
                    kKernelSetup + R"(
                %err    = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp %psize
                                        %palign %lsize
                         OpReturn
                         OpFunctionEnd
                 )" + kKernelDefinition;
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, EnqueueMissingFunctionBad) {
  std::string str = kHeader + "OpName %kfunc \"kfunc\"" + kBasicTypes +
                    kKernelTypesAndConstants + R"(
                %main   = OpFunction %voidt None %vfunct
                %mainl  = OpLabel
                )" + kKernelSetup + R"(
                %err    = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp %psize
                                        %palign %lsize
                         OpReturn
                         OpFunctionEnd
                 )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("kfunc"));
}

std::string forwardKernelNonDominantParameterBaseCode(
    std::string name = std::string()) {
  std::string op_name;
  if (name.empty()) {
    op_name = "";
  } else {
    op_name = "\nOpName %" + name + " \"" + name + "\"\n";
  }
  std::string out = kHeader + op_name + kBasicTypes + kKernelTypesAndConstants +
                    kKernelDefinition +
                    R"(
                %main   = OpFunction %voidt None %vfunct
                %mainl  = OpLabel
                )" + kKernelSetup;
  return out;
}

TEST_F(ValidateSSA, ForwardEnqueueKernelMissingParameter1Bad) {
  std::string str = forwardKernelNonDominantParameterBaseCode("missing") + R"(
                %err    = OpEnqueueKernel %missing %dqueue %flags %ndval
                                        %nevent %event %revent %kfunc %firstp
                                        %psize %palign %lsize
                          OpReturn
                          OpFunctionEnd
                )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

TEST_F(ValidateSSA, ForwardEnqueueKernelNonDominantParameter2Bad) {
  std::string str = forwardKernelNonDominantParameterBaseCode("dqueue2") + R"(
                %err     = OpEnqueueKernel %uintt %dqueue2 %flags %ndval
                                            %nevent %event %revent %kfunc
                                            %firstp %psize %palign %lsize
                %dqueue2 = OpGetDefaultQueue %queuet
                           OpReturn
                           OpFunctionEnd
                )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("dqueue2"));
}

TEST_F(ValidateSSA, ForwardEnqueueKernelNonDominantParameter3Bad) {
  std::string str = forwardKernelNonDominantParameterBaseCode("ndval2") + R"(
                %err    = OpEnqueueKernel %uintt %dqueue %flags %ndval2
                                        %nevent %event %revent %kfunc %firstp
                                        %psize %palign %lsize
                %ndval2  = OpBuildNDRange %ndt %gl %local %offset
                          OpReturn
                          OpFunctionEnd
                )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("ndval2"));
}

TEST_F(ValidateSSA, ForwardEnqueueKernelNonDominantParameter4Bad) {
  std::string str = forwardKernelNonDominantParameterBaseCode("nevent2") + R"(
              %err    = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent2
                                        %event %revent %kfunc %firstp %psize
                                        %palign %lsize
              %nevent2 = OpCopyObject %uintt %nevent
                        OpReturn
                        OpFunctionEnd
              )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("nevent2"));
}

TEST_F(ValidateSSA, ForwardEnqueueKernelNonDominantParameter5Bad) {
  std::string str = forwardKernelNonDominantParameterBaseCode("event2") + R"(
              %err     = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event2 %revent %kfunc %firstp %psize
                                        %palign %lsize
              %event2  = OpCopyObject %eventt %event
                         OpReturn
                         OpFunctionEnd
              )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("event2"));
}

TEST_F(ValidateSSA, ForwardEnqueueKernelNonDominantParameter6Bad) {
  std::string str = forwardKernelNonDominantParameterBaseCode("revent2") + R"(
              %err     = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent2 %kfunc %firstp %psize
                                        %palign %lsize
              %revent2 = OpCopyObject %eventt %revent
                         OpReturn
                         OpFunctionEnd
              )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("revent2"));
}

TEST_F(ValidateSSA, ForwardEnqueueKernelNonDominantParameter8Bad) {
  std::string str = forwardKernelNonDominantParameterBaseCode("firstp2") + R"(
              %err     = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp2 %psize
                                        %palign %lsize
              %firstp2 = OpCopyObject %int8t %firstp
                         OpReturn
                         OpFunctionEnd
              )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("firstp2"));
}

TEST_F(ValidateSSA, ForwardEnqueueKernelNonDominantParameter9Bad) {
  std::string str = forwardKernelNonDominantParameterBaseCode("psize2") + R"(
              %err    = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp %psize2
                                        %palign %lsize
              %psize2 = OpCopyObject %uintt %psize
                        OpReturn
                        OpFunctionEnd
              )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("psize2"));
}

TEST_F(ValidateSSA, ForwardEnqueueKernelNonDominantParameter10Bad) {
  std::string str = forwardKernelNonDominantParameterBaseCode("palign2") + R"(
              %err     = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp %psize
                                        %palign2 %lsize
              %palign2 = OpCopyObject %uintt %palign
                        OpReturn
                        OpFunctionEnd
              )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("palign2"));
}

TEST_F(ValidateSSA, ForwardEnqueueKernelNonDominantParameter11Bad) {
  std::string str = forwardKernelNonDominantParameterBaseCode("lsize2") + R"(
              %err     = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp %psize
                                        %palign %lsize2
              %lsize2  = OpCopyObject %uintt %lsize
                         OpReturn
                         OpFunctionEnd
              )";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("lsize2"));
}

static const bool kWithNDrange = true;
static const bool kNoNDrange = false;
std::pair<std::string, bool> cases[] = {
    {"OpGetKernelNDrangeSubGroupCount", kWithNDrange},
    {"OpGetKernelNDrangeMaxSubGroupSize", kWithNDrange},
    {"OpGetKernelWorkGroupSize", kNoNDrange},
    {"OpGetKernelPreferredWorkGroupSizeMultiple", kNoNDrange}};

INSTANTIATE_TEST_SUITE_P(KernelArgs, ValidateSSA, ::testing::ValuesIn(cases));

static const std::string return_instructions = R"(
  OpReturn
  OpFunctionEnd
)";

TEST_P(ValidateSSA, GetKernelGood) {
  std::string instruction = GetParam().first;
  bool with_ndrange = GetParam().second;
  std::string ndrange_param = with_ndrange ? " %ndval " : " ";

  std::stringstream ss;
  // clang-format off
  ss << forwardKernelNonDominantParameterBaseCode() + " %numsg = "
     << instruction + " %uintt" + ndrange_param + "%kfunc %firstp %psize %palign"
     << return_instructions;
  // clang-format on

  CompileSuccessfully(ss.str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateSSA, ForwardGetKernelGood) {
  std::string instruction = GetParam().first;
  bool with_ndrange = GetParam().second;
  std::string ndrange_param = with_ndrange ? " %ndval " : " ";

  // clang-format off
  std::string str = kHeader + kBasicTypes + kKernelTypesAndConstants +
               R"(
            %main    = OpFunction %voidt None %vfunct
            %mainl   = OpLabel
                )"
            + kKernelSetup + " %numsg = "
            + instruction + " %uintt" + ndrange_param + "%kfunc %firstp %psize %palign"
            + return_instructions + kKernelDefinition;
  // clang-format on

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateSSA, ForwardGetKernelMissingDefinitionBad) {
  std::string instruction = GetParam().first;
  bool with_ndrange = GetParam().second;
  std::string ndrange_param = with_ndrange ? " %ndval " : " ";

  std::stringstream ss;
  // clang-format off
  ss << forwardKernelNonDominantParameterBaseCode("missing") + " %numsg = "
     << instruction + " %uintt" + ndrange_param + "%missing %firstp %psize %palign"
     << return_instructions;
  // clang-format on

  CompileSuccessfully(ss.str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

TEST_P(ValidateSSA, ForwardGetKernelNDrangeSubGroupCountMissingParameter1Bad) {
  std::string instruction = GetParam().first;
  bool with_ndrange = GetParam().second;
  std::string ndrange_param = with_ndrange ? " %ndval " : " ";

  std::stringstream ss;
  // clang-format off
  ss << forwardKernelNonDominantParameterBaseCode("missing") + " %numsg = "
     << instruction + " %missing" + ndrange_param + "%kfunc %firstp %psize %palign"
     << return_instructions;
  // clang-format on

  CompileSuccessfully(ss.str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

TEST_P(ValidateSSA,
       ForwardGetKernelNDrangeSubGroupCountNonDominantParameter2Bad) {
  std::string instruction = GetParam().first;
  bool with_ndrange = GetParam().second;
  std::string ndrange_param = with_ndrange ? " %ndval2 " : " ";

  std::stringstream ss;
  // clang-format off
  ss << forwardKernelNonDominantParameterBaseCode("ndval2") + " %numsg = "
     << instruction + " %uintt" + ndrange_param + "%kfunc %firstp %psize %palign"
     << "\n %ndval2  = OpBuildNDRange %ndt %gl %local %offset"
     << return_instructions;
  // clang-format on

  if (GetParam().second) {
    CompileSuccessfully(ss.str());
    ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(), HasSubstr("ndval2"));
  }
}

TEST_P(ValidateSSA,
       ForwardGetKernelNDrangeSubGroupCountNonDominantParameter4Bad) {
  std::string instruction = GetParam().first;
  bool with_ndrange = GetParam().second;
  std::string ndrange_param = with_ndrange ? " %ndval " : " ";

  std::stringstream ss;
  // clang-format off
  ss << forwardKernelNonDominantParameterBaseCode("firstp2") + " %numsg = "
     << instruction + " %uintt" + ndrange_param + "%kfunc %firstp2 %psize %palign"
     << "\n %firstp2 = OpCopyObject %int8t %firstp"
     << return_instructions;
  // clang-format on

  CompileSuccessfully(ss.str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("firstp2"));
}

TEST_P(ValidateSSA,
       ForwardGetKernelNDrangeSubGroupCountNonDominantParameter5Bad) {
  std::string instruction = GetParam().first;
  bool with_ndrange = GetParam().second;
  std::string ndrange_param = with_ndrange ? " %ndval " : " ";

  std::stringstream ss;
  // clang-format off
  ss << forwardKernelNonDominantParameterBaseCode("psize2") + " %numsg = "
     << instruction + " %uintt" + ndrange_param + "%kfunc %firstp %psize2 %palign"
     << "\n %psize2  = OpCopyObject %uintt %psize"
     << return_instructions;
  // clang-format on

  CompileSuccessfully(ss.str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("psize2"));
}

TEST_P(ValidateSSA,
       ForwardGetKernelNDrangeSubGroupCountNonDominantParameter6Bad) {
  std::string instruction = GetParam().first;
  bool with_ndrange = GetParam().second;
  std::string ndrange_param = with_ndrange ? " %ndval " : " ";

  std::stringstream ss;
  // clang-format off
  ss << forwardKernelNonDominantParameterBaseCode("palign2") + " %numsg = "
     << instruction + " %uintt" + ndrange_param + "%kfunc %firstp %psize %palign2"
     << "\n %palign2 = OpCopyObject %uintt %palign"
     << return_instructions;
  // clang-format on

  if (GetParam().second) {
    CompileSuccessfully(ss.str());
    ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(), HasSubstr("palign2"));
  }
}

TEST_F(ValidateSSA, PhiGood) {
  std::string str = kHeader + kBasicTypes +
                    R"(
%func      = OpFunction %voidt None %vfunct
%preheader = OpLabel
%init      = OpCopyObject %uintt %zero
             OpBranch %loop
%loop      = OpLabel
%i         = OpPhi %uintt %init %preheader %loopi %loop
%loopi     = OpIAdd %uintt %i %one
             OpNop
%cond      = OpSLessThan %boolt %i %ten
             OpLoopMerge %endl %loop None
             OpBranchConditional %cond %loop %endl
%endl      = OpLabel
             OpReturn
             OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, PhiMissingTypeBad) {
  std::string str = kHeader + "OpName %missing \"missing\"" + kBasicTypes +
                    R"(
%func      = OpFunction %voidt None %vfunct
%preheader = OpLabel
%init      = OpCopyObject %uintt %zero
             OpBranch %loop
%loop      = OpLabel
%i         = OpPhi %missing %init %preheader %loopi %loop
%loopi     = OpIAdd %uintt %i %one
             OpNop
%cond      = OpSLessThan %boolt %i %ten
             OpLoopMerge %endl %loop None
             OpBranchConditional %cond %loop %endl
%endl      = OpLabel
             OpReturn
             OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

TEST_F(ValidateSSA, PhiMissingIdBad) {
  std::string str = kHeader + "OpName %missing \"missing\"" + kBasicTypes +
                    R"(
%func      = OpFunction %voidt None %vfunct
%preheader = OpLabel
%init      = OpCopyObject %uintt %zero
             OpBranch %loop
%loop      = OpLabel
%i         = OpPhi %uintt %missing %preheader %loopi %loop
%loopi     = OpIAdd %uintt %i %one
             OpNop
%cond      = OpSLessThan %boolt %i %ten
             OpLoopMerge %endl %loop None
             OpBranchConditional %cond %loop %endl
%endl      = OpLabel
             OpReturn
             OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

TEST_F(ValidateSSA, PhiMissingLabelBad) {
  std::string str = kHeader + "OpName %missing \"missing\"" + kBasicTypes +
                    R"(
%func      = OpFunction %voidt None %vfunct
%preheader = OpLabel
%init      = OpCopyObject %uintt %zero
             OpBranch %loop
%loop      = OpLabel
%i         = OpPhi %uintt %init %missing %loopi %loop
%loopi     = OpIAdd %uintt %i %one
             OpNop
%cond      = OpSLessThan %boolt %i %ten
             OpLoopMerge %endl %loop None
             OpBranchConditional %cond %loop %endl
%endl      = OpLabel
             OpReturn
             OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("missing"));
}

TEST_F(ValidateSSA, IdDominatesItsUseGood) {
  std::string str = kHeader + kBasicTypes +
                    R"(
%func      = OpFunction %voidt None %vfunct
%entry     = OpLabel
%cond      = OpSLessThan %boolt %one %ten
%eleven    = OpIAdd %uintt %one %ten
             OpSelectionMerge %merge None
             OpBranchConditional %cond %t %f
%t         = OpLabel
%twelve    = OpIAdd %uintt %eleven %one
             OpBranch %merge
%f         = OpLabel
%twentytwo = OpIAdd %uintt %eleven %ten
             OpBranch %merge
%merge     = OpLabel
             OpReturn
             OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, IdDoesNotDominateItsUseBad) {
  std::string str = kHeader +
                    "OpName %eleven \"eleven\"\n"
                    "OpName %true_block \"true_block\"\n"
                    "OpName %false_block \"false_block\"" +
                    kBasicTypes +
                    R"(
%func        = OpFunction %voidt None %vfunct
%entry       = OpLabel
%cond        = OpSLessThan %boolt %one %ten
               OpSelectionMerge %merge None
               OpBranchConditional %cond %true_block %false_block
%true_block  = OpLabel
%eleven      = OpIAdd %uintt %one %ten
%twelve      = OpIAdd %uintt %eleven %one
               OpBranch %merge
%false_block = OpLabel
%twentytwo   = OpIAdd %uintt %eleven %ten
               OpBranch %merge
%merge       = OpLabel
               OpReturn
               OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      MatchesRegex("ID .\\[%eleven\\] defined in block .\\[%true_block\\] "
                   "does not dominate its use in block .\\[%false_block\\]\n"
                   "  %false_block = OpLabel\n"));
}

TEST_F(ValidateSSA, PhiUseDoesntDominateDefinitionGood) {
  std::string str = kHeader + kBasicTypes +
                    R"(
%funcintptrt = OpTypePointer Function %uintt
%func        = OpFunction %voidt None %vfunct
%entry       = OpLabel
%var_one     = OpVariable %funcintptrt Function %one
%one_val     = OpLoad %uintt %var_one
               OpBranch %loop
%loop        = OpLabel
%i           = OpPhi %uintt %one_val %entry %inew %cont
%cond        = OpSLessThan %boolt %one %ten
               OpLoopMerge %merge %cont None
               OpBranchConditional %cond %body %merge
%body        = OpLabel
               OpBranch %cont
%cont        = OpLabel
%inew        = OpIAdd %uintt %i %one
               OpBranch %loop
%merge       = OpLabel
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA,
       PhiUseDoesntDominateUseOfPhiOperandUsedBeforeDefinitionBad) {
  std::string str = kHeader + "OpName %inew \"inew\"" + kBasicTypes +
                    R"(
%func        = OpFunction %voidt None %vfunct
%entry       = OpLabel
%var_one     = OpVariable %intptrt Function %one
%one_val     = OpLoad %uintt %var_one
               OpBranch %loop
%loop        = OpLabel
%i           = OpPhi %uintt %one_val %entry %inew %cont
%bad         = OpIAdd %uintt %inew %one
%cond        = OpSLessThan %boolt %one %ten
               OpLoopMerge %merge %cont None
               OpBranchConditional %cond %body %merge
%body        = OpLabel
               OpBranch %cont
%cont        = OpLabel
%inew        = OpIAdd %uintt %i %one
               OpBranch %loop
%merge       = OpLabel
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("ID .\\[%inew\\] has not been defined\n"
                           "  %19 = OpIAdd %uint %inew %uint_1\n"));
}

TEST_F(ValidateSSA, PhiUseMayComeFromNonDominatingBlockGood) {
  std::string str = kHeader + "OpName %if_true \"if_true\"\n" +
                    "OpName %exit \"exit\"\n" + "OpName %copy \"copy\"\n" +
                    kBasicTypes +
                    R"(
%func        = OpFunction %voidt None %vfunct
%entry       = OpLabel
               OpBranchConditional %false %if_true %exit

%if_true     = OpLabel
%copy        = OpCopyObject %boolt %false
               OpBranch %exit

; The use of %copy here is ok, even though it was defined
; in a block that does not dominate %exit.  That's the point
; of an OpPhi.
%exit        = OpLabel
%value       = OpPhi %boolt %false %entry %copy %if_true
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions()) << getDiagnosticString();
}

TEST_F(ValidateSSA, PhiUsesItsOwnDefinitionGood) {
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/415
  //
  // Non-phi instructions can't use their own definitions, as
  // already checked in test DominateUsageSameInstructionBad.
  std::string str = kHeader + "OpName %loop \"loop\"\n" +
                    "OpName %value \"value\"\n" + kBasicTypes +
                    R"(
%func        = OpFunction %voidt None %vfunct
%entry       = OpLabel
               OpBranch %loop

%loop        = OpLabel
%value       = OpPhi %boolt %false %entry %value %loop
               OpBranch %loop

               OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions()) << getDiagnosticString();
}

TEST_F(ValidateSSA, PhiVariableDefNotDominatedByParentBlockBad) {
  std::string str = kHeader + "OpName %if_true \"if_true\"\n" +
                    "OpName %if_false \"if_false\"\n" +
                    "OpName %exit \"exit\"\n" + "OpName %value \"phi\"\n" +
                    "OpName %true_copy \"true_copy\"\n" +
                    "OpName %false_copy \"false_copy\"\n" + kBasicTypes +
                    R"(
%func        = OpFunction %voidt None %vfunct
%entry       = OpLabel
               OpBranchConditional %false %if_true %if_false

%if_true     = OpLabel
%true_copy   = OpCopyObject %boolt %false
               OpBranch %exit

%if_false    = OpLabel
%false_copy  = OpCopyObject %boolt %false
               OpBranch %exit

; The (variable,Id) pairs are swapped.
%exit        = OpLabel
%value       = OpPhi %boolt %true_copy %if_false %false_copy %if_true
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      MatchesRegex("In OpPhi instruction .\\[%phi\\], ID .\\[%true_copy\\] "
                   "definition does not dominate its parent .\\[%if_false\\]\n"
                   "  %phi = OpPhi %bool %true_copy %if_false %false_copy "
                   "%if_true\n"));
}

TEST_F(ValidateSSA, PhiVariableDefDominatesButNotDefinedInParentBlock) {
  std::string str = kHeader + "OpName %if_true \"if_true\"\n" + kBasicTypes +
                    R"(
%func        = OpFunction %voidt None %vfunct
%entry       = OpLabel
               OpBranchConditional %false %if_true %if_false

%if_true     = OpLabel
%true_copy   = OpCopyObject %boolt %false
               OpBranch %if_tnext
%if_tnext    = OpLabel
               OpBranch %exit

%if_false    = OpLabel
%false_copy  = OpCopyObject %boolt %false
               OpBranch %if_fnext
%if_fnext    = OpLabel
               OpBranch %exit

%exit        = OpLabel
%value       = OpPhi %boolt %true_copy %if_tnext %false_copy %if_fnext
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA,
       DominanceCheckIgnoresUsesInUnreachableBlocksDefInBlockGood) {
  std::string str = kHeader + kBasicTypes +
                    R"(
%func        = OpFunction %voidt None %vfunct
%entry       = OpLabel
%def         = OpCopyObject %boolt %false
               OpReturn

%unreach     = OpLabel
%use         = OpCopyObject %boolt %def
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions()) << getDiagnosticString();
}

TEST_F(ValidateSSA, PhiVariableUnreachableDefNotInParentBlock) {
  std::string str = kHeader + "OpName %unreachable \"unreachable\"\n" +
                    kBasicTypes +
                    R"(
%func        = OpFunction %voidt None %vfunct
%entry       = OpLabel
               OpBranch %if_false

%unreachable = OpLabel
%copy        = OpCopyObject %boolt %false
               OpBranch %if_tnext
%if_tnext    = OpLabel
               OpBranch %exit

%if_false    = OpLabel
%false_copy  = OpCopyObject %boolt %false
               OpBranch %if_fnext
%if_fnext    = OpLabel
               OpBranch %exit

%exit        = OpLabel
%value       = OpPhi %boolt %copy %if_tnext %false_copy %if_fnext
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA,
       DominanceCheckIgnoresUsesInUnreachableBlocksDefIsParamGood) {
  std::string str = kHeader + kBasicTypes +
                    R"(
%void_fn_int = OpTypeFunction %voidt %uintt
%func        = OpFunction %voidt None %void_fn_int
%int_param   = OpFunctionParameter %uintt
%entry       = OpLabel
               OpReturn

%unreach     = OpLabel
%use         = OpCopyObject %uintt %int_param
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions()) << getDiagnosticString();
}

TEST_F(ValidateSSA, UseFunctionParameterFromOtherFunctionBad) {
  std::string str = kHeader +
                    "OpName %first \"first\"\n"
                    "OpName %func \"func\"\n" +
                    "OpName %func2 \"func2\"\n" + kBasicTypes +
                    R"(
%viifunct  = OpTypeFunction %voidt %uintt %uintt
%func      = OpFunction %voidt None %viifunct
%first     = OpFunctionParameter %uintt
%second    = OpFunctionParameter %uintt
             OpFunctionEnd
%func2     = OpFunction %voidt None %viifunct
%first2    = OpFunctionParameter %uintt
%second2   = OpFunctionParameter %uintt
%entry2    = OpLabel
%baduse    = OpIAdd %uintt %first %first2
             OpReturn
             OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      MatchesRegex("ID .\\[%first\\] used in function .\\[%func2\\] is used "
                   "outside of it's defining function .\\[%func\\]\n"
                   "  %func = OpFunction %void None %14\n"));
}

TEST_F(ValidateSSA, TypeForwardPointerForwardReference) {
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/429
  //
  // ForwardPointers can references instructions that have not been defined
  std::string str = R"(
               OpCapability Kernel
               OpCapability Addresses
               OpCapability Linkage
               OpMemoryModel Logical OpenCL
               OpName %intptrt "intptrt"
               OpTypeForwardPointer %intptrt UniformConstant
       %uint = OpTypeInt 32 0
    %intptrt = OpTypePointer UniformConstant %uint
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSSA, TypeStructForwardReference) {
  std::string str = R"(
               OpCapability Kernel
               OpCapability Addresses
               OpCapability Linkage
               OpMemoryModel Logical OpenCL
               OpName %structptr "structptr"
               OpTypeForwardPointer %structptr UniformConstant
       %uint = OpTypeInt 32 0
   %structt1 = OpTypeStruct %structptr %uint
   %structt2 = OpTypeStruct %uint %structptr
   %structt3 = OpTypeStruct %uint %uint %structptr
   %structt4 = OpTypeStruct %uint %uint %uint %structptr
  %structptr = OpTypePointer UniformConstant %structt1
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// TODO(umar): OpGroupMemberDecorate

}  // namespace
}  // namespace val
}  // namespace spvtools
