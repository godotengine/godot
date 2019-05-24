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

#include <string>

#include "gmock/gmock.h"
#include "source/opt/build_module.h"
#include "source/opt/value_number_table.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::HasSubstr;
using ::testing::MatchesRegex;
using RedundancyEliminationTest = PassTest<::testing::Test>;

// Test that it can get a simple case of local redundancy elimination.
// The rest of the test check for extra functionality.
TEST_F(RedundancyEliminationTest, RemoveRedundantLocalAdd) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Function %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %8 = OpVariable %6 Function
          %9 = OpLoad %5 %8
         %10 = OpFAdd %5 %9 %9
; CHECK: OpFAdd
; CHECK-NOT: OpFAdd
         %11 = OpFAdd %5 %9 %9
               OpReturn
               OpFunctionEnd
  )";
  SinglePassRunAndMatch<RedundancyEliminationPass>(text, false);
}

// Remove a redundant add across basic blocks.
TEST_F(RedundancyEliminationTest, RemoveRedundantAdd) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Function %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %8 = OpVariable %6 Function
          %9 = OpLoad %5 %8
         %10 = OpFAdd %5 %9 %9
               OpBranch %11
         %11 = OpLabel
; CHECK: OpFAdd
; CHECK-NOT: OpFAdd
         %12 = OpFAdd %5 %9 %9
               OpReturn
               OpFunctionEnd
  )";
  SinglePassRunAndMatch<RedundancyEliminationPass>(text, false);
}

// Remove a redundant add going through a multiple basic blocks.
TEST_F(RedundancyEliminationTest, RemoveRedundantAddDiamond) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Function %5
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
         %10 = OpVariable %6 Function
         %11 = OpLoad %5 %10
         %12 = OpFAdd %5 %11 %11
; CHECK: OpFAdd
; CHECK-NOT: OpFAdd
               OpBranchConditional %8 %13 %14
         %13 = OpLabel
               OpBranch %15
         %14 = OpLabel
               OpBranch %15
         %15 = OpLabel
         %16 = OpFAdd %5 %11 %11
               OpReturn
               OpFunctionEnd

  )";
  SinglePassRunAndMatch<RedundancyEliminationPass>(text, false);
}

// Remove a redundant add in a side node.
TEST_F(RedundancyEliminationTest, RemoveRedundantAddInSideNode) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Function %5
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
         %10 = OpVariable %6 Function
         %11 = OpLoad %5 %10
         %12 = OpFAdd %5 %11 %11
; CHECK: OpFAdd
; CHECK-NOT: OpFAdd
               OpBranchConditional %8 %13 %14
         %13 = OpLabel
               OpBranch %15
         %14 = OpLabel
         %16 = OpFAdd %5 %11 %11
               OpBranch %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd

  )";
  SinglePassRunAndMatch<RedundancyEliminationPass>(text, false);
}

// Remove a redundant add whose value is in the result of a phi node.
TEST_F(RedundancyEliminationTest, RemoveRedundantAddWithPhi) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Function %5
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
         %10 = OpVariable %6 Function
         %11 = OpLoad %5 %10
               OpBranchConditional %8 %13 %14
         %13 = OpLabel
         %add1 = OpFAdd %5 %11 %11
; CHECK: OpFAdd
               OpBranch %15
         %14 = OpLabel
         %add2 = OpFAdd %5 %11 %11
; CHECK: OpFAdd
               OpBranch %15
         %15 = OpLabel
; CHECK: OpPhi
          %phi = OpPhi %5 %add1 %13 %add2 %14
; CHECK-NOT: OpFAdd
         %16 = OpFAdd %5 %11 %11
               OpReturn
               OpFunctionEnd

  )";
  SinglePassRunAndMatch<RedundancyEliminationPass>(text, false);
}

// Keep the add because it is redundant on some paths, but not all paths.
TEST_F(RedundancyEliminationTest, KeepPartiallyRedundantAdd) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Function %5
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
         %10 = OpVariable %6 Function
         %11 = OpLoad %5 %10
               OpBranchConditional %8 %13 %14
         %13 = OpLabel
        %add = OpFAdd %5 %11 %11
               OpBranch %15
         %14 = OpLabel
               OpBranch %15
         %15 = OpLabel
         %16 = OpFAdd %5 %11 %11
               OpReturn
               OpFunctionEnd

  )";
  auto result = SinglePassRunAndDisassemble<RedundancyEliminationPass>(
      text, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

// Keep the add.  Even if it is redundant on all paths, there is no single id
// whose definition dominates the add and contains the same value.
TEST_F(RedundancyEliminationTest, KeepRedundantAddWithoutPhi) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Function %5
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
         %10 = OpVariable %6 Function
         %11 = OpLoad %5 %10
               OpBranchConditional %8 %13 %14
         %13 = OpLabel
         %add1 = OpFAdd %5 %11 %11
               OpBranch %15
         %14 = OpLabel
         %add2 = OpFAdd %5 %11 %11
               OpBranch %15
         %15 = OpLabel
         %16 = OpFAdd %5 %11 %11
               OpReturn
               OpFunctionEnd

  )";
  auto result = SinglePassRunAndDisassemble<RedundancyEliminationPass>(
      text, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
