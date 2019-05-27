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
using LocalRedundancyEliminationTest = PassTest<::testing::Test>;

// Remove an instruction when it was already computed.
TEST_F(LocalRedundancyEliminationTest, RemoveRedundantAdd) {
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
  SinglePassRunAndMatch<LocalRedundancyEliminationPass>(text, false);
}

// Make sure we keep instruction that are different, but look similar.
TEST_F(LocalRedundancyEliminationTest, KeepDifferentAdd) {
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
               OpStore %8 %10
         %11 = OpLoad %5 %8
; CHECK: %11 = OpLoad
         %12 = OpFAdd %5 %11 %11
; CHECK: OpFAdd [[:%\w+]] %11 %11
               OpReturn
               OpFunctionEnd
  )";
  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<LocalRedundancyEliminationPass>(text, false);
}

// This test is check that the values are being propagated properly, and that
// we are able to identify sequences of instruction that are not needed.
TEST_F(LocalRedundancyEliminationTest, RemoveMultipleInstructions) {
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
          %6 = OpTypePointer Uniform %5
          %8 = OpVariable %6 Uniform
          %2 = OpFunction %3 None %4
          %7 = OpLabel
; CHECK: [[r1:%\w+]] = OpLoad
          %9 = OpLoad %5 %8
; CHECK-NEXT: [[r2:%\w+]] = OpFAdd [[:%\w+]] [[r1]] [[r1]]
         %10 = OpFAdd %5 %9 %9
; CHECK-NEXT: [[r3:%\w+]] = OpFMul [[:%\w+]] [[r2]] [[r1]]
         %11 = OpFMul %5 %10 %9
; CHECK-NOT: OpLoad
         %12 = OpLoad %5 %8
; CHECK-NOT: OpFAdd [[:\w+]] %12 %12
         %13 = OpFAdd %5 %12 %12
; CHECK-NOT: OpFMul
         %14 = OpFMul %5 %13 %12
; CHECK-NEXT: [[:%\w+]] = OpFAdd [[:%\w+]] [[r3]] [[r3]]
         %15 = OpFAdd %5 %14 %11
               OpReturn
               OpFunctionEnd
  )";
  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<LocalRedundancyEliminationPass>(text, false);
}

// Redundant instructions in different blocks should be kept.
TEST_F(LocalRedundancyEliminationTest, KeepInstructionsInDifferentBlocks) {
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
        %bb1 = OpLabel
          %8 = OpVariable %6 Function
          %9 = OpLoad %5 %8
         %10 = OpFAdd %5 %9 %9
; CHECK: OpFAdd
               OpBranch %bb2
        %bb2 = OpLabel
; CHECK: OpFAdd
         %11 = OpFAdd %5 %9 %9
               OpReturn
               OpFunctionEnd
  )";
  SinglePassRunAndMatch<LocalRedundancyEliminationPass>(text, false);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
