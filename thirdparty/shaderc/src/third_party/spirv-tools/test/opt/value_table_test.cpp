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

namespace spvtools {
namespace opt {
namespace {

using ::testing::HasSubstr;
using ::testing::MatchesRegex;
using ValueTableTest = PassTest<::testing::Test>;

TEST_F(ValueTableTest, SameInstructionSameValue) {
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
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst), vtable.GetValueNumber(inst));
}

TEST_F(ValueTableTest, DifferentInstructionSameValue) {
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
         %11 = OpFAdd %5 %9 %9
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(10);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(11);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, SameValueDifferentBlock) {
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
         %12 = OpFAdd %5 %9 %9
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(10);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(12);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, DifferentValue) {
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
         %11 = OpFAdd %5 %9 %10
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(10);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(11);
  EXPECT_NE(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, DifferentValueDifferentBlock) {
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
         %12 = OpFAdd %5 %9 %10
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(10);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(12);
  EXPECT_NE(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, SameLoad) {
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
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst = context->get_def_use_mgr()->GetDef(9);
  EXPECT_EQ(vtable.GetValueNumber(inst), vtable.GetValueNumber(inst));
}

// Two different loads, even from the same memory, must given different value
// numbers if the memory is not read-only.
TEST_F(ValueTableTest, DifferentFunctionLoad) {
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
          %10 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(9);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(10);
  EXPECT_NE(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, DifferentUniformLoad) {
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
          %9 = OpLoad %5 %8
          %10 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(9);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, DifferentInputLoad) {
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
          %6 = OpTypePointer Input %5
          %8 = OpVariable %6 Input
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %9 = OpLoad %5 %8
          %10 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(9);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, DifferentUniformConstantLoad) {
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
          %6 = OpTypePointer UniformConstant %5
          %8 = OpVariable %6 UniformConstant
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %9 = OpLoad %5 %8
          %10 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(9);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, DifferentPushConstantLoad) {
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
          %6 = OpTypePointer PushConstant %5
          %8 = OpVariable %6 PushConstant
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %9 = OpLoad %5 %8
          %10 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(9);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, SameCall) {
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
          %6 = OpTypeFunction %5
          %7 = OpTypePointer Function %5
          %8 = OpVariable %7 Private
          %2 = OpFunction %3 None %4
          %9 = OpLabel
         %10 = OpFunctionCall %5 %11
               OpReturn
               OpFunctionEnd
         %11 = OpFunction %5 None %6
         %12 = OpLabel
         %13 = OpLoad %5 %8
               OpReturnValue %13
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst), vtable.GetValueNumber(inst));
}

// Function calls should be given a new value number, even if they are the same.
TEST_F(ValueTableTest, DifferentCall) {
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
          %6 = OpTypeFunction %5
          %7 = OpTypePointer Function %5
          %8 = OpVariable %7 Private
          %2 = OpFunction %3 None %4
          %9 = OpLabel
         %10 = OpFunctionCall %5 %11
         %12 = OpFunctionCall %5 %11
               OpReturn
               OpFunctionEnd
         %11 = OpFunction %5 None %6
         %13 = OpLabel
         %14 = OpLoad %5 %8
               OpReturnValue %14
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(10);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(12);
  EXPECT_NE(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

// It is possible to have two instruction that compute the same numerical value,
// but with different types.  They should have different value numbers.
TEST_F(ValueTableTest, DifferentTypes) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 0
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %5
          %2 = OpFunction %3 None %4
          %8 = OpLabel
          %9 = OpVariable %7 Function
         %10 = OpLoad %5 %9
         %11 = OpIAdd %5 %10 %10
         %12 = OpIAdd %6 %10 %10
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(11);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(12);
  EXPECT_NE(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, CopyObject) {
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
         %10 = OpCopyObject %5 %9
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(9);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

// Test that a phi where the operands have the same value assigned that value
// to the result of the phi.
TEST_F(ValueTableTest, PhiTest1) {
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
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %9 = OpVariable %6 Uniform
          %2 = OpFunction %3 None %4
         %10 = OpLabel
               OpBranchConditional %8 %11 %12
         %11 = OpLabel
         %13 = OpLoad %5 %9
               OpBranch %14
         %12 = OpLabel
         %15 = OpLoad %5 %9
               OpBranch %14
         %14 = OpLabel
         %16 = OpPhi %5 %13 %11 %15 %12
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(13);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(15);
  Instruction* phi = context->get_def_use_mgr()->GetDef(16);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(phi));
}

// When the values for the inputs to a phi do not match, then the phi should
// have its own value number.
TEST_F(ValueTableTest, PhiTest2) {
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
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %9 = OpVariable %6 Uniform
         %10 = OpVariable %6 Uniform
          %2 = OpFunction %3 None %4
         %11 = OpLabel
               OpBranchConditional %8 %12 %13
         %12 = OpLabel
         %14 = OpLoad %5 %9
               OpBranch %15
         %13 = OpLabel
         %16 = OpLoad %5 %10
               OpBranch %15
         %15 = OpLabel
         %17 = OpPhi %14 %12 %16 %13
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(14);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(16);
  Instruction* phi = context->get_def_use_mgr()->GetDef(17);
  EXPECT_NE(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
  EXPECT_NE(vtable.GetValueNumber(inst1), vtable.GetValueNumber(phi));
  EXPECT_NE(vtable.GetValueNumber(inst2), vtable.GetValueNumber(phi));
}

// Test that a phi node in a loop header gets a new value because one of its
// inputs comes from later in the loop.
TEST_F(ValueTableTest, PhiLoopTest) {
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
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %9 = OpVariable %6 Uniform
         %10 = OpVariable %6 Uniform
          %2 = OpFunction %3 None %4
         %11 = OpLabel
         %12 = OpLoad %5 %9
               OpSelectionMerge %13 None
               OpBranchConditional %8 %14 %13
         %14 = OpLabel
         %15 = OpPhi %5 %12 %11 %16 %14
         %16 = OpLoad %5 %9
               OpLoopMerge %17 %14 None
               OpBranchConditional %8 %14 %17
         %17 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %18 = OpPhi %5 %12 %11 %16 %17
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ValueNumberTable vtable(context.get());
  Instruction* inst1 = context->get_def_use_mgr()->GetDef(12);
  Instruction* inst2 = context->get_def_use_mgr()->GetDef(16);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));

  Instruction* phi1 = context->get_def_use_mgr()->GetDef(15);
  EXPECT_NE(vtable.GetValueNumber(inst1), vtable.GetValueNumber(phi1));

  Instruction* phi2 = context->get_def_use_mgr()->GetDef(18);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(phi2));
  EXPECT_NE(vtable.GetValueNumber(phi1), vtable.GetValueNumber(phi2));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
