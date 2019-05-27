// Copyright (c) 2018 Google LLC.
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

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/loop_fusion.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using FusionCompatibilityTest = PassTest<::testing::Test>;

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int i = 0; // Can't fuse, i=0 in first & i=10 in second
  for (; i < 10; i++) {}
  for (; i < 10; i++) {}
}
*/
TEST_F(FusionCompatibilityTest, SameInductionVariableDifferentBounds) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %20 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %31 = OpPhi %6 %9 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %31 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpIAdd %6 %31 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpBranch %22
         %22 = OpLabel
         %32 = OpPhi %6 %31 %12 %30 %25
               OpLoopMerge %24 %25 None
               OpBranch %26
         %26 = OpLabel
         %28 = OpSLessThan %17 %32 %16
               OpBranchConditional %28 %23 %24
         %23 = OpLabel
               OpBranch %25
         %25 = OpLabel
         %30 = OpIAdd %6 %32 %20
               OpStore %8 %30
               OpBranch %22
         %24 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  LoopFusion fusion(context.get(), loops[0], loops[1]);
  EXPECT_FALSE(fusion.AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 1
#version 440 core
void main() {
  for (int i = 0; i < 10; i++) {}
  for (int i = 0; i < 10; i++) {}
}
*/
TEST_F(FusionCompatibilityTest, Compatible) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %22 "i"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %20 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %22 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %32 = OpPhi %6 %9 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %32 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpIAdd %6 %32 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpStore %22 %9
               OpBranch %23
         %23 = OpLabel
         %33 = OpPhi %6 %9 %12 %31 %26
               OpLoopMerge %25 %26 None
               OpBranch %27
         %27 = OpLabel
         %29 = OpSLessThan %17 %33 %16
               OpBranchConditional %29 %24 %25
         %24 = OpLabel
               OpBranch %26
         %26 = OpLabel
         %31 = OpIAdd %6 %33 %20
               OpStore %22 %31
               OpBranch %23
         %25 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  LoopFusion fusion(context.get(), loops[0], loops[1]);
  EXPECT_TRUE(fusion.AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 2
#version 440 core
void main() {
  for (int i = 0; i < 10; i++) {}
  for (int j = 0; j < 10; j++) {}
}

*/
TEST_F(FusionCompatibilityTest, DifferentName) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %22 "j"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %20 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %22 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %32 = OpPhi %6 %9 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %32 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpIAdd %6 %32 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpStore %22 %9
               OpBranch %23
         %23 = OpLabel
         %33 = OpPhi %6 %9 %12 %31 %26
               OpLoopMerge %25 %26 None
               OpBranch %27
         %27 = OpLabel
         %29 = OpSLessThan %17 %33 %16
               OpBranchConditional %29 %24 %25
         %24 = OpLabel
               OpBranch %26
         %26 = OpLabel
         %31 = OpIAdd %6 %33 %20
               OpStore %22 %31
               OpBranch %23
         %25 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  LoopFusion fusion(context.get(), loops[0], loops[1]);
  EXPECT_TRUE(fusion.AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  // Can't fuse, different step
  for (int i = 0; i < 10; i++) {}
  for (int j = 0; j < 10; j=j+2) {}
}

*/
TEST_F(FusionCompatibilityTest, SameBoundsDifferentStep) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %22 "j"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %20 = OpConstant %6 1
         %31 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %22 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %33 = OpPhi %6 %9 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %33 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpIAdd %6 %33 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpStore %22 %9
               OpBranch %23
         %23 = OpLabel
         %34 = OpPhi %6 %9 %12 %32 %26
               OpLoopMerge %25 %26 None
               OpBranch %27
         %27 = OpLabel
         %29 = OpSLessThan %17 %34 %16
               OpBranchConditional %29 %24 %25
         %24 = OpLabel
               OpBranch %26
         %26 = OpLabel
         %32 = OpIAdd %6 %34 %31
               OpStore %22 %32
               OpBranch %23
         %25 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  LoopFusion fusion(context.get(), loops[0], loops[1]);
  EXPECT_FALSE(fusion.AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 4
#version 440 core
void main() {
  // Can't fuse, different upper bound
  for (int i = 0; i < 10; i++) {}
  for (int j = 0; j < 20; j++) {}
}

*/
TEST_F(FusionCompatibilityTest, DifferentUpperBound) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %22 "j"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %20 = OpConstant %6 1
         %29 = OpConstant %6 20
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %22 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %33 = OpPhi %6 %9 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %33 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpIAdd %6 %33 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpStore %22 %9
               OpBranch %23
         %23 = OpLabel
         %34 = OpPhi %6 %9 %12 %32 %26
               OpLoopMerge %25 %26 None
               OpBranch %27
         %27 = OpLabel
         %30 = OpSLessThan %17 %34 %29
               OpBranchConditional %30 %24 %25
         %24 = OpLabel
               OpBranch %26
         %26 = OpLabel
         %32 = OpIAdd %6 %34 %20
               OpStore %22 %32
               OpBranch %23
         %25 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  LoopFusion fusion(context.get(), loops[0], loops[1]);
  EXPECT_FALSE(fusion.AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 5
#version 440 core
void main() {
  // Can't fuse, different lower bound
  for (int i = 5; i < 10; i++) {}
  for (int j = 0; j < 10; j++) {}
}

*/
TEST_F(FusionCompatibilityTest, DifferentLowerBound) {
  const std::string text = R"(
                OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %22 "j"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 5
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %20 = OpConstant %6 1
         %23 = OpConstant %6 0
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %22 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %33 = OpPhi %6 %9 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %33 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpIAdd %6 %33 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpStore %22 %23
               OpBranch %24
         %24 = OpLabel
         %34 = OpPhi %6 %23 %12 %32 %27
               OpLoopMerge %26 %27 None
               OpBranch %28
         %28 = OpLabel
         %30 = OpSLessThan %17 %34 %16
               OpBranchConditional %30 %25 %26
         %25 = OpLabel
               OpBranch %27
         %27 = OpLabel
         %32 = OpIAdd %6 %34 %20
               OpStore %22 %32
               OpBranch %24
         %26 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  LoopFusion fusion(context.get(), loops[0], loops[1]);
  EXPECT_FALSE(fusion.AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 6
#version 440 core
void main() {
  // Can't fuse, break in first loop
  for (int i = 0; i < 10; i++) {
    if (i == 5) {
      break;
    }
  }
  for (int j = 0; j < 10; j++) {}
}

*/
TEST_F(FusionCompatibilityTest, Break) {
  const std::string text = R"(
                OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %28 "j"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %20 = OpConstant %6 5
         %26 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %28 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %38 = OpPhi %6 %9 %5 %27 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %38 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %21 = OpIEqual %17 %38 %20
               OpSelectionMerge %23 None
               OpBranchConditional %21 %22 %23
         %22 = OpLabel
               OpBranch %12
         %23 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %27 = OpIAdd %6 %38 %26
               OpStore %8 %27
               OpBranch %10
         %12 = OpLabel
               OpStore %28 %9
               OpBranch %29
         %29 = OpLabel
         %39 = OpPhi %6 %9 %12 %37 %32
               OpLoopMerge %31 %32 None
               OpBranch %33
         %33 = OpLabel
         %35 = OpSLessThan %17 %39 %16
               OpBranchConditional %35 %30 %31
         %30 = OpLabel
               OpBranch %32
         %32 = OpLabel
         %37 = OpIAdd %6 %39 %26
               OpStore %28 %37
               OpBranch %29
         %31 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  LoopFusion fusion(context.get(), loops[0], loops[1]);
  EXPECT_FALSE(fusion.AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
layout(location = 0) in vec4 c;
void main() {
  int N = int(c.x);
  for (int i = 0; i < N; i++) {}
  for (int j = 0; j < N; j++) {}
}

*/
TEST_F(FusionCompatibilityTest, UnknownButSameUpperBound) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %12
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "N"
               OpName %12 "c"
               OpName %19 "i"
               OpName %33 "j"
               OpDecorate %12 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpTypeFloat 32
         %10 = OpTypeVector %9 4
         %11 = OpTypePointer Input %10
         %12 = OpVariable %11 Input
         %13 = OpTypeInt 32 0
         %14 = OpConstant %13 0
         %15 = OpTypePointer Input %9
         %20 = OpConstant %6 0
         %28 = OpTypeBool
         %31 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %33 = OpVariable %7 Function
         %16 = OpAccessChain %15 %12 %14
         %17 = OpLoad %9 %16
         %18 = OpConvertFToS %6 %17
               OpStore %8 %18
               OpStore %19 %20
               OpBranch %21
         %21 = OpLabel
         %44 = OpPhi %6 %20 %5 %32 %24
               OpLoopMerge %23 %24 None
               OpBranch %25
         %25 = OpLabel
         %29 = OpSLessThan %28 %44 %18
               OpBranchConditional %29 %22 %23
         %22 = OpLabel
               OpBranch %24
         %24 = OpLabel
         %32 = OpIAdd %6 %44 %31
               OpStore %19 %32
               OpBranch %21
         %23 = OpLabel
               OpStore %33 %20
               OpBranch %34
         %34 = OpLabel
         %46 = OpPhi %6 %20 %23 %43 %37
               OpLoopMerge %36 %37 None
               OpBranch %38
         %38 = OpLabel
         %41 = OpSLessThan %28 %46 %18
               OpBranchConditional %41 %35 %36
         %35 = OpLabel
               OpBranch %37
         %37 = OpLabel
         %43 = OpIAdd %6 %46 %31
               OpStore %33 %43
               OpBranch %34
         %36 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  LoopFusion fusion(context.get(), loops[0], loops[1]);
  EXPECT_TRUE(fusion.AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
layout(location = 0) in vec4 c;
void main() {
  int N = int(c.x);
  for (int i = 0; N > j; i++) {}
  for (int j = 0; N > j; j++) {}
}
*/
TEST_F(FusionCompatibilityTest, UnknownButSameUpperBoundReverseCondition) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %12
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "N"
               OpName %12 "c"
               OpName %19 "i"
               OpName %33 "j"
               OpDecorate %12 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpTypeFloat 32
         %10 = OpTypeVector %9 4
         %11 = OpTypePointer Input %10
         %12 = OpVariable %11 Input
         %13 = OpTypeInt 32 0
         %14 = OpConstant %13 0
         %15 = OpTypePointer Input %9
         %20 = OpConstant %6 0
         %28 = OpTypeBool
         %31 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %33 = OpVariable %7 Function
         %16 = OpAccessChain %15 %12 %14
         %17 = OpLoad %9 %16
         %18 = OpConvertFToS %6 %17
               OpStore %8 %18
               OpStore %19 %20
               OpBranch %21
         %21 = OpLabel
         %45 = OpPhi %6 %20 %5 %32 %24
               OpLoopMerge %23 %24 None
               OpBranch %25
         %25 = OpLabel
         %29 = OpSGreaterThan %28 %18 %45
               OpBranchConditional %29 %22 %23
         %22 = OpLabel
               OpBranch %24
         %24 = OpLabel
         %32 = OpIAdd %6 %45 %31
               OpStore %19 %32
               OpBranch %21
         %23 = OpLabel
               OpStore %33 %20
               OpBranch %34
         %34 = OpLabel
         %47 = OpPhi %6 %20 %23 %43 %37
               OpLoopMerge %36 %37 None
               OpBranch %38
         %38 = OpLabel
         %41 = OpSGreaterThan %28 %18 %47
               OpBranchConditional %41 %35 %36
         %35 = OpLabel
               OpBranch %37
         %37 = OpLabel
         %43 = OpIAdd %6 %47 %31
               OpStore %33 %43
               OpBranch %34
         %36 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  LoopFusion fusion(context.get(), loops[0], loops[1]);
  EXPECT_TRUE(fusion.AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
layout(location = 0) in vec4 c;
void main() {
  // Can't fuse different bound
  int N = int(c.x);
  for (int i = 0; i < N; i++) {}
  for (int j = 0; j < N+1; j++) {}
}

*/
TEST_F(FusionCompatibilityTest, UnknownUpperBoundAddition) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %12
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "N"
               OpName %12 "c"
               OpName %19 "i"
               OpName %33 "j"
               OpDecorate %12 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpTypeFloat 32
         %10 = OpTypeVector %9 4
         %11 = OpTypePointer Input %10
         %12 = OpVariable %11 Input
         %13 = OpTypeInt 32 0
         %14 = OpConstant %13 0
         %15 = OpTypePointer Input %9
         %20 = OpConstant %6 0
         %28 = OpTypeBool
         %31 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %33 = OpVariable %7 Function
         %16 = OpAccessChain %15 %12 %14
         %17 = OpLoad %9 %16
         %18 = OpConvertFToS %6 %17
               OpStore %8 %18
               OpStore %19 %20
               OpBranch %21
         %21 = OpLabel
         %45 = OpPhi %6 %20 %5 %32 %24
               OpLoopMerge %23 %24 None
               OpBranch %25
         %25 = OpLabel
         %29 = OpSLessThan %28 %45 %18
               OpBranchConditional %29 %22 %23
         %22 = OpLabel
               OpBranch %24
         %24 = OpLabel
         %32 = OpIAdd %6 %45 %31
               OpStore %19 %32
               OpBranch %21
         %23 = OpLabel
               OpStore %33 %20
               OpBranch %34
         %34 = OpLabel
         %47 = OpPhi %6 %20 %23 %44 %37
               OpLoopMerge %36 %37 None
               OpBranch %38
         %38 = OpLabel
         %41 = OpIAdd %6 %18 %31
         %42 = OpSLessThan %28 %47 %41
               OpBranchConditional %42 %35 %36
         %35 = OpLabel
               OpBranch %37
         %37 = OpLabel
         %44 = OpIAdd %6 %47 %31
               OpStore %33 %44
               OpBranch %34
         %36 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  LoopFusion fusion(context.get(), loops[0], loops[1]);
  EXPECT_FALSE(fusion.AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 10
#version 440 core
void main() {
  for (int i = 0; i < 10; i++) {}
  for (int j = 0; j < 10; j++) {}
  for (int k = 0; k < 10; k++) {}
}

*/
TEST_F(FusionCompatibilityTest, SeveralAdjacentLoops) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %22 "j"
               OpName %32 "k"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %20 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %22 = OpVariable %7 Function
         %32 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %42 = OpPhi %6 %9 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %42 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpIAdd %6 %42 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpStore %22 %9
               OpBranch %23
         %23 = OpLabel
         %43 = OpPhi %6 %9 %12 %31 %26
               OpLoopMerge %25 %26 None
               OpBranch %27
         %27 = OpLabel
         %29 = OpSLessThan %17 %43 %16
               OpBranchConditional %29 %24 %25
         %24 = OpLabel
               OpBranch %26
         %26 = OpLabel
         %31 = OpIAdd %6 %43 %20
               OpStore %22 %31
               OpBranch %23
         %25 = OpLabel
               OpStore %32 %9
               OpBranch %33
         %33 = OpLabel
         %44 = OpPhi %6 %9 %25 %41 %36
               OpLoopMerge %35 %36 None
               OpBranch %37
         %37 = OpLabel
         %39 = OpSLessThan %17 %44 %16
               OpBranchConditional %39 %34 %35
         %34 = OpLabel
               OpBranch %36
         %36 = OpLabel
         %41 = OpIAdd %6 %44 %20
               OpStore %32 %41
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 3u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  auto loop_0 = loops[0];
  auto loop_1 = loops[1];
  auto loop_2 = loops[2];

  EXPECT_FALSE(LoopFusion(context.get(), loop_0, loop_0).AreCompatible());
  EXPECT_FALSE(LoopFusion(context.get(), loop_0, loop_2).AreCompatible());
  EXPECT_FALSE(LoopFusion(context.get(), loop_1, loop_0).AreCompatible());
  EXPECT_TRUE(LoopFusion(context.get(), loop_0, loop_1).AreCompatible());
  EXPECT_TRUE(LoopFusion(context.get(), loop_1, loop_2).AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  // Can't fuse, not adjacent
  int x = 0;
  for (int i = 0; i < 10; i++) {
    if (i > 10) {
      x++;
    }
  }
  x++;
  for (int j = 0; j < 10; j++) {}
  for (int k = 0; k < 10; k++) {}
}

*/
TEST_F(FusionCompatibilityTest, NonAdjacentLoops) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "x"
               OpName %10 "i"
               OpName %31 "j"
               OpName %41 "k"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 10
         %18 = OpTypeBool
         %25 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %31 = OpVariable %7 Function
         %41 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
         %52 = OpPhi %6 %9 %5 %56 %14
         %51 = OpPhi %6 %9 %5 %28 %14
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %19 = OpSLessThan %18 %51 %17
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
         %21 = OpSGreaterThan %18 %52 %17
               OpSelectionMerge %23 None
               OpBranchConditional %21 %22 %23
         %22 = OpLabel
         %26 = OpIAdd %6 %52 %25
               OpStore %8 %26
               OpBranch %23
         %23 = OpLabel
         %56 = OpPhi %6 %52 %12 %26 %22
               OpBranch %14
         %14 = OpLabel
         %28 = OpIAdd %6 %51 %25
               OpStore %10 %28
               OpBranch %11
         %13 = OpLabel
         %30 = OpIAdd %6 %52 %25
               OpStore %8 %30
               OpStore %31 %9
               OpBranch %32
         %32 = OpLabel
         %53 = OpPhi %6 %9 %13 %40 %35
               OpLoopMerge %34 %35 None
               OpBranch %36
         %36 = OpLabel
         %38 = OpSLessThan %18 %53 %17
               OpBranchConditional %38 %33 %34
         %33 = OpLabel
               OpBranch %35
         %35 = OpLabel
         %40 = OpIAdd %6 %53 %25
               OpStore %31 %40
               OpBranch %32
         %34 = OpLabel
               OpStore %41 %9
               OpBranch %42
         %42 = OpLabel
         %54 = OpPhi %6 %9 %34 %50 %45
               OpLoopMerge %44 %45 None
               OpBranch %46
         %46 = OpLabel
         %48 = OpSLessThan %18 %54 %17
               OpBranchConditional %48 %43 %44
         %43 = OpLabel
               OpBranch %45
         %45 = OpLabel
         %50 = OpIAdd %6 %54 %25
               OpStore %41 %50
               OpBranch %42
         %44 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 3u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  auto loop_0 = loops[0];
  auto loop_1 = loops[1];
  auto loop_2 = loops[2];

  EXPECT_FALSE(LoopFusion(context.get(), loop_0, loop_0).AreCompatible());
  EXPECT_FALSE(LoopFusion(context.get(), loop_0, loop_2).AreCompatible());
  EXPECT_FALSE(LoopFusion(context.get(), loop_0, loop_1).AreCompatible());
  EXPECT_TRUE(LoopFusion(context.get(), loop_1, loop_2).AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 12
#version 440 core
void main() {
  int j = 0;
  int i = 0;
  for (; i < 10; i++) {}
  for (; j < 10; j++) {}
}

*/
TEST_F(FusionCompatibilityTest, CompatibleInitDeclaredBeforeLoops) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "j"
               OpName %10 "i"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 10
         %18 = OpTypeBool
         %21 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
         %32 = OpPhi %6 %9 %5 %22 %14
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %19 = OpSLessThan %18 %32 %17
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
               OpBranch %14
         %14 = OpLabel
         %22 = OpIAdd %6 %32 %21
               OpStore %10 %22
               OpBranch %11
         %13 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %33 = OpPhi %6 %9 %13 %31 %26
               OpLoopMerge %25 %26 None
               OpBranch %27
         %27 = OpLabel
         %29 = OpSLessThan %18 %33 %17
               OpBranchConditional %29 %24 %25
         %24 = OpLabel
               OpBranch %26
         %26 = OpLabel
         %31 = OpIAdd %6 %33 %21
               OpStore %8 %31
               OpBranch %23
         %25 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  EXPECT_TRUE(LoopFusion(context.get(), loops[0], loops[1]).AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 13 regenerate!
#version 440 core
void main() {
  int[10] a;
  int[10] b;
  // Can't fuse, several induction variables
  for (int j = 0; j < 10; j++) {
    b[i] = a[i];
  }
  for (int i = 0, j = 0; i < 10; i++, j = j+2) {
  }
}

*/
TEST_F(FusionCompatibilityTest, SeveralInductionVariables) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "j"
               OpName %23 "b"
               OpName %25 "a"
               OpName %33 "i"
               OpName %34 "j"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %19 = OpTypeInt 32 0
         %20 = OpConstant %19 10
         %21 = OpTypeArray %6 %20
         %22 = OpTypePointer Function %21
         %31 = OpConstant %6 1
         %48 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %25 = OpVariable %22 Function
         %33 = OpVariable %7 Function
         %34 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %50 = OpPhi %6 %9 %5 %32 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %50 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpAccessChain %7 %25 %50
         %28 = OpLoad %6 %27
         %29 = OpAccessChain %7 %23 %50
               OpStore %29 %28
               OpBranch %13
         %13 = OpLabel
         %32 = OpIAdd %6 %50 %31
               OpStore %8 %32
               OpBranch %10
         %12 = OpLabel
               OpStore %33 %9
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %52 = OpPhi %6 %9 %12 %49 %38
         %51 = OpPhi %6 %9 %12 %46 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %51 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %44 = OpAccessChain %7 %25 %52
               OpStore %44 %51
               OpBranch %38
         %38 = OpLabel
         %46 = OpIAdd %6 %51 %31
               OpStore %33 %46
         %49 = OpIAdd %6 %52 %48
               OpStore %34 %49
               OpBranch %35
         %37 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  EXPECT_FALSE(LoopFusion(context.get(), loops[0], loops[1]).AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 14
#version 440 core
void main() {
  // Fine
  for (int i = 0; i < 10; i = i + 2) {}
  for (int j = 0; j < 10; j = j + 2) {}
}

*/
TEST_F(FusionCompatibilityTest, CompatibleNonIncrementStep) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "j"
               OpName %10 "i"
               OpName %11 "i"
               OpName %24 "j"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %18 = OpConstant %6 10
         %19 = OpTypeBool
         %22 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %11 = OpVariable %7 Function
         %24 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %9
               OpStore %11 %9
               OpBranch %12
         %12 = OpLabel
         %34 = OpPhi %6 %9 %5 %23 %15
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
         %20 = OpSLessThan %19 %34 %18
               OpBranchConditional %20 %13 %14
         %13 = OpLabel
               OpBranch %15
         %15 = OpLabel
         %23 = OpIAdd %6 %34 %22
               OpStore %11 %23
               OpBranch %12
         %14 = OpLabel
               OpStore %24 %9
               OpBranch %25
         %25 = OpLabel
         %35 = OpPhi %6 %9 %14 %33 %28
               OpLoopMerge %27 %28 None
               OpBranch %29
         %29 = OpLabel
         %31 = OpSLessThan %19 %35 %18
               OpBranchConditional %31 %26 %27
         %26 = OpLabel
               OpBranch %28
         %28 = OpLabel
         %33 = OpIAdd %6 %35 %22
               OpStore %24 %33
               OpBranch %25
         %27 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  EXPECT_TRUE(LoopFusion(context.get(), loops[0], loops[1]).AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 15
#version 440 core

int j = 0;

void main() {
  // Not compatible, unknown init for second.
  for (int i = 0; i < 10; i = i + 2) {}
  for (; j < 10; j = j + 2) {}
}

*/
TEST_F(FusionCompatibilityTest, UnknonInitForSecondLoop) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "j"
               OpName %11 "i"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Private %6
          %8 = OpVariable %7 Private
          %9 = OpConstant %6 0
         %10 = OpTypePointer Function %6
         %18 = OpConstant %6 10
         %19 = OpTypeBool
         %22 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %11 = OpVariable %10 Function
               OpStore %8 %9
               OpStore %11 %9
               OpBranch %12
         %12 = OpLabel
         %33 = OpPhi %6 %9 %5 %23 %15
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
         %20 = OpSLessThan %19 %33 %18
               OpBranchConditional %20 %13 %14
         %13 = OpLabel
               OpBranch %15
         %15 = OpLabel
         %23 = OpIAdd %6 %33 %22
               OpStore %11 %23
               OpBranch %12
         %14 = OpLabel
               OpBranch %24
         %24 = OpLabel
               OpLoopMerge %26 %27 None
               OpBranch %28
         %28 = OpLabel
         %29 = OpLoad %6 %8
         %30 = OpSLessThan %19 %29 %18
               OpBranchConditional %30 %25 %26
         %25 = OpLabel
               OpBranch %27
         %27 = OpLabel
         %31 = OpLoad %6 %8
         %32 = OpIAdd %6 %31 %22
               OpStore %8 %32
               OpBranch %24
         %26 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  EXPECT_FALSE(LoopFusion(context.get(), loops[0], loops[1]).AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 16
#version 440 core
void main() {
  // Not compatible, continue in loop 0
  for (int i = 0; i < 10; ++i) {
    if (i % 2 == 1) {
      continue;
    }
  }
  for (int j = 0; j < 10; ++j) {}
}

*/
TEST_F(FusionCompatibilityTest, Continue) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %29 "j"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %20 = OpConstant %6 2
         %22 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %29 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %39 = OpPhi %6 %9 %5 %28 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %39 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %21 = OpSMod %6 %39 %20
         %23 = OpIEqual %17 %21 %22
               OpSelectionMerge %25 None
               OpBranchConditional %23 %24 %25
         %24 = OpLabel
               OpBranch %13
         %25 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %28 = OpIAdd %6 %39 %22
               OpStore %8 %28
               OpBranch %10
         %12 = OpLabel
               OpStore %29 %9
               OpBranch %30
         %30 = OpLabel
         %40 = OpPhi %6 %9 %12 %38 %33
               OpLoopMerge %32 %33 None
               OpBranch %34
         %34 = OpLabel
         %36 = OpSLessThan %17 %40 %16
               OpBranchConditional %36 %31 %32
         %31 = OpLabel
               OpBranch %33
         %33 = OpLabel
         %38 = OpIAdd %6 %40 %22
               OpStore %29 %38
               OpBranch %30
         %32 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  EXPECT_FALSE(LoopFusion(context.get(), loops[0], loops[1]).AreCompatible());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  // Compatible
  for (int i = 0; i < 10; ++i) {
    if (i % 2 == 1) {
    } else {
      a[i] = i;
    }
  }
  for (int j = 0; j < 10; ++j) {}
}

*/
TEST_F(FusionCompatibilityTest, IfElseInLoop) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %31 "a"
               OpName %37 "j"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %20 = OpConstant %6 2
         %22 = OpConstant %6 1
         %27 = OpTypeInt 32 0
         %28 = OpConstant %27 10
         %29 = OpTypeArray %6 %28
         %30 = OpTypePointer Function %29
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %31 = OpVariable %30 Function
         %37 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %47 = OpPhi %6 %9 %5 %36 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %47 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %21 = OpSMod %6 %47 %20
         %23 = OpIEqual %17 %21 %22
               OpSelectionMerge %25 None
               OpBranchConditional %23 %24 %26
         %24 = OpLabel
               OpBranch %25
         %26 = OpLabel
         %34 = OpAccessChain %7 %31 %47
               OpStore %34 %47
               OpBranch %25
         %25 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %36 = OpIAdd %6 %47 %22
               OpStore %8 %36
               OpBranch %10
         %12 = OpLabel
               OpStore %37 %9
               OpBranch %38
         %38 = OpLabel
         %48 = OpPhi %6 %9 %12 %46 %41
               OpLoopMerge %40 %41 None
               OpBranch %42
         %42 = OpLabel
         %44 = OpSLessThan %17 %48 %16
               OpBranchConditional %44 %39 %40
         %39 = OpLabel
               OpBranch %41
         %41 = OpLabel
         %46 = OpIAdd %6 %48 %22
               OpStore %37 %46
               OpBranch %38
         %40 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function& f = *module->begin();
  LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInBinaryLayoutOrder();

  EXPECT_TRUE(LoopFusion(context.get(), loops[0], loops[1]).AreCompatible());
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
