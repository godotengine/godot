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

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/pass.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/function_utils.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::UnorderedElementsAre;
using PassClassTest = PassTest<::testing::Test>;

/*
Generated from the following GLSL
#version 330 core
layout(location = 0) out vec4 c;
void main() {
  int i = 0;
  for(; i < 10; ++i) {
  }
}
*/
TEST_F(PassClassTest, BasicVisitFromEntryPoint) {
  const std::string text = R"(
                OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %5 "i"
               OpName %3 "c"
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %10 = OpConstant %8 0
         %11 = OpConstant %8 10
         %12 = OpTypeBool
         %13 = OpConstant %8 1
         %14 = OpTypeFloat 32
         %15 = OpTypeVector %14 4
         %16 = OpTypePointer Output %15
          %3 = OpVariable %16 Output
          %2 = OpFunction %6 None %7
         %17 = OpLabel
          %5 = OpVariable %9 Function
               OpStore %5 %10
               OpBranch %18
         %18 = OpLabel
               OpLoopMerge %19 %20 None
               OpBranch %21
         %21 = OpLabel
         %22 = OpLoad %8 %5
         %23 = OpSLessThan %12 %22 %11
               OpBranchConditional %23 %24 %19
         %24 = OpLabel
               OpBranch %20
         %20 = OpLabel
         %25 = OpLoad %8 %5
         %26 = OpIAdd %8 %25 %13
               OpStore %5 %26
               OpBranch %18
         %19 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  // clang-format on
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const Function* f = spvtest::GetFunction(module, 2);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  EXPECT_EQ(ld.NumLoops(), 1u);

  Loop& loop = ld.GetLoopByIndex(0);
  EXPECT_EQ(loop.GetHeaderBlock(), spvtest::GetBasicBlock(f, 18));
  EXPECT_EQ(loop.GetLatchBlock(), spvtest::GetBasicBlock(f, 20));
  EXPECT_EQ(loop.GetMergeBlock(), spvtest::GetBasicBlock(f, 19));

  EXPECT_FALSE(loop.HasNestedLoops());
  EXPECT_FALSE(loop.IsNested());
  EXPECT_EQ(loop.GetDepth(), 1u);
}

/*
Generated from the following GLSL:
#version 330 core
layout(location = 0) out vec4 c;
void main() {
  for(int i = 0; i < 10; ++i) {}
  for(int i = 0; i < 10; ++i) {}
}

But it was "hacked" to make the first loop merge block the second loop header.
*/
TEST_F(PassClassTest, LoopWithNoPreHeader) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %4 "i"
               OpName %5 "i"
               OpName %3 "c"
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %10 = OpConstant %8 0
         %11 = OpConstant %8 10
         %12 = OpTypeBool
         %13 = OpConstant %8 1
         %14 = OpTypeFloat 32
         %15 = OpTypeVector %14 4
         %16 = OpTypePointer Output %15
          %3 = OpVariable %16 Output
          %2 = OpFunction %6 None %7
         %17 = OpLabel
          %4 = OpVariable %9 Function
          %5 = OpVariable %9 Function
               OpStore %4 %10
               OpStore %5 %10
               OpBranch %18
         %18 = OpLabel
               OpLoopMerge %27 %20 None
               OpBranch %21
         %21 = OpLabel
         %22 = OpLoad %8 %4
         %23 = OpSLessThan %12 %22 %11
               OpBranchConditional %23 %24 %27
         %24 = OpLabel
               OpBranch %20
         %20 = OpLabel
         %25 = OpLoad %8 %4
         %26 = OpIAdd %8 %25 %13
               OpStore %4 %26
               OpBranch %18
         %27 = OpLabel
               OpLoopMerge %28 %29 None
               OpBranch %30
         %30 = OpLabel
         %31 = OpLoad %8 %5
         %32 = OpSLessThan %12 %31 %11
               OpBranchConditional %32 %33 %28
         %33 = OpLabel
               OpBranch %29
         %29 = OpLabel
         %34 = OpLoad %8 %5
         %35 = OpIAdd %8 %34 %13
               OpStore %5 %35
               OpBranch %27
         %28 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  // clang-format on
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const Function* f = spvtest::GetFunction(module, 2);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  EXPECT_EQ(ld.NumLoops(), 2u);

  Loop* loop = ld[27];
  EXPECT_EQ(loop->GetPreHeaderBlock(), nullptr);
  EXPECT_NE(loop->GetOrCreatePreHeaderBlock(), nullptr);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
in vec4 c;
void main() {
  int i = 0;
  bool cond = c[0] == 0;
  for (; i < 10; i++) {
    if (cond) {
      return;
    }
    else {
      return;
    }
  }
  bool cond2 = i == 9;
}
*/
TEST_F(PassClassTest, NoLoop) {
  const std::string text = R"(; SPIR-V
; Version: 1.0
; Generator: Khronos Glslang Reference Front End; 3
; Bound: 47
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %16
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 330
               OpName %4 "main"
               OpName %16 "c"
               OpDecorate %16 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %10 = OpTypeBool
         %11 = OpTypePointer Function %10
         %13 = OpTypeFloat 32
         %14 = OpTypeVector %13 4
         %15 = OpTypePointer Input %14
         %16 = OpVariable %15 Input
         %17 = OpTypeInt 32 0
         %18 = OpConstant %17 0
         %19 = OpTypePointer Input %13
         %22 = OpConstant %13 0
         %30 = OpConstant %6 10
         %39 = OpConstant %6 1
         %46 = OpUndef %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %20 = OpAccessChain %19 %16 %18
         %21 = OpLoad %13 %20
         %23 = OpFOrdEqual %10 %21 %22
               OpBranch %24
         %24 = OpLabel
         %45 = OpPhi %6 %9 %5 %40 %27
               OpLoopMerge %26 %27 None
               OpBranch %28
         %28 = OpLabel
         %31 = OpSLessThan %10 %45 %30
               OpBranchConditional %31 %25 %26
         %25 = OpLabel
               OpSelectionMerge %34 None
               OpBranchConditional %23 %33 %36
         %33 = OpLabel
               OpReturn
         %36 = OpLabel
               OpReturn
         %34 = OpLabel
               OpBranch %27
         %27 = OpLabel
         %40 = OpIAdd %6 %46 %39
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
  const Function* f = spvtest::GetFunction(module, 4);
  LoopDescriptor ld{context.get(), f};

  EXPECT_EQ(ld.NumLoops(), 0u);
}

/*
Generated from following GLSL with latch block artificially inserted to be
seperate from continue.
#version 430
void main(void) {
    float x[10];
    for (int i = 0; i < 10; ++i) {
      x[i] = i;
    }
}
*/
TEST_F(PassClassTest, LoopLatchNotContinue) {
  const std::string text = R"(OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %2 "main"
               OpName %3 "i"
               OpName %4 "x"
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeInt 32 1
          %8 = OpTypePointer Function %7
          %9 = OpConstant %7 0
         %10 = OpConstant %7 10
         %11 = OpTypeBool
         %12 = OpTypeFloat 32
         %13 = OpTypeInt 32 0
         %14 = OpConstant %13 10
         %15 = OpTypeArray %12 %14
         %16 = OpTypePointer Function %15
         %17 = OpTypePointer Function %12
         %18 = OpConstant %7 1
          %2 = OpFunction %5 None %6
         %19 = OpLabel
          %3 = OpVariable %8 Function
          %4 = OpVariable %16 Function
               OpStore %3 %9
               OpBranch %20
         %20 = OpLabel
         %21 = OpPhi %7 %9 %19 %22 %30
               OpLoopMerge %24 %23 None
               OpBranch %25
         %25 = OpLabel
         %26 = OpSLessThan %11 %21 %10
               OpBranchConditional %26 %27 %24
         %27 = OpLabel
         %28 = OpConvertSToF %12 %21
         %29 = OpAccessChain %17 %4 %21
               OpStore %29 %28
               OpBranch %23
         %23 = OpLabel
         %22 = OpIAdd %7 %21 %18
               OpStore %3 %22
               OpBranch %30
         %30 = OpLabel
               OpBranch %20
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
  const Function* f = spvtest::GetFunction(module, 2);
  LoopDescriptor ld{context.get(), f};

  EXPECT_EQ(ld.NumLoops(), 1u);

  Loop& loop = ld.GetLoopByIndex(0u);

  EXPECT_NE(loop.GetLatchBlock(), loop.GetContinueBlock());

  EXPECT_EQ(loop.GetContinueBlock()->id(), 23u);
  EXPECT_EQ(loop.GetLatchBlock()->id(), 30u);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
