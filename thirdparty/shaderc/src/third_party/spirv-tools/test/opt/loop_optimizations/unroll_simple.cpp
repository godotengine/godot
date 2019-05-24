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

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/opt/loop_unroller.h"
#include "source/opt/loop_utils.h"
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
  float x[4];
  for (int i = 0; i < 4; ++i) {
    x[i] = 1.0f;
  }
}
*/
TEST_F(PassClassTest, SimpleFullyUnrollTest) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(
            OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical GLSL450
            OpEntryPoint Fragment %2 "main" %3
            OpExecutionMode %2 OriginUpperLeft
            OpSource GLSL 330
            OpName %2 "main"
            OpName %5 "x"
            OpName %3 "c"
            OpDecorate %3 Location 0
            %6 = OpTypeVoid
            %7 = OpTypeFunction %6
            %8 = OpTypeInt 32 1
            %9 = OpTypePointer Function %8
            %10 = OpConstant %8 0
            %11 = OpConstant %8 4
            %12 = OpTypeBool
            %13 = OpTypeFloat 32
            %14 = OpTypeInt 32 0
            %15 = OpConstant %14 4
            %16 = OpTypeArray %13 %15
            %17 = OpTypePointer Function %16
            %18 = OpConstant %13 1
            %19 = OpTypePointer Function %13
            %20 = OpConstant %8 1
            %21 = OpTypeVector %13 4
            %22 = OpTypePointer Output %21
            %3 = OpVariable %22 Output
            %2 = OpFunction %6 None %7
            %23 = OpLabel
            %5 = OpVariable %17 Function
            OpBranch %24
            %24 = OpLabel
            %35 = OpPhi %8 %10 %23 %34 %26
            OpLoopMerge %25 %26 Unroll
            OpBranch %27
            %27 = OpLabel
            %29 = OpSLessThan %12 %35 %11
            OpBranchConditional %29 %30 %25
            %30 = OpLabel
            %32 = OpAccessChain %19 %5 %35
            OpStore %32 %18
            OpBranch %26
            %26 = OpLabel
            %34 = OpIAdd %8 %35 %20
            OpBranch %24
            %25 = OpLabel
            OpReturn
            OpFunctionEnd
  )";

  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 330
OpName %2 "main"
OpName %4 "x"
OpName %3 "c"
OpDecorate %3 Location 0
%5 = OpTypeVoid
%6 = OpTypeFunction %5
%7 = OpTypeInt 32 1
%8 = OpTypePointer Function %7
%9 = OpConstant %7 0
%10 = OpConstant %7 4
%11 = OpTypeBool
%12 = OpTypeFloat 32
%13 = OpTypeInt 32 0
%14 = OpConstant %13 4
%15 = OpTypeArray %12 %14
%16 = OpTypePointer Function %15
%17 = OpConstant %12 1
%18 = OpTypePointer Function %12
%19 = OpConstant %7 1
%20 = OpTypeVector %12 4
%21 = OpTypePointer Output %20
%3 = OpVariable %21 Output
%2 = OpFunction %5 None %6
%22 = OpLabel
%4 = OpVariable %16 Function
OpBranch %23
%23 = OpLabel
OpBranch %28
%28 = OpLabel
%29 = OpSLessThan %11 %9 %10
OpBranch %30
%30 = OpLabel
%31 = OpAccessChain %18 %4 %9
OpStore %31 %17
OpBranch %26
%26 = OpLabel
%25 = OpIAdd %7 %9 %19
OpBranch %32
%32 = OpLabel
OpBranch %34
%34 = OpLabel
%35 = OpSLessThan %11 %25 %10
OpBranch %36
%36 = OpLabel
%37 = OpAccessChain %18 %4 %25
OpStore %37 %17
OpBranch %38
%38 = OpLabel
%39 = OpIAdd %7 %25 %19
OpBranch %40
%40 = OpLabel
OpBranch %42
%42 = OpLabel
%43 = OpSLessThan %11 %39 %10
OpBranch %44
%44 = OpLabel
%45 = OpAccessChain %18 %4 %39
OpStore %45 %17
OpBranch %46
%46 = OpLabel
%47 = OpIAdd %7 %39 %19
OpBranch %48
%48 = OpLabel
OpBranch %50
%50 = OpLabel
%51 = OpSLessThan %11 %47 %10
OpBranch %52
%52 = OpLabel
%53 = OpAccessChain %18 %4 %47
OpStore %53 %17
OpBranch %54
%54 = OpLabel
%55 = OpIAdd %7 %47 %19
OpBranch %27
%27 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(text, output, false);
}

template <int factor>
class PartialUnrollerTestPass : public Pass {
 public:
  PartialUnrollerTestPass() : Pass() {}

  const char* name() const override { return "Loop unroller"; }

  Status Process() override {
    for (Function& f : *context()->module()) {
      LoopDescriptor& loop_descriptor = *context()->GetLoopDescriptor(&f);
      for (auto& loop : loop_descriptor) {
        LoopUtils loop_utils{context(), &loop};
        loop_utils.PartiallyUnroll(factor);
      }
    }

    return Pass::Status::SuccessWithChange;
  }
};

/*
Generated from the following GLSL
#version 330 core
layout(location = 0) out vec4 c;
void main() {
  float x[10];
  for (int i = 0; i < 10; ++i) {
    x[i] = 1.0f;
  }
}
*/
TEST_F(PassClassTest, SimplePartialUnroll) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(
            OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical GLSL450
            OpEntryPoint Fragment %2 "main" %3
            OpExecutionMode %2 OriginUpperLeft
            OpSource GLSL 330
            OpName %2 "main"
            OpName %5 "x"
            OpName %3 "c"
            OpDecorate %3 Location 0
            %6 = OpTypeVoid
            %7 = OpTypeFunction %6
            %8 = OpTypeInt 32 1
            %9 = OpTypePointer Function %8
            %10 = OpConstant %8 0
            %11 = OpConstant %8 10
            %12 = OpTypeBool
            %13 = OpTypeFloat 32
            %14 = OpTypeInt 32 0
            %15 = OpConstant %14 10
            %16 = OpTypeArray %13 %15
            %17 = OpTypePointer Function %16
            %18 = OpConstant %13 1
            %19 = OpTypePointer Function %13
            %20 = OpConstant %8 1
            %21 = OpTypeVector %13 4
            %22 = OpTypePointer Output %21
            %3 = OpVariable %22 Output
            %2 = OpFunction %6 None %7
            %23 = OpLabel
            %5 = OpVariable %17 Function
            OpBranch %24
            %24 = OpLabel
            %35 = OpPhi %8 %10 %23 %34 %26
            OpLoopMerge %25 %26 Unroll
            OpBranch %27
            %27 = OpLabel
            %29 = OpSLessThan %12 %35 %11
            OpBranchConditional %29 %30 %25
            %30 = OpLabel
            %32 = OpAccessChain %19 %5 %35
            OpStore %32 %18
            OpBranch %26
            %26 = OpLabel
            %34 = OpIAdd %8 %35 %20
            OpBranch %24
            %25 = OpLabel
            OpReturn
            OpFunctionEnd
  )";

  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 330
OpName %2 "main"
OpName %4 "x"
OpName %3 "c"
OpDecorate %3 Location 0
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
%17 = OpConstant %12 1
%18 = OpTypePointer Function %12
%19 = OpConstant %7 1
%20 = OpTypeVector %12 4
%21 = OpTypePointer Output %20
%3 = OpVariable %21 Output
%2 = OpFunction %5 None %6
%22 = OpLabel
%4 = OpVariable %16 Function
OpBranch %23
%23 = OpLabel
%24 = OpPhi %7 %9 %22 %39 %38
OpLoopMerge %27 %38 DontUnroll
OpBranch %28
%28 = OpLabel
%29 = OpSLessThan %11 %24 %10
OpBranchConditional %29 %30 %27
%30 = OpLabel
%31 = OpAccessChain %18 %4 %24
OpStore %31 %17
OpBranch %26
%26 = OpLabel
%25 = OpIAdd %7 %24 %19
OpBranch %32
%32 = OpLabel
OpBranch %34
%34 = OpLabel
%35 = OpSLessThan %11 %25 %10
OpBranch %36
%36 = OpLabel
%37 = OpAccessChain %18 %4 %25
OpStore %37 %17
OpBranch %38
%38 = OpLabel
%39 = OpIAdd %7 %25 %19
OpBranch %23
%27 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, output, false);
}

/*
Generated from the following GLSL
#version 330 core
layout(location = 0) out vec4 c;
void main() {
  float x[10];
  for (int i = 0; i < 10; ++i) {
    x[i] = 1.0f;
  }
}
*/
TEST_F(PassClassTest, SimpleUnevenPartialUnroll) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(
            OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical GLSL450
            OpEntryPoint Fragment %2 "main" %3
            OpExecutionMode %2 OriginUpperLeft
            OpSource GLSL 330
            OpName %2 "main"
            OpName %5 "x"
            OpName %3 "c"
            OpDecorate %3 Location 0
            %6 = OpTypeVoid
            %7 = OpTypeFunction %6
            %8 = OpTypeInt 32 1
            %9 = OpTypePointer Function %8
            %10 = OpConstant %8 0
            %11 = OpConstant %8 10
            %12 = OpTypeBool
            %13 = OpTypeFloat 32
            %14 = OpTypeInt 32 0
            %15 = OpConstant %14 10
            %16 = OpTypeArray %13 %15
            %17 = OpTypePointer Function %16
            %18 = OpConstant %13 1
            %19 = OpTypePointer Function %13
            %20 = OpConstant %8 1
            %21 = OpTypeVector %13 4
            %22 = OpTypePointer Output %21
            %3 = OpVariable %22 Output
            %2 = OpFunction %6 None %7
            %23 = OpLabel
            %5 = OpVariable %17 Function
            OpBranch %24
            %24 = OpLabel
            %35 = OpPhi %8 %10 %23 %34 %26
            OpLoopMerge %25 %26 Unroll
            OpBranch %27
            %27 = OpLabel
            %29 = OpSLessThan %12 %35 %11
            OpBranchConditional %29 %30 %25
            %30 = OpLabel
            %32 = OpAccessChain %19 %5 %35
            OpStore %32 %18
            OpBranch %26
            %26 = OpLabel
            %34 = OpIAdd %8 %35 %20
            OpBranch %24
            %25 = OpLabel
            OpReturn
            OpFunctionEnd
  )";

  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 330
OpName %2 "main"
OpName %4 "x"
OpName %3 "c"
OpDecorate %3 Location 0
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
%17 = OpConstant %12 1
%18 = OpTypePointer Function %12
%19 = OpConstant %7 1
%20 = OpTypeVector %12 4
%21 = OpTypePointer Output %20
%3 = OpVariable %21 Output
%58 = OpConstant %13 1
%2 = OpFunction %5 None %6
%22 = OpLabel
%4 = OpVariable %16 Function
OpBranch %23
%23 = OpLabel
%24 = OpPhi %7 %9 %22 %25 %26
OpLoopMerge %32 %26 Unroll
OpBranch %28
%28 = OpLabel
%29 = OpSLessThan %11 %24 %58
OpBranchConditional %29 %30 %32
%30 = OpLabel
%31 = OpAccessChain %18 %4 %24
OpStore %31 %17
OpBranch %26
%26 = OpLabel
%25 = OpIAdd %7 %24 %19
OpBranch %23
%32 = OpLabel
OpBranch %33
%33 = OpLabel
%34 = OpPhi %7 %24 %32 %57 %56
OpLoopMerge %41 %56 DontUnroll
OpBranch %35
%35 = OpLabel
%36 = OpSLessThan %11 %34 %10
OpBranchConditional %36 %37 %41
%37 = OpLabel
%38 = OpAccessChain %18 %4 %34
OpStore %38 %17
OpBranch %39
%39 = OpLabel
%40 = OpIAdd %7 %34 %19
OpBranch %42
%42 = OpLabel
OpBranch %44
%44 = OpLabel
%45 = OpSLessThan %11 %40 %10
OpBranch %46
%46 = OpLabel
%47 = OpAccessChain %18 %4 %40
OpStore %47 %17
OpBranch %48
%48 = OpLabel
%49 = OpIAdd %7 %40 %19
OpBranch %50
%50 = OpLabel
OpBranch %52
%52 = OpLabel
%53 = OpSLessThan %11 %49 %10
OpBranch %54
%54 = OpLabel
%55 = OpAccessChain %18 %4 %49
OpStore %55 %17
OpBranch %56
%56 = OpLabel
%57 = OpIAdd %7 %49 %19
OpBranch %33
%41 = OpLabel
OpReturn
%27 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  // By unrolling by a factor that doesn't divide evenly into the number of loop
  // iterations we perfom an additional transform when partially unrolling to
  // account for the remainder.
  SinglePassRunAndCheck<PartialUnrollerTestPass<3>>(text, output, false);
}

/* Generated from
#version 410 core
layout(location=0) flat in int upper_bound;
void main() {
    float x[10];
    for (int i = 2; i < 8; i+=2) {
        x[i] = i;
    }
}
*/
TEST_F(PassClassTest, SimpleLoopIterationsCheck) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %5 "x"
OpName %3 "upper_bound"
OpDecorate %3 Flat
OpDecorate %3 Location 0
%6 = OpTypeVoid
%7 = OpTypeFunction %6
%8 = OpTypeInt 32 1
%9 = OpTypePointer Function %8
%10 = OpConstant %8 2
%11 = OpConstant %8 8
%12 = OpTypeBool
%13 = OpTypeFloat 32
%14 = OpTypeInt 32 0
%15 = OpConstant %14 10
%16 = OpTypeArray %13 %15
%17 = OpTypePointer Function %16
%18 = OpTypePointer Function %13
%19 = OpTypePointer Input %8
%3 = OpVariable %19 Input
%2 = OpFunction %6 None %7
%20 = OpLabel
%5 = OpVariable %17 Function
OpBranch %21
%21 = OpLabel
%34 = OpPhi %8 %10 %20 %33 %23
OpLoopMerge %22 %23 Unroll
OpBranch %24
%24 = OpLabel
%26 = OpSLessThan %12 %34 %11
OpBranchConditional %26 %27 %22
%27 = OpLabel
%30 = OpConvertSToF %13 %34
%31 = OpAccessChain %18 %5 %34
OpStore %31 %30
OpBranch %23
%23 = OpLabel
%33 = OpIAdd %8 %34 %10
OpBranch %21
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  Function* f = spvtest::GetFunction(module, 2);

  LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(f);
  EXPECT_EQ(loop_descriptor.NumLoops(), 1u);

  Loop& loop = loop_descriptor.GetLoopByIndex(0);

  EXPECT_TRUE(loop.HasUnrollLoopControl());

  BasicBlock* condition = loop.FindConditionBlock();
  EXPECT_EQ(condition->id(), 24u);

  Instruction* induction = loop.FindConditionVariable(condition);
  EXPECT_EQ(induction->result_id(), 34u);

  LoopUtils loop_utils{context.get(), &loop};
  EXPECT_TRUE(loop_utils.CanPerformUnroll());

  size_t iterations = 0;
  EXPECT_TRUE(loop.FindNumberOfIterations(induction, &*condition->ctail(),
                                          &iterations));
  EXPECT_EQ(iterations, 3u);
}

/* Generated from
#version 410 core
void main() {
    float x[10];
    for (int i = -1; i < 6; i+=3) {
        x[i] = i;
    }
}
*/
TEST_F(PassClassTest, SimpleLoopIterationsCheckSignedInit) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %5 "x"
OpName %3 "upper_bound"
OpDecorate %3 Flat
OpDecorate %3 Location 0
%6 = OpTypeVoid
%7 = OpTypeFunction %6
%8 = OpTypeInt 32 1
%9 = OpTypePointer Function %8
%10 = OpConstant %8 -1
%11 = OpConstant %8 6
%12 = OpTypeBool
%13 = OpTypeFloat 32
%14 = OpTypeInt 32 0
%15 = OpConstant %14 10
%16 = OpTypeArray %13 %15
%17 = OpTypePointer Function %16
%18 = OpTypePointer Function %13
%19 = OpConstant %8 3
%20 = OpTypePointer Input %8
%3 = OpVariable %20 Input
%2 = OpFunction %6 None %7
%21 = OpLabel
%5 = OpVariable %17 Function
OpBranch %22
%22 = OpLabel
%35 = OpPhi %8 %10 %21 %34 %24
OpLoopMerge %23 %24 None
OpBranch %25
%25 = OpLabel
%27 = OpSLessThan %12 %35 %11
OpBranchConditional %27 %28 %23
%28 = OpLabel
%31 = OpConvertSToF %13 %35
%32 = OpAccessChain %18 %5 %35
OpStore %32 %31
OpBranch %24
%24 = OpLabel
%34 = OpIAdd %8 %35 %19
OpBranch %22
%23 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  Function* f = spvtest::GetFunction(module, 2);

  LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(f);

  EXPECT_EQ(loop_descriptor.NumLoops(), 1u);

  Loop& loop = loop_descriptor.GetLoopByIndex(0);

  EXPECT_FALSE(loop.HasUnrollLoopControl());

  BasicBlock* condition = loop.FindConditionBlock();
  EXPECT_EQ(condition->id(), 25u);

  Instruction* induction = loop.FindConditionVariable(condition);
  EXPECT_EQ(induction->result_id(), 35u);

  LoopUtils loop_utils{context.get(), &loop};
  EXPECT_TRUE(loop_utils.CanPerformUnroll());

  size_t iterations = 0;
  EXPECT_TRUE(loop.FindNumberOfIterations(induction, &*condition->ctail(),
                                          &iterations));
  EXPECT_EQ(iterations, 3u);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
    float out_array[6];
    for (uint i = 0; i < 2; i++) {
      for (int x = 0; x < 3; ++x) {
        out_array[x + i*3] = i;
      }
    }
}
*/
TEST_F(PassClassTest, UnrollNestedLoops) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %35 "out_array"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 0
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 2
         %17 = OpTypeBool
         %19 = OpTypeInt 32 1
         %20 = OpTypePointer Function %19
         %22 = OpConstant %19 0
         %29 = OpConstant %19 3
         %31 = OpTypeFloat 32
         %32 = OpConstant %6 6
         %33 = OpTypeArray %31 %32
         %34 = OpTypePointer Function %33
         %39 = OpConstant %6 3
         %44 = OpTypePointer Function %31
         %47 = OpConstant %19 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %35 = OpVariable %34 Function
               OpBranch %10
         %10 = OpLabel
         %51 = OpPhi %6 %9 %5 %50 %13
               OpLoopMerge %12 %13 Unroll
               OpBranch %14
         %14 = OpLabel
         %18 = OpULessThan %17 %51 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %54 = OpPhi %19 %22 %11 %48 %26
               OpLoopMerge %25 %26 Unroll
               OpBranch %27
         %27 = OpLabel
         %30 = OpSLessThan %17 %54 %29
               OpBranchConditional %30 %24 %25
         %24 = OpLabel
         %37 = OpBitcast %6 %54
         %40 = OpIMul %6 %51 %39
         %41 = OpIAdd %6 %37 %40
         %43 = OpConvertUToF %31 %51
         %45 = OpAccessChain %44 %35 %41
               OpStore %45 %43
               OpBranch %26
         %26 = OpLabel
         %48 = OpIAdd %19 %54 %47
               OpBranch %23
         %25 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %50 = OpIAdd %6 %51 %47
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 0
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpConstant %6 2
%10 = OpTypeBool
%11 = OpTypeInt 32 1
%12 = OpTypePointer Function %11
%13 = OpConstant %11 0
%14 = OpConstant %11 3
%15 = OpTypeFloat 32
%16 = OpConstant %6 6
%17 = OpTypeArray %15 %16
%18 = OpTypePointer Function %17
%19 = OpConstant %6 3
%20 = OpTypePointer Function %15
%21 = OpConstant %11 1
%2 = OpFunction %4 None %5
%22 = OpLabel
%3 = OpVariable %18 Function
OpBranch %23
%23 = OpLabel
OpBranch %28
%28 = OpLabel
%29 = OpULessThan %10 %8 %9
OpBranch %30
%30 = OpLabel
OpBranch %31
%31 = OpLabel
OpBranch %36
%36 = OpLabel
%37 = OpSLessThan %10 %13 %14
OpBranch %38
%38 = OpLabel
%39 = OpBitcast %6 %13
%40 = OpIMul %6 %8 %19
%41 = OpIAdd %6 %39 %40
%42 = OpConvertUToF %15 %8
%43 = OpAccessChain %20 %3 %41
OpStore %43 %42
OpBranch %34
%34 = OpLabel
%33 = OpIAdd %11 %13 %21
OpBranch %44
%44 = OpLabel
OpBranch %46
%46 = OpLabel
%47 = OpSLessThan %10 %33 %14
OpBranch %48
%48 = OpLabel
%49 = OpBitcast %6 %33
%50 = OpIMul %6 %8 %19
%51 = OpIAdd %6 %49 %50
%52 = OpConvertUToF %15 %8
%53 = OpAccessChain %20 %3 %51
OpStore %53 %52
OpBranch %54
%54 = OpLabel
%55 = OpIAdd %11 %33 %21
OpBranch %56
%56 = OpLabel
OpBranch %58
%58 = OpLabel
%59 = OpSLessThan %10 %55 %14
OpBranch %60
%60 = OpLabel
%61 = OpBitcast %6 %55
%62 = OpIMul %6 %8 %19
%63 = OpIAdd %6 %61 %62
%64 = OpConvertUToF %15 %8
%65 = OpAccessChain %20 %3 %63
OpStore %65 %64
OpBranch %66
%66 = OpLabel
%67 = OpIAdd %11 %55 %21
OpBranch %35
%35 = OpLabel
OpBranch %26
%26 = OpLabel
%25 = OpIAdd %6 %8 %21
OpBranch %68
%68 = OpLabel
OpBranch %70
%70 = OpLabel
%71 = OpULessThan %10 %25 %9
OpBranch %72
%72 = OpLabel
OpBranch %73
%73 = OpLabel
OpBranch %74
%74 = OpLabel
%75 = OpSLessThan %10 %13 %14
OpBranch %76
%76 = OpLabel
%77 = OpBitcast %6 %13
%78 = OpIMul %6 %25 %19
%79 = OpIAdd %6 %77 %78
%80 = OpConvertUToF %15 %25
%81 = OpAccessChain %20 %3 %79
OpStore %81 %80
OpBranch %82
%82 = OpLabel
%83 = OpIAdd %11 %13 %21
OpBranch %84
%84 = OpLabel
OpBranch %85
%85 = OpLabel
%86 = OpSLessThan %10 %83 %14
OpBranch %87
%87 = OpLabel
%88 = OpBitcast %6 %83
%89 = OpIMul %6 %25 %19
%90 = OpIAdd %6 %88 %89
%91 = OpConvertUToF %15 %25
%92 = OpAccessChain %20 %3 %90
OpStore %92 %91
OpBranch %93
%93 = OpLabel
%94 = OpIAdd %11 %83 %21
OpBranch %95
%95 = OpLabel
OpBranch %96
%96 = OpLabel
%97 = OpSLessThan %10 %94 %14
OpBranch %98
%98 = OpLabel
%99 = OpBitcast %6 %94
%100 = OpIMul %6 %25 %19
%101 = OpIAdd %6 %99 %100
%102 = OpConvertUToF %15 %25
%103 = OpAccessChain %20 %3 %101
OpStore %103 %102
OpBranch %104
%104 = OpLabel
%105 = OpIAdd %11 %94 %21
OpBranch %106
%106 = OpLabel
OpBranch %107
%107 = OpLabel
%108 = OpIAdd %6 %25 %21
OpBranch %27
%27 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;
  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(text, output, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
    float out_array[2];
    for (int i = -3; i < -1; i++) {
      out_array[3 + i] = i;
    }
}
*/
TEST_F(PassClassTest, NegativeConditionAndInit) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %23 "out_array"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 -3
         %16 = OpConstant %6 -1
         %17 = OpTypeBool
         %19 = OpTypeInt 32 0
         %20 = OpConstant %19 2
         %21 = OpTypeArray %6 %20
         %22 = OpTypePointer Function %21
         %25 = OpConstant %6 3
         %30 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %23 = OpVariable %22 Function
               OpBranch %10
         %10 = OpLabel
         %32 = OpPhi %6 %9 %5 %31 %13
               OpLoopMerge %12 %13 Unroll
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %32 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpIAdd %6 %32 %25
         %28 = OpAccessChain %7 %23 %26
               OpStore %28 %32
               OpBranch %13
         %13 = OpLabel
         %31 = OpIAdd %6 %32 %30
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  const std::string expected = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 -3
%9 = OpConstant %6 -1
%10 = OpTypeBool
%11 = OpTypeInt 32 0
%12 = OpConstant %11 2
%13 = OpTypeArray %6 %12
%14 = OpTypePointer Function %13
%15 = OpConstant %6 3
%16 = OpConstant %6 1
%2 = OpFunction %4 None %5
%17 = OpLabel
%3 = OpVariable %14 Function
OpBranch %18
%18 = OpLabel
OpBranch %23
%23 = OpLabel
%24 = OpSLessThan %10 %8 %9
OpBranch %25
%25 = OpLabel
%26 = OpIAdd %6 %8 %15
%27 = OpAccessChain %7 %3 %26
OpStore %27 %8
OpBranch %21
%21 = OpLabel
%20 = OpIAdd %6 %8 %16
OpBranch %28
%28 = OpLabel
OpBranch %30
%30 = OpLabel
%31 = OpSLessThan %10 %20 %9
OpBranch %32
%32 = OpLabel
%33 = OpIAdd %6 %20 %15
%34 = OpAccessChain %7 %3 %33
OpStore %34 %20
OpBranch %35
%35 = OpLabel
%36 = OpIAdd %6 %20 %16
OpBranch %22
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  // SinglePassRunAndCheck<LoopUnroller>(text, expected, false);

  Function* f = spvtest::GetFunction(module, 4);

  LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(f);
  EXPECT_EQ(loop_descriptor.NumLoops(), 1u);

  Loop& loop = loop_descriptor.GetLoopByIndex(0);

  EXPECT_TRUE(loop.HasUnrollLoopControl());

  BasicBlock* condition = loop.FindConditionBlock();
  EXPECT_EQ(condition->id(), 14u);

  Instruction* induction = loop.FindConditionVariable(condition);
  EXPECT_EQ(induction->result_id(), 32u);

  LoopUtils loop_utils{context.get(), &loop};
  EXPECT_TRUE(loop_utils.CanPerformUnroll());

  size_t iterations = 0;
  EXPECT_TRUE(loop.FindNumberOfIterations(induction, &*condition->ctail(),
                                          &iterations));
  EXPECT_EQ(iterations, 2u);
  SinglePassRunAndCheck<LoopUnroller>(text, expected, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
    float out_array[9];
    for (int i = -10; i < -1; i++) {
      out_array[i] = i;
    }
}
*/
TEST_F(PassClassTest, NegativeConditionAndInitResidualUnroll) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %23 "out_array"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 -10
         %16 = OpConstant %6 -1
         %17 = OpTypeBool
         %19 = OpTypeInt 32 0
         %20 = OpConstant %19 9
         %21 = OpTypeArray %6 %20
         %22 = OpTypePointer Function %21
         %25 = OpConstant %6 10
         %30 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %23 = OpVariable %22 Function
               OpBranch %10
         %10 = OpLabel
         %32 = OpPhi %6 %9 %5 %31 %13
               OpLoopMerge %12 %13 Unroll
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %32 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpIAdd %6 %32 %25
         %28 = OpAccessChain %7 %23 %26
               OpStore %28 %32
               OpBranch %13
         %13 = OpLabel
         %31 = OpIAdd %6 %32 %30
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  const std::string expected = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 -10
%9 = OpConstant %6 -1
%10 = OpTypeBool
%11 = OpTypeInt 32 0
%12 = OpConstant %11 9
%13 = OpTypeArray %6 %12
%14 = OpTypePointer Function %13
%15 = OpConstant %6 10
%16 = OpConstant %6 1
%48 = OpConstant %6 -9
%2 = OpFunction %4 None %5
%17 = OpLabel
%3 = OpVariable %14 Function
OpBranch %18
%18 = OpLabel
%19 = OpPhi %6 %8 %17 %20 %21
OpLoopMerge %28 %21 Unroll
OpBranch %23
%23 = OpLabel
%24 = OpSLessThan %10 %19 %48
OpBranchConditional %24 %25 %28
%25 = OpLabel
%26 = OpIAdd %6 %19 %15
%27 = OpAccessChain %7 %3 %26
OpStore %27 %19
OpBranch %21
%21 = OpLabel
%20 = OpIAdd %6 %19 %16
OpBranch %18
%28 = OpLabel
OpBranch %29
%29 = OpLabel
%30 = OpPhi %6 %19 %28 %47 %46
OpLoopMerge %38 %46 DontUnroll
OpBranch %31
%31 = OpLabel
%32 = OpSLessThan %10 %30 %9
OpBranchConditional %32 %33 %38
%33 = OpLabel
%34 = OpIAdd %6 %30 %15
%35 = OpAccessChain %7 %3 %34
OpStore %35 %30
OpBranch %36
%36 = OpLabel
%37 = OpIAdd %6 %30 %16
OpBranch %39
%39 = OpLabel
OpBranch %41
%41 = OpLabel
%42 = OpSLessThan %10 %37 %9
OpBranch %43
%43 = OpLabel
%44 = OpIAdd %6 %37 %15
%45 = OpAccessChain %7 %3 %44
OpStore %45 %37
OpBranch %46
%46 = OpLabel
%47 = OpIAdd %6 %37 %16
OpBranch %29
%38 = OpLabel
OpReturn
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  Function* f = spvtest::GetFunction(module, 4);

  LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(f);
  EXPECT_EQ(loop_descriptor.NumLoops(), 1u);

  Loop& loop = loop_descriptor.GetLoopByIndex(0);

  EXPECT_TRUE(loop.HasUnrollLoopControl());

  BasicBlock* condition = loop.FindConditionBlock();
  EXPECT_EQ(condition->id(), 14u);

  Instruction* induction = loop.FindConditionVariable(condition);
  EXPECT_EQ(induction->result_id(), 32u);

  LoopUtils loop_utils{context.get(), &loop};
  EXPECT_TRUE(loop_utils.CanPerformUnroll());

  size_t iterations = 0;
  EXPECT_TRUE(loop.FindNumberOfIterations(induction, &*condition->ctail(),
                                          &iterations));
  EXPECT_EQ(iterations, 9u);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, expected, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
    float out_array[10];
    for (uint i = 0; i < 2; i++) {
      for (int x = 0; x < 5; ++x) {
        out_array[x + i*5] = i;
      }
    }
}
*/
TEST_F(PassClassTest, UnrollNestedLoopsValidateDescriptor) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %35 "out_array"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 0
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 2
         %17 = OpTypeBool
         %19 = OpTypeInt 32 1
         %20 = OpTypePointer Function %19
         %22 = OpConstant %19 0
         %29 = OpConstant %19 5
         %31 = OpTypeFloat 32
         %32 = OpConstant %6 10
         %33 = OpTypeArray %31 %32
         %34 = OpTypePointer Function %33
         %39 = OpConstant %6 5
         %44 = OpTypePointer Function %31
         %47 = OpConstant %19 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %35 = OpVariable %34 Function
               OpBranch %10
         %10 = OpLabel
         %51 = OpPhi %6 %9 %5 %50 %13
               OpLoopMerge %12 %13 Unroll
               OpBranch %14
         %14 = OpLabel
         %18 = OpULessThan %17 %51 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %54 = OpPhi %19 %22 %11 %48 %26
               OpLoopMerge %25 %26 Unroll
               OpBranch %27
         %27 = OpLabel
         %30 = OpSLessThan %17 %54 %29
               OpBranchConditional %30 %24 %25
         %24 = OpLabel
         %37 = OpBitcast %6 %54
         %40 = OpIMul %6 %51 %39
         %41 = OpIAdd %6 %37 %40
         %43 = OpConvertUToF %31 %51
         %45 = OpAccessChain %44 %35 %41
               OpStore %45 %43
               OpBranch %26
         %26 = OpLabel
         %48 = OpIAdd %19 %54 %47
               OpBranch %23
         %25 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %50 = OpIAdd %6 %51 %47
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  {  // Test fully unroll
    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                               << text << std::endl;
    SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

    Function* f = spvtest::GetFunction(module, 4);
    LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(f);
    EXPECT_EQ(loop_descriptor.NumLoops(), 2u);

    Loop& outer_loop = loop_descriptor.GetLoopByIndex(1);

    EXPECT_TRUE(outer_loop.HasUnrollLoopControl());

    Loop& inner_loop = loop_descriptor.GetLoopByIndex(0);

    EXPECT_TRUE(inner_loop.HasUnrollLoopControl());

    EXPECT_EQ(outer_loop.GetBlocks().size(), 9u);

    EXPECT_EQ(inner_loop.GetBlocks().size(), 4u);
    EXPECT_EQ(outer_loop.NumImmediateChildren(), 1u);
    EXPECT_EQ(inner_loop.NumImmediateChildren(), 0u);

    {
      LoopUtils loop_utils{context.get(), &inner_loop};
      loop_utils.FullyUnroll();
      loop_utils.Finalize();
    }

    EXPECT_EQ(loop_descriptor.NumLoops(), 1u);
    EXPECT_EQ(outer_loop.GetBlocks().size(), 25u);
    EXPECT_EQ(outer_loop.NumImmediateChildren(), 0u);
    {
      LoopUtils loop_utils{context.get(), &outer_loop};
      loop_utils.FullyUnroll();
      loop_utils.Finalize();
    }
    EXPECT_EQ(loop_descriptor.NumLoops(), 0u);
  }

  {  // Test partially unroll
    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                               << text << std::endl;
    SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

    Function* f = spvtest::GetFunction(module, 4);
    LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(f);
    EXPECT_EQ(loop_descriptor.NumLoops(), 2u);

    Loop& outer_loop = loop_descriptor.GetLoopByIndex(1);

    EXPECT_TRUE(outer_loop.HasUnrollLoopControl());

    Loop& inner_loop = loop_descriptor.GetLoopByIndex(0);

    EXPECT_TRUE(inner_loop.HasUnrollLoopControl());

    EXPECT_EQ(outer_loop.GetBlocks().size(), 9u);

    EXPECT_EQ(inner_loop.GetBlocks().size(), 4u);

    EXPECT_EQ(outer_loop.NumImmediateChildren(), 1u);
    EXPECT_EQ(inner_loop.NumImmediateChildren(), 0u);

    LoopUtils loop_utils{context.get(), &inner_loop};
    loop_utils.PartiallyUnroll(2);
    loop_utils.Finalize();

    // The number of loops should actually grow.
    EXPECT_EQ(loop_descriptor.NumLoops(), 3u);
    EXPECT_EQ(outer_loop.GetBlocks().size(), 18u);
    EXPECT_EQ(outer_loop.NumImmediateChildren(), 2u);
  }
}

/*
Generated from the following GLSL
#version 410 core
void main() {
  float out_array[3];
  for (int i = 3; i > 0; --i) {
    out_array[i] = i;
  }
}
*/
TEST_F(PassClassTest, FullyUnrollNegativeStepLoopTest) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %24 "out_array"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 3
         %16 = OpConstant %6 0
         %17 = OpTypeBool
         %19 = OpTypeFloat 32
         %20 = OpTypeInt 32 0
         %21 = OpConstant %20 3
         %22 = OpTypeArray %19 %21
         %23 = OpTypePointer Function %22
         %28 = OpTypePointer Function %19
         %31 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %24 = OpVariable %23 Function
               OpBranch %10
         %10 = OpLabel
         %33 = OpPhi %6 %9 %5 %32 %13
               OpLoopMerge %12 %13 Unroll
               OpBranch %14
         %14 = OpLabel
         %18 = OpSGreaterThan %17 %33 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpConvertSToF %19 %33
         %29 = OpAccessChain %28 %24 %33
               OpStore %29 %27
               OpBranch %13
         %13 = OpLabel
         %32 = OpISub %6 %33 %31
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 3
%9 = OpConstant %6 0
%10 = OpTypeBool
%11 = OpTypeFloat 32
%12 = OpTypeInt 32 0
%13 = OpConstant %12 3
%14 = OpTypeArray %11 %13
%15 = OpTypePointer Function %14
%16 = OpTypePointer Function %11
%17 = OpConstant %6 1
%2 = OpFunction %4 None %5
%18 = OpLabel
%3 = OpVariable %15 Function
OpBranch %19
%19 = OpLabel
OpBranch %24
%24 = OpLabel
%25 = OpSGreaterThan %10 %8 %9
OpBranch %26
%26 = OpLabel
%27 = OpConvertSToF %11 %8
%28 = OpAccessChain %16 %3 %8
OpStore %28 %27
OpBranch %22
%22 = OpLabel
%21 = OpISub %6 %8 %17
OpBranch %29
%29 = OpLabel
OpBranch %31
%31 = OpLabel
%32 = OpSGreaterThan %10 %21 %9
OpBranch %33
%33 = OpLabel
%34 = OpConvertSToF %11 %21
%35 = OpAccessChain %16 %3 %21
OpStore %35 %34
OpBranch %36
%36 = OpLabel
%37 = OpISub %6 %21 %17
OpBranch %38
%38 = OpLabel
OpBranch %40
%40 = OpLabel
%41 = OpSGreaterThan %10 %37 %9
OpBranch %42
%42 = OpLabel
%43 = OpConvertSToF %11 %37
%44 = OpAccessChain %16 %3 %37
OpStore %44 %43
OpBranch %45
%45 = OpLabel
%46 = OpISub %6 %37 %17
OpBranch %23
%23 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(text, output, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
  float out_array[3];
  for (int i = 9; i > 0; i-=3) {
    out_array[i] = i;
  }
}
*/
TEST_F(PassClassTest, FullyUnrollNegativeNonOneStepLoop) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %24 "out_array"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 9
         %16 = OpConstant %6 0
         %17 = OpTypeBool
         %19 = OpTypeFloat 32
         %20 = OpTypeInt 32 0
         %21 = OpConstant %20 3
         %22 = OpTypeArray %19 %21
         %23 = OpTypePointer Function %22
         %28 = OpTypePointer Function %19
         %30 = OpConstant %6 3
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %24 = OpVariable %23 Function
               OpBranch %10
         %10 = OpLabel
         %33 = OpPhi %6 %9 %5 %32 %13
               OpLoopMerge %12 %13 Unroll
               OpBranch %14
         %14 = OpLabel
         %18 = OpSGreaterThan %17 %33 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpConvertSToF %19 %33
         %29 = OpAccessChain %28 %24 %33
               OpStore %29 %27
               OpBranch %13
         %13 = OpLabel
         %32 = OpISub %6 %33 %30
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 9
%9 = OpConstant %6 0
%10 = OpTypeBool
%11 = OpTypeFloat 32
%12 = OpTypeInt 32 0
%13 = OpConstant %12 3
%14 = OpTypeArray %11 %13
%15 = OpTypePointer Function %14
%16 = OpTypePointer Function %11
%17 = OpConstant %6 3
%2 = OpFunction %4 None %5
%18 = OpLabel
%3 = OpVariable %15 Function
OpBranch %19
%19 = OpLabel
OpBranch %24
%24 = OpLabel
%25 = OpSGreaterThan %10 %8 %9
OpBranch %26
%26 = OpLabel
%27 = OpConvertSToF %11 %8
%28 = OpAccessChain %16 %3 %8
OpStore %28 %27
OpBranch %22
%22 = OpLabel
%21 = OpISub %6 %8 %17
OpBranch %29
%29 = OpLabel
OpBranch %31
%31 = OpLabel
%32 = OpSGreaterThan %10 %21 %9
OpBranch %33
%33 = OpLabel
%34 = OpConvertSToF %11 %21
%35 = OpAccessChain %16 %3 %21
OpStore %35 %34
OpBranch %36
%36 = OpLabel
%37 = OpISub %6 %21 %17
OpBranch %38
%38 = OpLabel
OpBranch %40
%40 = OpLabel
%41 = OpSGreaterThan %10 %37 %9
OpBranch %42
%42 = OpLabel
%43 = OpConvertSToF %11 %37
%44 = OpAccessChain %16 %3 %37
OpStore %44 %43
OpBranch %45
%45 = OpLabel
%46 = OpISub %6 %37 %17
OpBranch %23
%23 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(text, output, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
  float out_array[3];
  for (int i = 0; i < 7; i+=3) {
    out_array[i] = i;
  }
}
*/
TEST_F(PassClassTest, FullyUnrollNonDivisibleStepLoop) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %4 "main"
OpExecutionMode %4 OriginUpperLeft
OpSource GLSL 410
OpName %4 "main"
OpName %24 "out_array"
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%9 = OpConstant %6 0
%16 = OpConstant %6 7
%17 = OpTypeBool
%19 = OpTypeFloat 32
%20 = OpTypeInt 32 0
%21 = OpConstant %20 3
%22 = OpTypeArray %19 %21
%23 = OpTypePointer Function %22
%28 = OpTypePointer Function %19
%30 = OpConstant %6 3
%4 = OpFunction %2 None %3
%5 = OpLabel
%24 = OpVariable %23 Function
OpBranch %10
%10 = OpLabel
%33 = OpPhi %6 %9 %5 %32 %13
OpLoopMerge %12 %13 Unroll
OpBranch %14
%14 = OpLabel
%18 = OpSLessThan %17 %33 %16
OpBranchConditional %18 %11 %12
%11 = OpLabel
%27 = OpConvertSToF %19 %33
%29 = OpAccessChain %28 %24 %33
OpStore %29 %27
OpBranch %13
%13 = OpLabel
%32 = OpIAdd %6 %33 %30
OpBranch %10
%12 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpConstant %6 7
%10 = OpTypeBool
%11 = OpTypeFloat 32
%12 = OpTypeInt 32 0
%13 = OpConstant %12 3
%14 = OpTypeArray %11 %13
%15 = OpTypePointer Function %14
%16 = OpTypePointer Function %11
%17 = OpConstant %6 3
%2 = OpFunction %4 None %5
%18 = OpLabel
%3 = OpVariable %15 Function
OpBranch %19
%19 = OpLabel
OpBranch %24
%24 = OpLabel
%25 = OpSLessThan %10 %8 %9
OpBranch %26
%26 = OpLabel
%27 = OpConvertSToF %11 %8
%28 = OpAccessChain %16 %3 %8
OpStore %28 %27
OpBranch %22
%22 = OpLabel
%21 = OpIAdd %6 %8 %17
OpBranch %29
%29 = OpLabel
OpBranch %31
%31 = OpLabel
%32 = OpSLessThan %10 %21 %9
OpBranch %33
%33 = OpLabel
%34 = OpConvertSToF %11 %21
%35 = OpAccessChain %16 %3 %21
OpStore %35 %34
OpBranch %36
%36 = OpLabel
%37 = OpIAdd %6 %21 %17
OpBranch %38
%38 = OpLabel
OpBranch %40
%40 = OpLabel
%41 = OpSLessThan %10 %37 %9
OpBranch %42
%42 = OpLabel
%43 = OpConvertSToF %11 %37
%44 = OpAccessChain %16 %3 %37
OpStore %44 %43
OpBranch %45
%45 = OpLabel
%46 = OpIAdd %6 %37 %17
OpBranch %23
%23 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(text, output, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
  float out_array[4];
  for (int i = 11; i > 0; i-=3) {
    out_array[i] = i;
  }
}
*/
TEST_F(PassClassTest, FullyUnrollNegativeNonDivisibleStepLoop) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %4 "main"
OpExecutionMode %4 OriginUpperLeft
OpSource GLSL 410
OpName %4 "main"
OpName %24 "out_array"
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%9 = OpConstant %6 11
%16 = OpConstant %6 0
%17 = OpTypeBool
%19 = OpTypeFloat 32
%20 = OpTypeInt 32 0
%21 = OpConstant %20 4
%22 = OpTypeArray %19 %21
%23 = OpTypePointer Function %22
%28 = OpTypePointer Function %19
%30 = OpConstant %6 3
%4 = OpFunction %2 None %3
%5 = OpLabel
%24 = OpVariable %23 Function
OpBranch %10
%10 = OpLabel
%33 = OpPhi %6 %9 %5 %32 %13
OpLoopMerge %12 %13 Unroll
OpBranch %14
%14 = OpLabel
%18 = OpSGreaterThan %17 %33 %16
OpBranchConditional %18 %11 %12
%11 = OpLabel
%27 = OpConvertSToF %19 %33
%29 = OpAccessChain %28 %24 %33
OpStore %29 %27
OpBranch %13
%13 = OpLabel
%32 = OpISub %6 %33 %30
OpBranch %10
%12 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 11
%9 = OpConstant %6 0
%10 = OpTypeBool
%11 = OpTypeFloat 32
%12 = OpTypeInt 32 0
%13 = OpConstant %12 4
%14 = OpTypeArray %11 %13
%15 = OpTypePointer Function %14
%16 = OpTypePointer Function %11
%17 = OpConstant %6 3
%2 = OpFunction %4 None %5
%18 = OpLabel
%3 = OpVariable %15 Function
OpBranch %19
%19 = OpLabel
OpBranch %24
%24 = OpLabel
%25 = OpSGreaterThan %10 %8 %9
OpBranch %26
%26 = OpLabel
%27 = OpConvertSToF %11 %8
%28 = OpAccessChain %16 %3 %8
OpStore %28 %27
OpBranch %22
%22 = OpLabel
%21 = OpISub %6 %8 %17
OpBranch %29
%29 = OpLabel
OpBranch %31
%31 = OpLabel
%32 = OpSGreaterThan %10 %21 %9
OpBranch %33
%33 = OpLabel
%34 = OpConvertSToF %11 %21
%35 = OpAccessChain %16 %3 %21
OpStore %35 %34
OpBranch %36
%36 = OpLabel
%37 = OpISub %6 %21 %17
OpBranch %38
%38 = OpLabel
OpBranch %40
%40 = OpLabel
%41 = OpSGreaterThan %10 %37 %9
OpBranch %42
%42 = OpLabel
%43 = OpConvertSToF %11 %37
%44 = OpAccessChain %16 %3 %37
OpStore %44 %43
OpBranch %45
%45 = OpLabel
%46 = OpISub %6 %37 %17
OpBranch %47
%47 = OpLabel
OpBranch %49
%49 = OpLabel
%50 = OpSGreaterThan %10 %46 %9
OpBranch %51
%51 = OpLabel
%52 = OpConvertSToF %11 %46
%53 = OpAccessChain %16 %3 %46
OpStore %53 %52
OpBranch %54
%54 = OpLabel
%55 = OpISub %6 %46 %17
OpBranch %23
%23 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(text, output, false);
}

// With LocalMultiStoreElimPass
static const std::string multiple_phi_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %8 "foo("
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeFunction %6
         %10 = OpTypePointer Function %6
         %12 = OpConstant %6 0
         %14 = OpConstant %6 3
         %22 = OpConstant %6 6
         %23 = OpTypeBool
         %31 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %40 = OpFunctionCall %6 %8
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %6 None %7
          %9 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %41 = OpPhi %6 %12 %9 %34 %19
         %42 = OpPhi %6 %14 %9 %29 %19
         %43 = OpPhi %6 %12 %9 %32 %19
               OpLoopMerge %18 %19 Unroll
               OpBranch %20
         %20 = OpLabel
         %24 = OpSLessThan %23 %43 %22
               OpBranchConditional %24 %17 %18
         %17 = OpLabel
         %27 = OpIMul %6 %43 %41
         %29 = OpIAdd %6 %42 %27
               OpBranch %19
         %19 = OpLabel
         %32 = OpIAdd %6 %43 %31
         %34 = OpISub %6 %41 %31
               OpBranch %16
         %18 = OpLabel
         %37 = OpIAdd %6 %42 %41
               OpReturnValue %37
               OpFunctionEnd
    )";

TEST_F(PassClassTest, PartiallyUnrollResidualMultipleInductionVariables) {
  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "foo("
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypeFunction %6
%8 = OpTypePointer Function %6
%9 = OpConstant %6 0
%10 = OpConstant %6 3
%11 = OpConstant %6 6
%12 = OpTypeBool
%13 = OpConstant %6 1
%82 = OpTypeInt 32 0
%83 = OpConstant %82 2
%2 = OpFunction %4 None %5
%14 = OpLabel
%15 = OpFunctionCall %6 %3
OpReturn
OpFunctionEnd
%3 = OpFunction %6 None %7
%16 = OpLabel
OpBranch %17
%17 = OpLabel
%18 = OpPhi %6 %9 %16 %19 %20
%21 = OpPhi %6 %10 %16 %22 %20
%23 = OpPhi %6 %9 %16 %24 %20
OpLoopMerge %31 %20 Unroll
OpBranch %26
%26 = OpLabel
%27 = OpSLessThan %12 %23 %83
OpBranchConditional %27 %28 %31
%28 = OpLabel
%29 = OpIMul %6 %23 %18
%22 = OpIAdd %6 %21 %29
OpBranch %20
%20 = OpLabel
%24 = OpIAdd %6 %23 %13
%19 = OpISub %6 %18 %13
OpBranch %17
%31 = OpLabel
OpBranch %32
%32 = OpLabel
%33 = OpPhi %6 %18 %31 %81 %79
%34 = OpPhi %6 %21 %31 %78 %79
%35 = OpPhi %6 %23 %31 %80 %79
OpLoopMerge %44 %79 DontUnroll
OpBranch %36
%36 = OpLabel
%37 = OpSLessThan %12 %35 %11
OpBranchConditional %37 %38 %44
%38 = OpLabel
%39 = OpIMul %6 %35 %33
%40 = OpIAdd %6 %34 %39
OpBranch %41
%41 = OpLabel
%42 = OpIAdd %6 %35 %13
%43 = OpISub %6 %33 %13
OpBranch %46
%46 = OpLabel
OpBranch %50
%50 = OpLabel
%51 = OpSLessThan %12 %42 %11
OpBranch %52
%52 = OpLabel
%53 = OpIMul %6 %42 %43
%54 = OpIAdd %6 %40 %53
OpBranch %55
%55 = OpLabel
%56 = OpIAdd %6 %42 %13
%57 = OpISub %6 %43 %13
OpBranch %58
%58 = OpLabel
OpBranch %62
%62 = OpLabel
%63 = OpSLessThan %12 %56 %11
OpBranch %64
%64 = OpLabel
%65 = OpIMul %6 %56 %57
%66 = OpIAdd %6 %54 %65
OpBranch %67
%67 = OpLabel
%68 = OpIAdd %6 %56 %13
%69 = OpISub %6 %57 %13
OpBranch %70
%70 = OpLabel
OpBranch %74
%74 = OpLabel
%75 = OpSLessThan %12 %68 %11
OpBranch %76
%76 = OpLabel
%77 = OpIMul %6 %68 %69
%78 = OpIAdd %6 %66 %77
OpBranch %79
%79 = OpLabel
%80 = OpIAdd %6 %68 %13
%81 = OpISub %6 %69 %13
OpBranch %32
%44 = OpLabel
%45 = OpIAdd %6 %34 %33
OpReturnValue %45
%25 = OpLabel
%30 = OpIAdd %6 %34 %33
OpReturnValue %30
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, multiple_phi_shader,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << multiple_phi_shader << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<PartialUnrollerTestPass<4>>(multiple_phi_shader, output,
                                                    false);
}

TEST_F(PassClassTest, PartiallyUnrollMultipleInductionVariables) {
  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "foo("
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypeFunction %6
%8 = OpTypePointer Function %6
%9 = OpConstant %6 0
%10 = OpConstant %6 3
%11 = OpConstant %6 6
%12 = OpTypeBool
%13 = OpConstant %6 1
%2 = OpFunction %4 None %5
%14 = OpLabel
%15 = OpFunctionCall %6 %3
OpReturn
OpFunctionEnd
%3 = OpFunction %6 None %7
%16 = OpLabel
OpBranch %17
%17 = OpLabel
%18 = OpPhi %6 %9 %16 %42 %40
%21 = OpPhi %6 %10 %16 %39 %40
%23 = OpPhi %6 %9 %16 %41 %40
OpLoopMerge %25 %40 DontUnroll
OpBranch %26
%26 = OpLabel
%27 = OpSLessThan %12 %23 %11
OpBranchConditional %27 %28 %25
%28 = OpLabel
%29 = OpIMul %6 %23 %18
%22 = OpIAdd %6 %21 %29
OpBranch %20
%20 = OpLabel
%24 = OpIAdd %6 %23 %13
%19 = OpISub %6 %18 %13
OpBranch %31
%31 = OpLabel
OpBranch %35
%35 = OpLabel
%36 = OpSLessThan %12 %24 %11
OpBranch %37
%37 = OpLabel
%38 = OpIMul %6 %24 %19
%39 = OpIAdd %6 %22 %38
OpBranch %40
%40 = OpLabel
%41 = OpIAdd %6 %24 %13
%42 = OpISub %6 %19 %13
OpBranch %17
%25 = OpLabel
%30 = OpIAdd %6 %21 %18
OpReturnValue %30
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, multiple_phi_shader,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << multiple_phi_shader << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(multiple_phi_shader, output,
                                                    false);
}

TEST_F(PassClassTest, FullyUnrollMultipleInductionVariables) {
  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "foo("
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypeFunction %6
%8 = OpTypePointer Function %6
%9 = OpConstant %6 0
%10 = OpConstant %6 3
%11 = OpConstant %6 6
%12 = OpTypeBool
%13 = OpConstant %6 1
%2 = OpFunction %4 None %5
%14 = OpLabel
%15 = OpFunctionCall %6 %3
OpReturn
OpFunctionEnd
%3 = OpFunction %6 None %7
%16 = OpLabel
OpBranch %17
%17 = OpLabel
OpBranch %26
%26 = OpLabel
%27 = OpSLessThan %12 %9 %11
OpBranch %28
%28 = OpLabel
%29 = OpIMul %6 %9 %9
%22 = OpIAdd %6 %10 %29
OpBranch %20
%20 = OpLabel
%24 = OpIAdd %6 %9 %13
%19 = OpISub %6 %9 %13
OpBranch %31
%31 = OpLabel
OpBranch %35
%35 = OpLabel
%36 = OpSLessThan %12 %24 %11
OpBranch %37
%37 = OpLabel
%38 = OpIMul %6 %24 %19
%39 = OpIAdd %6 %22 %38
OpBranch %40
%40 = OpLabel
%41 = OpIAdd %6 %24 %13
%42 = OpISub %6 %19 %13
OpBranch %43
%43 = OpLabel
OpBranch %47
%47 = OpLabel
%48 = OpSLessThan %12 %41 %11
OpBranch %49
%49 = OpLabel
%50 = OpIMul %6 %41 %42
%51 = OpIAdd %6 %39 %50
OpBranch %52
%52 = OpLabel
%53 = OpIAdd %6 %41 %13
%54 = OpISub %6 %42 %13
OpBranch %55
%55 = OpLabel
OpBranch %59
%59 = OpLabel
%60 = OpSLessThan %12 %53 %11
OpBranch %61
%61 = OpLabel
%62 = OpIMul %6 %53 %54
%63 = OpIAdd %6 %51 %62
OpBranch %64
%64 = OpLabel
%65 = OpIAdd %6 %53 %13
%66 = OpISub %6 %54 %13
OpBranch %67
%67 = OpLabel
OpBranch %71
%71 = OpLabel
%72 = OpSLessThan %12 %65 %11
OpBranch %73
%73 = OpLabel
%74 = OpIMul %6 %65 %66
%75 = OpIAdd %6 %63 %74
OpBranch %76
%76 = OpLabel
%77 = OpIAdd %6 %65 %13
%78 = OpISub %6 %66 %13
OpBranch %79
%79 = OpLabel
OpBranch %83
%83 = OpLabel
%84 = OpSLessThan %12 %77 %11
OpBranch %85
%85 = OpLabel
%86 = OpIMul %6 %77 %78
%87 = OpIAdd %6 %75 %86
OpBranch %88
%88 = OpLabel
%89 = OpIAdd %6 %77 %13
%90 = OpISub %6 %78 %13
OpBranch %25
%25 = OpLabel
%30 = OpIAdd %6 %87 %90
OpReturnValue %30
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, multiple_phi_shader,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << multiple_phi_shader << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(multiple_phi_shader, output, false);
}

/*
Generated from the following GLSL
#version 440 core
void main()
{
    int j = 0;
    for (int i = 0; i <= 2; ++i)
        ++j;

    for (int i = 1; i >= 0; --i)
        ++j;
}
*/
TEST_F(PassClassTest, FullyUnrollEqualToOperations) {
  // With LocalMultiStoreElimPass
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 2
         %18 = OpTypeBool
         %21 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %11
         %11 = OpLabel
         %37 = OpPhi %6 %9 %5 %22 %14
         %38 = OpPhi %6 %9 %5 %24 %14
               OpLoopMerge %13 %14 Unroll
               OpBranch %15
         %15 = OpLabel
         %19 = OpSLessThanEqual %18 %38 %17
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
         %22 = OpIAdd %6 %37 %21
               OpBranch %14
         %14 = OpLabel
         %24 = OpIAdd %6 %38 %21
               OpBranch %11
         %13 = OpLabel
               OpBranch %26
         %26 = OpLabel
         %39 = OpPhi %6 %37 %13 %34 %29
         %40 = OpPhi %6 %21 %13 %36 %29
               OpLoopMerge %28 %29 Unroll
               OpBranch %30
         %30 = OpLabel
         %32 = OpSGreaterThanEqual %18 %40 %9
               OpBranchConditional %32 %27 %28
         %27 = OpLabel
         %34 = OpIAdd %6 %39 %21
               OpBranch %29
         %29 = OpLabel
         %36 = OpISub %6 %40 %21
               OpBranch %26
         %28 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 440
OpName %2 "main"
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%5 = OpTypeInt 32 1
%6 = OpTypePointer Function %5
%7 = OpConstant %5 0
%8 = OpConstant %5 2
%9 = OpTypeBool
%10 = OpConstant %5 1
%2 = OpFunction %3 None %4
%11 = OpLabel
OpBranch %12
%12 = OpLabel
OpBranch %19
%19 = OpLabel
%20 = OpSLessThanEqual %9 %7 %8
OpBranch %21
%21 = OpLabel
%14 = OpIAdd %5 %7 %10
OpBranch %15
%15 = OpLabel
%17 = OpIAdd %5 %7 %10
OpBranch %41
%41 = OpLabel
OpBranch %44
%44 = OpLabel
%45 = OpSLessThanEqual %9 %17 %8
OpBranch %46
%46 = OpLabel
%47 = OpIAdd %5 %14 %10
OpBranch %48
%48 = OpLabel
%49 = OpIAdd %5 %17 %10
OpBranch %50
%50 = OpLabel
OpBranch %53
%53 = OpLabel
%54 = OpSLessThanEqual %9 %49 %8
OpBranch %55
%55 = OpLabel
%56 = OpIAdd %5 %47 %10
OpBranch %57
%57 = OpLabel
%58 = OpIAdd %5 %49 %10
OpBranch %18
%18 = OpLabel
OpBranch %22
%22 = OpLabel
OpBranch %29
%29 = OpLabel
%30 = OpSGreaterThanEqual %9 %10 %7
OpBranch %31
%31 = OpLabel
%24 = OpIAdd %5 %56 %10
OpBranch %25
%25 = OpLabel
%27 = OpISub %5 %10 %10
OpBranch %32
%32 = OpLabel
OpBranch %35
%35 = OpLabel
%36 = OpSGreaterThanEqual %9 %27 %7
OpBranch %37
%37 = OpLabel
%38 = OpIAdd %5 %24 %10
OpBranch %39
%39 = OpLabel
%40 = OpISub %5 %27 %10
OpBranch %28
%28 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(text, output, false);
}

// With LocalMultiStoreElimPass
const std::string condition_in_header = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %o
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 430
               OpDecorate %o Location 0
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
        %int = OpTypeInt 32 1
     %int_n2 = OpConstant %int -2
      %int_2 = OpConstant %int 2
       %bool = OpTypeBool
      %float = OpTypeFloat 32
%_ptr_Output_float = OpTypePointer Output %float
          %o = OpVariable %_ptr_Output_float Output
    %float_1 = OpConstant %float 1
       %main = OpFunction %void None %6
         %15 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %27 = OpPhi %int %int_n2 %15 %26 %18
         %21 = OpSLessThanEqual %bool %27 %int_2
               OpLoopMerge %17 %18 Unroll
               OpBranchConditional %21 %22 %17
         %22 = OpLabel
         %23 = OpLoad %float %o
         %24 = OpFAdd %float %23 %float_1
               OpStore %o %24
               OpBranch %18
         %18 = OpLabel
         %26 = OpIAdd %int %27 %int_2
               OpBranch %16
         %17 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

TEST_F(PassClassTest, FullyUnrollConditionIsInHeaderBlock) {
  const std::string output = R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "main" %2
OpExecutionMode %1 OriginUpperLeft
OpSource GLSL 430
OpDecorate %2 Location 0
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%5 = OpTypeInt 32 1
%6 = OpConstant %5 -2
%7 = OpConstant %5 2
%8 = OpTypeBool
%9 = OpTypeFloat 32
%10 = OpTypePointer Output %9
%2 = OpVariable %10 Output
%11 = OpConstant %9 1
%1 = OpFunction %3 None %4
%12 = OpLabel
OpBranch %13
%13 = OpLabel
%17 = OpSLessThanEqual %8 %6 %7
OpBranch %19
%19 = OpLabel
%20 = OpLoad %9 %2
%21 = OpFAdd %9 %20 %11
OpStore %2 %21
OpBranch %16
%16 = OpLabel
%15 = OpIAdd %5 %6 %7
OpBranch %22
%22 = OpLabel
%24 = OpSLessThanEqual %8 %15 %7
OpBranch %25
%25 = OpLabel
%26 = OpLoad %9 %2
%27 = OpFAdd %9 %26 %11
OpStore %2 %27
OpBranch %28
%28 = OpLabel
%29 = OpIAdd %5 %15 %7
OpBranch %30
%30 = OpLabel
%32 = OpSLessThanEqual %8 %29 %7
OpBranch %33
%33 = OpLabel
%34 = OpLoad %9 %2
%35 = OpFAdd %9 %34 %11
OpStore %2 %35
OpBranch %36
%36 = OpLabel
%37 = OpIAdd %5 %29 %7
OpBranch %18
%18 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, condition_in_header,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << condition_in_header << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(condition_in_header, output, false);
}

TEST_F(PassClassTest, PartiallyUnrollResidualConditionIsInHeaderBlock) {
  const std::string output = R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "main" %2
OpExecutionMode %1 OriginUpperLeft
OpSource GLSL 430
OpDecorate %2 Location 0
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%5 = OpTypeInt 32 1
%6 = OpConstant %5 -2
%7 = OpConstant %5 2
%8 = OpTypeBool
%9 = OpTypeFloat 32
%10 = OpTypePointer Output %9
%2 = OpVariable %10 Output
%11 = OpConstant %9 1
%40 = OpTypeInt 32 0
%41 = OpConstant %40 1
%1 = OpFunction %3 None %4
%12 = OpLabel
OpBranch %13
%13 = OpLabel
%14 = OpPhi %5 %6 %12 %15 %16
%17 = OpSLessThanEqual %8 %14 %41
OpLoopMerge %22 %16 Unroll
OpBranchConditional %17 %19 %22
%19 = OpLabel
%20 = OpLoad %9 %2
%21 = OpFAdd %9 %20 %11
OpStore %2 %21
OpBranch %16
%16 = OpLabel
%15 = OpIAdd %5 %14 %7
OpBranch %13
%22 = OpLabel
OpBranch %23
%23 = OpLabel
%24 = OpPhi %5 %14 %22 %39 %38
%25 = OpSLessThanEqual %8 %24 %7
OpLoopMerge %31 %38 DontUnroll
OpBranchConditional %25 %26 %31
%26 = OpLabel
%27 = OpLoad %9 %2
%28 = OpFAdd %9 %27 %11
OpStore %2 %28
OpBranch %29
%29 = OpLabel
%30 = OpIAdd %5 %24 %7
OpBranch %32
%32 = OpLabel
%34 = OpSLessThanEqual %8 %30 %7
OpBranch %35
%35 = OpLabel
%36 = OpLoad %9 %2
%37 = OpFAdd %9 %36 %11
OpStore %2 %37
OpBranch %38
%38 = OpLabel
%39 = OpIAdd %5 %30 %7
OpBranch %23
%31 = OpLabel
OpReturn
%18 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, condition_in_header,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << condition_in_header << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(condition_in_header, output,
                                                    false);
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
TEST_F(PassClassTest, PartiallyUnrollLatchNotContinue) {
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
               OpLoopMerge %24 %23 Unroll
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

  const std::string expected = R"(OpCapability Shader
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
%63 = OpConstant %13 1
%2 = OpFunction %5 None %6
%19 = OpLabel
%3 = OpVariable %8 Function
%4 = OpVariable %16 Function
OpStore %3 %9
OpBranch %20
%20 = OpLabel
%21 = OpPhi %7 %9 %19 %22 %23
OpLoopMerge %31 %25 Unroll
OpBranch %26
%26 = OpLabel
%27 = OpSLessThan %11 %21 %63
OpBranchConditional %27 %28 %31
%28 = OpLabel
%29 = OpConvertSToF %12 %21
%30 = OpAccessChain %17 %4 %21
OpStore %30 %29
OpBranch %25
%25 = OpLabel
%22 = OpIAdd %7 %21 %18
OpStore %3 %22
OpBranch %23
%23 = OpLabel
OpBranch %20
%31 = OpLabel
OpBranch %32
%32 = OpLabel
%33 = OpPhi %7 %21 %31 %61 %62
OpLoopMerge %42 %60 DontUnroll
OpBranch %34
%34 = OpLabel
%35 = OpSLessThan %11 %33 %10
OpBranchConditional %35 %36 %42
%36 = OpLabel
%37 = OpConvertSToF %12 %33
%38 = OpAccessChain %17 %4 %33
OpStore %38 %37
OpBranch %39
%39 = OpLabel
%40 = OpIAdd %7 %33 %18
OpStore %3 %40
OpBranch %41
%41 = OpLabel
OpBranch %43
%43 = OpLabel
OpBranch %45
%45 = OpLabel
%46 = OpSLessThan %11 %40 %10
OpBranch %47
%47 = OpLabel
%48 = OpConvertSToF %12 %40
%49 = OpAccessChain %17 %4 %40
OpStore %49 %48
OpBranch %50
%50 = OpLabel
%51 = OpIAdd %7 %40 %18
OpStore %3 %51
OpBranch %52
%52 = OpLabel
OpBranch %53
%53 = OpLabel
OpBranch %55
%55 = OpLabel
%56 = OpSLessThan %11 %51 %10
OpBranch %57
%57 = OpLabel
%58 = OpConvertSToF %12 %51
%59 = OpAccessChain %17 %4 %51
OpStore %59 %58
OpBranch %60
%60 = OpLabel
%61 = OpIAdd %7 %51 %18
OpStore %3 %61
OpBranch %62
%62 = OpLabel
OpBranch %32
%42 = OpLabel
OpReturn
%24 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<PartialUnrollerTestPass<3>>(text, expected, true);

  // Make sure the latch block information is preserved and propagated correctly
  // by the pass.
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  PartialUnrollerTestPass<3> unroller;
  unroller.SetContextForTesting(context.get());
  unroller.Process();

  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const Function* f = spvtest::GetFunction(module, 2);
  LoopDescriptor ld{context.get(), f};

  EXPECT_EQ(ld.NumLoops(), 2u);

  Loop& loop_1 = ld.GetLoopByIndex(0u);
  EXPECT_NE(loop_1.GetLatchBlock(), loop_1.GetContinueBlock());

  Loop& loop_2 = ld.GetLoopByIndex(1u);
  EXPECT_NE(loop_2.GetLatchBlock(), loop_2.GetContinueBlock());
}

// Test that a loop with a self-referencing OpPhi instruction is handled
// correctly.
TEST_F(PassClassTest, OpPhiSelfReference) {
  const std::string text = R"(
  ; Find the two adds from the unrolled loop
  ; CHECK: OpIAdd
  ; CHECK: OpIAdd
  ; CHECK: OpIAdd %uint %uint_0 %uint_1
  ; CHECK-NEXT: OpReturn
          OpCapability Shader
     %1 = OpExtInstImport "GLSL.std.450"
          OpMemoryModel Logical GLSL450
          OpEntryPoint GLCompute %2 "main"
          OpExecutionMode %2 LocalSize 8 8 1
          OpSource HLSL 600
  %uint = OpTypeInt 32 0
  %void = OpTypeVoid
     %5 = OpTypeFunction %void
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
  %bool = OpTypeBool
  %true = OpConstantTrue %bool
     %2 = OpFunction %void None %5
    %10 = OpLabel
          OpBranch %19
    %19 = OpLabel
    %20 = OpPhi %uint %uint_0 %10 %20 %21
    %22 = OpPhi %uint %uint_0 %10 %23 %21
    %24 = OpULessThanEqual %bool %22 %uint_1
          OpLoopMerge %25 %21 Unroll
          OpBranchConditional %24 %21 %25
    %21 = OpLabel
    %23 = OpIAdd %uint %22 %uint_1
          OpBranch %19
    %25 = OpLabel
    %14 = OpIAdd %uint %20 %uint_1
          OpReturn
          OpFunctionEnd
  )";

  const bool kFullyUnroll = true;
  const uint32_t kUnrollFactor = 0;
  SinglePassRunAndMatch<opt::LoopUnroller>(text, true, kFullyUnroll,
                                           kUnrollFactor);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
