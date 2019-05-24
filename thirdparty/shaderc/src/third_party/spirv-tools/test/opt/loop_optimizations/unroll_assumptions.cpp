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

template <int factor>
class PartialUnrollerTestPass : public Pass {
 public:
  PartialUnrollerTestPass() : Pass() {}

  const char* name() const override { return "Loop unroller"; }

  Status Process() override {
    bool changed = false;
    for (Function& f : *context()->module()) {
      LoopDescriptor& loop_descriptor = *context()->GetLoopDescriptor(&f);
      for (auto& loop : loop_descriptor) {
        LoopUtils loop_utils{context(), &loop};
        if (loop_utils.PartiallyUnroll(factor)) {
          changed = true;
        }
      }
    }

    if (changed) return Pass::Status::SuccessWithChange;
    return Pass::Status::SuccessWithoutChange;
  }
};

/*
Generated from the following GLSL
#version 410 core
layout(location = 0) flat in int in_upper_bound;
void main() {
  for (int i = 0; i < in_upper_bound; ++i) {
    x[i] = 1.0f;
  }
}
*/
TEST_F(PassClassTest, CheckUpperBound) {
  // clang-format off
  // With LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "in_upper_bound"
OpName %4 "x"
OpDecorate %3 Flat
OpDecorate %3 Location 0
%5 = OpTypeVoid
%6 = OpTypeFunction %5
%7 = OpTypeInt 32 1
%8 = OpTypePointer Function %7
%9 = OpConstant %7 0
%10 = OpTypePointer Input %7
%3 = OpVariable %10 Input
%11 = OpTypeBool
%12 = OpTypeFloat 32
%13 = OpTypeInt 32 0
%14 = OpConstant %13 10
%15 = OpTypeArray %12 %14
%16 = OpTypePointer Function %15
%17 = OpConstant %12 1
%18 = OpTypePointer Function %12
%19 = OpConstant %7 1
%2 = OpFunction %5 None %6
%20 = OpLabel
%4 = OpVariable %16 Function
OpBranch %21
%21 = OpLabel
%22 = OpPhi %7 %9 %20 %23 %24
OpLoopMerge %25 %24 Unroll
OpBranch %26
%26 = OpLabel
%27 = OpLoad %7 %3
%28 = OpSLessThan %11 %22 %27
OpBranchConditional %28 %29 %25
%29 = OpLabel
%30 = OpAccessChain %18 %4 %22
OpStore %30 %17
OpBranch %24
%24 = OpLabel
%23 = OpIAdd %7 %22 %19
OpBranch %21
%25 = OpLabel
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
    float out_array[10];
    for (uint i = 0; i < 2; i++) {
      for (float x = 0; x < 5; ++x) {
        out_array[x + i*5] = i;
      }
    }
}
*/
TEST_F(PassClassTest, UnrollNestedLoopsInvalid) {
  // clang-format off
  // With LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
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
%14 = OpConstant %11 5
%15 = OpTypeFloat 32
%16 = OpConstant %6 10
%17 = OpTypeArray %15 %16
%18 = OpTypePointer Function %17
%19 = OpConstant %6 5
%20 = OpTypePointer Function %15
%21 = OpConstant %11 1
%22 = OpUndef %11
%2 = OpFunction %4 None %5
%23 = OpLabel
%3 = OpVariable %18 Function
OpBranch %24
%24 = OpLabel
%25 = OpPhi %6 %8 %23 %26 %27
%28 = OpPhi %11 %22 %23 %29 %27
OpLoopMerge %30 %27 Unroll
OpBranch %31
%31 = OpLabel
%32 = OpULessThan %10 %25 %9
OpBranchConditional %32 %33 %30
%33 = OpLabel
OpBranch %34
%34 = OpLabel
%29 = OpPhi %11 %13 %33 %35 %36
OpLoopMerge %37 %36 None
OpBranch %38
%38 = OpLabel
%39 = OpSLessThan %10 %29 %14
OpBranchConditional %39 %40 %37
%40 = OpLabel
%41 = OpBitcast %6 %29
%42 = OpIMul %6 %25 %19
%43 = OpIAdd %6 %41 %42
%44 = OpConvertUToF %15 %25
%45 = OpAccessChain %20 %3 %43
OpStore %45 %44
OpBranch %36
%36 = OpLabel
%35 = OpIAdd %11 %29 %21
OpBranch %34
%37 = OpLabel
OpBranch %27
%27 = OpLabel
%26 = OpIAdd %6 %25 %21
OpBranch %24
%30 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
}

/*
Generated from the following GLSL
#version 440 core
void main(){
  float x[10];
  for (int i = 0; i < 10; i++) {
    if (i == 5) {
      break;
    }
    x[i] = i;
  }
}
*/
TEST_F(PassClassTest, BreakInBody) {
  // clang-format off
  // With LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 440
OpName %2 "main"
OpName %3 "x"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpConstant %6 10
%10 = OpTypeBool
%11 = OpConstant %6 5
%12 = OpTypeFloat 32
%13 = OpTypeInt 32 0
%14 = OpConstant %13 10
%15 = OpTypeArray %12 %14
%16 = OpTypePointer Function %15
%17 = OpTypePointer Function %12
%18 = OpConstant %6 1
%2 = OpFunction %4 None %5
%19 = OpLabel
%3 = OpVariable %16 Function
OpBranch %20
%20 = OpLabel
%21 = OpPhi %6 %8 %19 %22 %23
OpLoopMerge %24 %23 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %10 %21 %9
OpBranchConditional %26 %27 %24
%27 = OpLabel
%28 = OpIEqual %10 %21 %11
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
OpBranch %24
%29 = OpLabel
%31 = OpConvertSToF %12 %21
%32 = OpAccessChain %17 %3 %21
OpStore %32 %31
OpBranch %23
%23 = OpLabel
%22 = OpIAdd %6 %21 %18
OpBranch %20
%24 = OpLabel
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
}

/*
Generated from the following GLSL
#version 440 core
void main(){
  float x[10];
  for (int i = 0; i < 10; i++) {
    if (i == 5) {
      continue;
    }
    x[i] = i;
  }
}
*/
TEST_F(PassClassTest, ContinueInBody) {
  // clang-format off
  // With LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 440
OpName %2 "main"
OpName %3 "x"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpConstant %6 10
%10 = OpTypeBool
%11 = OpConstant %6 5
%12 = OpTypeFloat 32
%13 = OpTypeInt 32 0
%14 = OpConstant %13 10
%15 = OpTypeArray %12 %14
%16 = OpTypePointer Function %15
%17 = OpTypePointer Function %12
%18 = OpConstant %6 1
%2 = OpFunction %4 None %5
%19 = OpLabel
%3 = OpVariable %16 Function
OpBranch %20
%20 = OpLabel
%21 = OpPhi %6 %8 %19 %22 %23
OpLoopMerge %24 %23 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %10 %21 %9
OpBranchConditional %26 %27 %24
%27 = OpLabel
%28 = OpIEqual %10 %21 %11
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
OpBranch %23
%29 = OpLabel
%31 = OpConvertSToF %12 %21
%32 = OpAccessChain %17 %3 %21
OpStore %32 %31
OpBranch %23
%23 = OpLabel
%22 = OpIAdd %6 %21 %18
OpBranch %20
%24 = OpLabel
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
}

/*
Generated from the following GLSL
#version 440 core
void main(){
  float x[10];
  for (int i = 0; i < 10; i++) {
    if (i == 5) {
      return;
    }
    x[i] = i;
  }
}
*/
TEST_F(PassClassTest, ReturnInBody) {
  // clang-format off
  // With LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 440
OpName %2 "main"
OpName %3 "x"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpConstant %6 10
%10 = OpTypeBool
%11 = OpConstant %6 5
%12 = OpTypeFloat 32
%13 = OpTypeInt 32 0
%14 = OpConstant %13 10
%15 = OpTypeArray %12 %14
%16 = OpTypePointer Function %15
%17 = OpTypePointer Function %12
%18 = OpConstant %6 1
%2 = OpFunction %4 None %5
%19 = OpLabel
%3 = OpVariable %16 Function
OpBranch %20
%20 = OpLabel
%21 = OpPhi %6 %8 %19 %22 %23
OpLoopMerge %24 %23 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %10 %21 %9
OpBranchConditional %26 %27 %24
%27 = OpLabel
%28 = OpIEqual %10 %21 %11
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
OpReturn
%29 = OpLabel
%31 = OpConvertSToF %12 %21
%32 = OpAccessChain %17 %3 %21
OpStore %32 %31
OpBranch %23
%23 = OpLabel
%22 = OpIAdd %6 %21 %18
OpBranch %20
%24 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
}

/*
Generated from the following GLSL
#version 440 core
void main() {
  int j = 0;
  for (int i = 0; i < 10 && i > 0; i++) {
    j++;
  }
}
*/
TEST_F(PassClassTest, MultipleConditionsSingleVariable) {
  // clang-format off
  // With LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
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
%8 = OpConstant %5 10
%9 = OpTypeBool
%10 = OpConstant %5 1
%2 = OpFunction %3 None %4
%11 = OpLabel
OpBranch %12
%12 = OpLabel
%13 = OpPhi %5 %7 %11 %14 %15
%16 = OpPhi %5 %7 %11 %17 %15
OpLoopMerge %18 %15 Unroll
OpBranch %19
%19 = OpLabel
%20 = OpSLessThan %9 %16 %8
%21 = OpSGreaterThan %9 %16 %7
%22 = OpLogicalAnd %9 %20 %21
OpBranchConditional %22 %23 %18
%23 = OpLabel
%14 = OpIAdd %5 %13 %10
OpBranch %15
%15 = OpLabel
%17 = OpIAdd %5 %16 %10
OpBranch %12
%18 = OpLabel
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
Generated from the following GLSL
#version 440 core
void main() {
  int i = 0;
  int j = 0;
  int k = 0;
  for (; i < 10 && j > 0; i++, j++) {
    k++;
  }
}
*/
TEST_F(PassClassTest, MultipleConditionsMultipleVariables) {
  // clang-format off
  // With LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
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
%8 = OpConstant %5 10
%9 = OpTypeBool
%10 = OpConstant %5 1
%2 = OpFunction %3 None %4
%11 = OpLabel
OpBranch %12
%12 = OpLabel
%13 = OpPhi %5 %7 %11 %14 %15
%16 = OpPhi %5 %7 %11 %17 %15
%18 = OpPhi %5 %7 %11 %19 %15
OpLoopMerge %20 %15 Unroll
OpBranch %21
%21 = OpLabel
%22 = OpSLessThan %9 %13 %8
%23 = OpSGreaterThan %9 %16 %7
%24 = OpLogicalAnd %9 %22 %23
OpBranchConditional %24 %25 %20
%25 = OpLabel
%19 = OpIAdd %5 %18 %10
OpBranch %15
%15 = OpLabel
%14 = OpIAdd %5 %13 %10
%17 = OpIAdd %5 %16 %10
OpBranch %12
%20 = OpLabel
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
Generated from the following GLSL
#version 440 core
void main() {
  float i = 0.0;
  int j = 0;
  for (; i < 10; i++) {
    j++;
  }
}
*/
TEST_F(PassClassTest, FloatingPointLoop) {
  // clang-format off
  // With LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 440
OpName %2 "main"
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%5 = OpTypeFloat 32
%6 = OpTypePointer Function %5
%7 = OpConstant %5 0
%8 = OpTypeInt 32 1
%9 = OpTypePointer Function %8
%10 = OpConstant %8 0
%11 = OpConstant %5 10
%12 = OpTypeBool
%13 = OpConstant %8 1
%14 = OpConstant %5 1
%2 = OpFunction %3 None %4
%15 = OpLabel
OpBranch %16
%16 = OpLabel
%17 = OpPhi %5 %7 %15 %18 %19
%20 = OpPhi %8 %10 %15 %21 %19
OpLoopMerge %22 %19 Unroll
OpBranch %23
%23 = OpLabel
%24 = OpFOrdLessThan %12 %17 %11
OpBranchConditional %24 %25 %22
%25 = OpLabel
%21 = OpIAdd %8 %20 %13
OpBranch %19
%19 = OpLabel
%18 = OpFAdd %5 %17 %14
OpBranch %16
%22 = OpLabel
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
Generated from the following GLSL
#version 440 core
void main() {
  int i = 2;
  int j = 0;
  if (j == 0) { i = 5; }
  for (; i < 3; ++i) {
    j++;
  }
}
*/
TEST_F(PassClassTest, InductionPhiOutsideLoop) {
  // clang-format off
  // With LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
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
%7 = OpConstant %5 2
%8 = OpConstant %5 0
%9 = OpTypeBool
%10 = OpConstant %5 5
%11 = OpConstant %5 3
%12 = OpConstant %5 1
%2 = OpFunction %3 None %4
%13 = OpLabel
%14 = OpIEqual %9 %8 %8
OpSelectionMerge %15 None
OpBranchConditional %14 %16 %15
%16 = OpLabel
OpBranch %15
%15 = OpLabel
%17 = OpPhi %5 %7 %13 %10 %16
OpBranch %18
%18 = OpLabel
%19 = OpPhi %5 %17 %15 %20 %21
%22 = OpPhi %5 %8 %15 %23 %21
OpLoopMerge %24 %21 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %9 %19 %11
OpBranchConditional %26 %27 %24
%27 = OpLabel
%23 = OpIAdd %5 %22 %12
OpBranch %21
%21 = OpLabel
%20 = OpIAdd %5 %19 %12
OpBranch %18
%24 = OpLabel
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
Generated from the following GLSL
#version 440 core
void main() {
  int j = 0;
  for (int i = 0; i == 0; ++i) {
    ++j;
  }
  for (int i = 0; i != 3; ++i) {
    ++j;
  }
  for (int i = 0; i < 3; i *= 2) {
    ++j;
  }
  for (int i = 10; i > 3; i /= 2) {
    ++j;
  }
  for (int i = 10; i > 3; i |= 2) {
    ++j;
  }
  for (int i = 10; i > 3; i &= 2) {
    ++j;
  }
  for (int i = 10; i > 3; i ^= 2) {
    ++j;
  }
  for (int i = 0; i < 3; i << 2) {
    ++j;
  }
  for (int i = 10; i > 3; i >> 2) {
    ++j;
  }
}
*/
TEST_F(PassClassTest, UnsupportedLoopTypes) {
  // clang-format off
  // With LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
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
%8 = OpTypeBool
%9 = OpConstant %5 1
%10 = OpConstant %5 3
%11 = OpConstant %5 2
%12 = OpConstant %5 10
%2 = OpFunction %3 None %4
%13 = OpLabel
OpBranch %14
%14 = OpLabel
%15 = OpPhi %5 %7 %13 %16 %17
%18 = OpPhi %5 %7 %13 %19 %17
OpLoopMerge %20 %17 Unroll
OpBranch %21
%21 = OpLabel
%22 = OpIEqual %8 %18 %7
OpBranchConditional %22 %23 %20
%23 = OpLabel
%16 = OpIAdd %5 %15 %9
OpBranch %17
%17 = OpLabel
%19 = OpIAdd %5 %18 %9
OpBranch %14
%20 = OpLabel
OpBranch %24
%24 = OpLabel
%25 = OpPhi %5 %15 %20 %26 %27
%28 = OpPhi %5 %7 %20 %29 %27
OpLoopMerge %30 %27 Unroll
OpBranch %31
%31 = OpLabel
%32 = OpINotEqual %8 %28 %10
OpBranchConditional %32 %33 %30
%33 = OpLabel
%26 = OpIAdd %5 %25 %9
OpBranch %27
%27 = OpLabel
%29 = OpIAdd %5 %28 %9
OpBranch %24
%30 = OpLabel
OpBranch %34
%34 = OpLabel
%35 = OpPhi %5 %25 %30 %36 %37
%38 = OpPhi %5 %7 %30 %39 %37
OpLoopMerge %40 %37 Unroll
OpBranch %41
%41 = OpLabel
%42 = OpSLessThan %8 %38 %10
OpBranchConditional %42 %43 %40
%43 = OpLabel
%36 = OpIAdd %5 %35 %9
OpBranch %37
%37 = OpLabel
%39 = OpIMul %5 %38 %11
OpBranch %34
%40 = OpLabel
OpBranch %44
%44 = OpLabel
%45 = OpPhi %5 %35 %40 %46 %47
%48 = OpPhi %5 %12 %40 %49 %47
OpLoopMerge %50 %47 Unroll
OpBranch %51
%51 = OpLabel
%52 = OpSGreaterThan %8 %48 %10
OpBranchConditional %52 %53 %50
%53 = OpLabel
%46 = OpIAdd %5 %45 %9
OpBranch %47
%47 = OpLabel
%49 = OpSDiv %5 %48 %11
OpBranch %44
%50 = OpLabel
OpBranch %54
%54 = OpLabel
%55 = OpPhi %5 %45 %50 %56 %57
%58 = OpPhi %5 %12 %50 %59 %57
OpLoopMerge %60 %57 Unroll
OpBranch %61
%61 = OpLabel
%62 = OpSGreaterThan %8 %58 %10
OpBranchConditional %62 %63 %60
%63 = OpLabel
%56 = OpIAdd %5 %55 %9
OpBranch %57
%57 = OpLabel
%59 = OpBitwiseOr %5 %58 %11
OpBranch %54
%60 = OpLabel
OpBranch %64
%64 = OpLabel
%65 = OpPhi %5 %55 %60 %66 %67
%68 = OpPhi %5 %12 %60 %69 %67
OpLoopMerge %70 %67 Unroll
OpBranch %71
%71 = OpLabel
%72 = OpSGreaterThan %8 %68 %10
OpBranchConditional %72 %73 %70
%73 = OpLabel
%66 = OpIAdd %5 %65 %9
OpBranch %67
%67 = OpLabel
%69 = OpBitwiseAnd %5 %68 %11
OpBranch %64
%70 = OpLabel
OpBranch %74
%74 = OpLabel
%75 = OpPhi %5 %65 %70 %76 %77
%78 = OpPhi %5 %12 %70 %79 %77
OpLoopMerge %80 %77 Unroll
OpBranch %81
%81 = OpLabel
%82 = OpSGreaterThan %8 %78 %10
OpBranchConditional %82 %83 %80
%83 = OpLabel
%76 = OpIAdd %5 %75 %9
OpBranch %77
%77 = OpLabel
%79 = OpBitwiseXor %5 %78 %11
OpBranch %74
%80 = OpLabel
OpBranch %84
%84 = OpLabel
%85 = OpPhi %5 %75 %80 %86 %87
OpLoopMerge %88 %87 Unroll
OpBranch %89
%89 = OpLabel
%90 = OpSLessThan %8 %7 %10
OpBranchConditional %90 %91 %88
%91 = OpLabel
%86 = OpIAdd %5 %85 %9
OpBranch %87
%87 = OpLabel
%92 = OpShiftLeftLogical %5 %7 %11
OpBranch %84
%88 = OpLabel
OpBranch %93
%93 = OpLabel
%94 = OpPhi %5 %85 %88 %95 %96
OpLoopMerge %97 %96 Unroll
OpBranch %98
%98 = OpLabel
%99 = OpSGreaterThan %8 %12 %10
OpBranchConditional %99 %100 %97
%100 = OpLabel
%95 = OpIAdd %5 %94 %9
OpBranch %96
%96 = OpLabel
%101 = OpShiftRightArithmetic %5 %12 %11
OpBranch %93
%97 = OpLabel
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
#version 430

layout(location = 0) out float o;

void main(void) {
    for (int j = 2; j < 0; j += 1) {
      o += 1.0;
    }
}
*/
TEST_F(PassClassTest, NegativeNumberOfIterations) {
  // clang-format off
  // With LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 430
OpName %2 "main"
OpName %3 "o"
OpDecorate %3 Location 0
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 2
%9 = OpConstant %6 0
%10 = OpTypeBool
%11 = OpTypeFloat 32
%12 = OpTypePointer Output %11
%3 = OpVariable %12 Output
%13 = OpConstant %11 1
%14 = OpConstant %6 1
%2 = OpFunction %4 None %5
%15 = OpLabel
OpBranch %16
%16 = OpLabel
%17 = OpPhi %6 %8 %15 %18 %19
OpLoopMerge %20 %19 None
OpBranch %21
%21 = OpLabel
%22 = OpSLessThan %10 %17 %9
OpBranchConditional %22 %23 %20
%23 = OpLabel
%24 = OpLoad %11 %3
%25 = OpFAdd %11 %24 %13
OpStore %3 %25
OpBranch %19
%19 = OpLabel
%18 = OpIAdd %6 %17 %14
OpBranch %16
%20 = OpLabel
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
#version 430

layout(location = 0) out float o;

void main(void) {
  float s = 0.0;
  for (int j = 0; j < 3; j += 1) {
    s += 1.0;
    j += 1;
  }
  o = s;
}
*/
TEST_F(PassClassTest, MultipleStepOperations) {
  // clang-format off
  // With LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 430
OpName %2 "main"
OpName %3 "o"
OpDecorate %3 Location 0
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeFloat 32
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpTypeInt 32 1
%10 = OpTypePointer Function %9
%11 = OpConstant %9 0
%12 = OpConstant %9 3
%13 = OpTypeBool
%14 = OpConstant %6 1
%15 = OpConstant %9 1
%16 = OpTypePointer Output %6
%3 = OpVariable %16 Output
%2 = OpFunction %4 None %5
%17 = OpLabel
OpBranch %18
%18 = OpLabel
%19 = OpPhi %6 %8 %17 %20 %21
%22 = OpPhi %9 %11 %17 %23 %21
OpLoopMerge %24 %21 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %13 %22 %12
OpBranchConditional %26 %27 %24
%27 = OpLabel
%20 = OpFAdd %6 %19 %14
%28 = OpIAdd %9 %22 %15
OpBranch %21
%21 = OpLabel
%23 = OpIAdd %9 %28 %15
OpBranch %18
%24 = OpLabel
OpStore %3 %19
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
#version 430

layout(location = 0) out float o;

void main(void) {
  float s = 0.0;
  for (int j = 10; j > 20; j -= 1) {
    s += 1.0;
  }
  o = s;
}
*/

TEST_F(PassClassTest, ConditionFalseFromStartGreaterThan) {
  // clang-format off
  // With LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 430
OpName %2 "main"
OpName %3 "o"
OpDecorate %3 Location 0
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeFloat 32
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpTypeInt 32 1
%10 = OpTypePointer Function %9
%11 = OpConstant %9 10
%12 = OpConstant %9 20
%13 = OpTypeBool
%14 = OpConstant %6 1
%15 = OpConstant %9 1
%16 = OpTypePointer Output %6
%3 = OpVariable %16 Output
%2 = OpFunction %4 None %5
%17 = OpLabel
OpBranch %18
%18 = OpLabel
%19 = OpPhi %6 %8 %17 %20 %21
%22 = OpPhi %9 %11 %17 %23 %21
OpLoopMerge %24 %21 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSGreaterThan %13 %22 %12
OpBranchConditional %26 %27 %24
%27 = OpLabel
%20 = OpFAdd %6 %19 %14
OpBranch %21
%21 = OpLabel
%23 = OpISub %9 %22 %15
OpBranch %18
%24 = OpLabel
OpStore %3 %19
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
#version 430

layout(location = 0) out float o;

void main(void) {
  float s = 0.0;
  for (int j = 10; j >= 20; j -= 1) {
    s += 1.0;
  }
  o = s;
}
*/
TEST_F(PassClassTest, ConditionFalseFromStartGreaterThanOrEqual) {
  // clang-format off
  // With LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 430
OpName %2 "main"
OpName %3 "o"
OpDecorate %3 Location 0
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeFloat 32
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpTypeInt 32 1
%10 = OpTypePointer Function %9
%11 = OpConstant %9 10
%12 = OpConstant %9 20
%13 = OpTypeBool
%14 = OpConstant %6 1
%15 = OpConstant %9 1
%16 = OpTypePointer Output %6
%3 = OpVariable %16 Output
%2 = OpFunction %4 None %5
%17 = OpLabel
OpBranch %18
%18 = OpLabel
%19 = OpPhi %6 %8 %17 %20 %21
%22 = OpPhi %9 %11 %17 %23 %21
OpLoopMerge %24 %21 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSGreaterThanEqual %13 %22 %12
OpBranchConditional %26 %27 %24
%27 = OpLabel
%20 = OpFAdd %6 %19 %14
OpBranch %21
%21 = OpLabel
%23 = OpISub %9 %22 %15
OpBranch %18
%24 = OpLabel
OpStore %3 %19
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
#version 430

layout(location = 0) out float o;

void main(void) {
  float s = 0.0;
  for (int j = 20; j < 10; j -= 1) {
    s += 1.0;
  }
  o = s;
}
*/
TEST_F(PassClassTest, ConditionFalseFromStartLessThan) {
  // clang-format off
  // With LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 430
OpName %2 "main"
OpName %3 "o"
OpDecorate %3 Location 0
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeFloat 32
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpTypeInt 32 1
%10 = OpTypePointer Function %9
%11 = OpConstant %9 20
%12 = OpConstant %9 10
%13 = OpTypeBool
%14 = OpConstant %6 1
%15 = OpConstant %9 1
%16 = OpTypePointer Output %6
%3 = OpVariable %16 Output
%2 = OpFunction %4 None %5
%17 = OpLabel
OpBranch %18
%18 = OpLabel
%19 = OpPhi %6 %8 %17 %20 %21
%22 = OpPhi %9 %11 %17 %23 %21
OpLoopMerge %24 %21 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %13 %22 %12
OpBranchConditional %26 %27 %24
%27 = OpLabel
%20 = OpFAdd %6 %19 %14
OpBranch %21
%21 = OpLabel
%23 = OpISub %9 %22 %15
OpBranch %18
%24 = OpLabel
OpStore %3 %19
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
#version 430

layout(location = 0) out float o;

void main(void) {
  float s = 0.0;
  for (int j = 20; j <= 10; j -= 1) {
    s += 1.0;
  }
  o = s;
}
*/
TEST_F(PassClassTest, ConditionFalseFromStartLessThanEqual) {
  // clang-format off
  // With LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 430
OpName %2 "main"
OpName %3 "o"
OpDecorate %3 Location 0
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeFloat 32
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpTypeInt 32 1
%10 = OpTypePointer Function %9
%11 = OpConstant %9 20
%12 = OpConstant %9 10
%13 = OpTypeBool
%14 = OpConstant %6 1
%15 = OpConstant %9 1
%16 = OpTypePointer Output %6
%3 = OpVariable %16 Output
%2 = OpFunction %4 None %5
%17 = OpLabel
OpBranch %18
%18 = OpLabel
%19 = OpPhi %6 %8 %17 %20 %21
%22 = OpPhi %9 %11 %17 %23 %21
OpLoopMerge %24 %21 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSLessThanEqual %13 %22 %12
OpBranchConditional %26 %27 %24
%27 = OpLabel
%20 = OpFAdd %6 %19 %14
OpBranch %21
%21 = OpLabel
%23 = OpISub %9 %22 %15
OpBranch %18
%24 = OpLabel
OpStore %3 %19
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

  LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
