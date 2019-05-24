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

#include "effcee/effcee.h"
#include "gmock/gmock.h"
#include "source/opt/build_module.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/loop_utils.h"
#include "source/opt/pass.h"
#include "test/opt//assembly_builder.h"
#include "test/opt/function_utils.h"

namespace spvtools {
namespace opt {
namespace {

bool Validate(const std::vector<uint32_t>& bin) {
  spv_target_env target_env = SPV_ENV_UNIVERSAL_1_2;
  spv_context spvContext = spvContextCreate(target_env);
  spv_diagnostic diagnostic = nullptr;
  spv_const_binary_t binary = {bin.data(), bin.size()};
  spv_result_t error = spvValidate(spvContext, &binary, &diagnostic);
  if (error != 0) spvDiagnosticPrint(diagnostic);
  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(spvContext);
  return error == 0;
}

void Match(const std::string& original, IRContext* context,
           bool do_validation = true) {
  std::vector<uint32_t> bin;
  context->module()->ToBinary(&bin, true);
  if (do_validation) {
    EXPECT_TRUE(Validate(bin));
  }
  std::string assembly;
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_2);
  EXPECT_TRUE(
      tools.Disassemble(bin, &assembly, SPV_BINARY_TO_TEXT_OPTION_NO_HEADER))
      << "Disassembling failed for shader:\n"
      << assembly << std::endl;
  auto match_result = effcee::Match(assembly, original);
  EXPECT_EQ(effcee::Result::Status::Ok, match_result.status())
      << match_result.message() << "\nChecking result:\n"
      << assembly;
}

using LCSSATest = ::testing::Test;

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
layout(location = 0) out vec4 c;
void main() {
  int i = 0;
  for (; i < 10; i++) {
  }
  if (i != 0) {
    i = 1;
  }
}
*/
TEST_F(LCSSATest, SimpleLCSSA) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] %19 None
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: [[phi:%\w+]] = OpPhi {{%\w+}} %30 %20
; CHECK-NEXT: %27 = OpINotEqual {{%\w+}} [[phi]] %9
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %3 "c"
               OpDecorate %3 Location 0
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeInt 32 1
          %8 = OpTypePointer Function %7
          %9 = OpConstant %7 0
         %10 = OpConstant %7 10
         %11 = OpTypeBool
         %12 = OpConstant %7 1
         %13 = OpTypeFloat 32
         %14 = OpTypeVector %13 4
         %15 = OpTypePointer Output %14
          %3 = OpVariable %15 Output
          %2 = OpFunction %5 None %6
         %16 = OpLabel
               OpBranch %17
         %17 = OpLabel
         %30 = OpPhi %7 %9 %16 %25 %19
               OpLoopMerge %18 %19 None
               OpBranch %20
         %20 = OpLabel
         %22 = OpSLessThan %11 %30 %10
               OpBranchConditional %22 %23 %18
         %23 = OpLabel
               OpBranch %19
         %19 = OpLabel
         %25 = OpIAdd %7 %30 %12
               OpBranch %17
         %18 = OpLabel
         %27 = OpINotEqual %11 %30 %9
               OpSelectionMerge %28 None
               OpBranchConditional %27 %29 %28
         %29 = OpLabel
               OpBranch %28
         %28 = OpLabel
         %31 = OpPhi %7 %30 %18 %12 %29
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

  Loop* loop = ld[17];
  EXPECT_FALSE(loop->IsLCSSA());
  LoopUtils Util(context.get(), loop);
  Util.MakeLoopClosedSSA();
  EXPECT_TRUE(loop->IsLCSSA());
  Match(text, context.get());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
layout(location = 0) out vec4 c;
void main() {
  int i = 0;
  for (; i < 10; i++) {
  }
  if (i != 0) {
    i = 1;
  }
}
*/
// Same test as above, but should reuse an existing phi.
TEST_F(LCSSATest, PhiReuseLCSSA) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] %19 None
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: [[phi:%\w+]] = OpPhi {{%\w+}} %30 %20
; CHECK-NEXT: %27 = OpINotEqual {{%\w+}} [[phi]] %9
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %3 "c"
               OpDecorate %3 Location 0
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeInt 32 1
          %8 = OpTypePointer Function %7
          %9 = OpConstant %7 0
         %10 = OpConstant %7 10
         %11 = OpTypeBool
         %12 = OpConstant %7 1
         %13 = OpTypeFloat 32
         %14 = OpTypeVector %13 4
         %15 = OpTypePointer Output %14
          %3 = OpVariable %15 Output
          %2 = OpFunction %5 None %6
         %16 = OpLabel
               OpBranch %17
         %17 = OpLabel
         %30 = OpPhi %7 %9 %16 %25 %19
               OpLoopMerge %18 %19 None
               OpBranch %20
         %20 = OpLabel
         %22 = OpSLessThan %11 %30 %10
               OpBranchConditional %22 %23 %18
         %23 = OpLabel
               OpBranch %19
         %19 = OpLabel
         %25 = OpIAdd %7 %30 %12
               OpBranch %17
         %18 = OpLabel
         %32 = OpPhi %7 %30 %20
         %27 = OpINotEqual %11 %30 %9
               OpSelectionMerge %28 None
               OpBranchConditional %27 %29 %28
         %29 = OpLabel
               OpBranch %28
         %28 = OpLabel
         %31 = OpPhi %7 %30 %18 %12 %29
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

  Loop* loop = ld[17];
  EXPECT_FALSE(loop->IsLCSSA());
  LoopUtils Util(context.get(), loop);
  Util.MakeLoopClosedSSA();
  EXPECT_TRUE(loop->IsLCSSA());
  Match(text, context.get());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
layout(location = 0) out vec4 c;
void main() {
  int i = 0;
  int j = 0;
  for (; i < 10; i++) {}
  for (; j < 10; j++) {}
  if (j != 0) {
    i = 1;
  }
}
*/
TEST_F(LCSSATest, DualLoopLCSSA) {
  const std::string text = R"(
; CHECK: %20 = OpLabel
; CHECK-NEXT: [[phi:%\w+]] = OpPhi %6 %17 %21
; CHECK: %33 = OpLabel
; CHECK-NEXT: {{%\w+}} = OpPhi {{%\w+}} [[phi]] %28 %11 %34
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %3 "c"
               OpDecorate %3 Location 0
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpConstant %6 0
          %9 = OpConstant %6 10
         %10 = OpTypeBool
         %11 = OpConstant %6 1
         %12 = OpTypeFloat 32
         %13 = OpTypeVector %12 4
         %14 = OpTypePointer Output %13
          %3 = OpVariable %14 Output
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
               OpBranch %19
         %19 = OpLabel
         %18 = OpIAdd %6 %17 %11
               OpBranch %16
         %20 = OpLabel
               OpBranch %24
         %24 = OpLabel
         %25 = OpPhi %6 %8 %20 %26 %27
               OpLoopMerge %28 %27 None
               OpBranch %29
         %29 = OpLabel
         %30 = OpSLessThan %10 %25 %9
               OpBranchConditional %30 %31 %28
         %31 = OpLabel
               OpBranch %27
         %27 = OpLabel
         %26 = OpIAdd %6 %25 %11
               OpBranch %24
         %28 = OpLabel
         %32 = OpINotEqual %10 %25 %8
               OpSelectionMerge %33 None
               OpBranchConditional %32 %34 %33
         %34 = OpLabel
               OpBranch %33
         %33 = OpLabel
         %35 = OpPhi %6 %17 %28 %11 %34
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

  Loop* loop = ld[16];
  EXPECT_FALSE(loop->IsLCSSA());
  LoopUtils Util(context.get(), loop);
  Util.MakeLoopClosedSSA();
  EXPECT_TRUE(loop->IsLCSSA());
  Match(text, context.get());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
layout(location = 0) out vec4 c;
void main() {
  int i = 0;
  if (i != 0) {
    for (; i < 10; i++) {}
  }
  if (i != 0) {
    i = 1;
  }
}
*/
TEST_F(LCSSATest, PhiUserLCSSA) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] %22 None
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: [[phi:%\w+]] = OpPhi {{%\w+}} %20 %24
; CHECK: %17 = OpLabel
; CHECK-NEXT: {{%\w+}} = OpPhi {{%\w+}} %8 %15 [[phi]] %23
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %3 "c"
               OpDecorate %3 Location 0
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpConstant %6 0
          %9 = OpTypeBool
         %10 = OpConstant %6 10
         %11 = OpConstant %6 1
         %12 = OpTypeFloat 32
         %13 = OpTypeVector %12 4
         %14 = OpTypePointer Output %13
          %3 = OpVariable %14 Output
          %2 = OpFunction %4 None %5
         %15 = OpLabel
         %16 = OpINotEqual %9 %8 %8
               OpSelectionMerge %17 None
               OpBranchConditional %16 %18 %17
         %18 = OpLabel
               OpBranch %19
         %19 = OpLabel
         %20 = OpPhi %6 %8 %18 %21 %22
               OpLoopMerge %23 %22 None
               OpBranch %24
         %24 = OpLabel
         %25 = OpSLessThan %9 %20 %10
               OpBranchConditional %25 %26 %23
         %26 = OpLabel
               OpBranch %22
         %22 = OpLabel
         %21 = OpIAdd %6 %20 %11
               OpBranch %19
         %23 = OpLabel
               OpBranch %17
         %17 = OpLabel
         %27 = OpPhi %6 %8 %15 %20 %23
         %28 = OpINotEqual %9 %27 %8
               OpSelectionMerge %29 None
               OpBranchConditional %28 %30 %29
         %30 = OpLabel
               OpBranch %29
         %29 = OpLabel
         %31 = OpPhi %6 %27 %17 %11 %30
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

  Loop* loop = ld[19];
  EXPECT_FALSE(loop->IsLCSSA());
  LoopUtils Util(context.get(), loop);
  Util.MakeLoopClosedSSA();
  EXPECT_TRUE(loop->IsLCSSA());
  Match(text, context.get());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
void main() {
  int i = 0;
  if (i != 0) {
    for (; i < 10; i++) {
      if (i > 5) break;
    }
  }
  if (i != 0) {
    i = 1;
  }
}
*/
TEST_F(LCSSATest, LCSSAWithBreak) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] %19 None
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: [[phi:%\w+]] = OpPhi {{%\w+}} %17 %21 %17 %26
; CHECK: %14 = OpLabel
; CHECK-NEXT: {{%\w+}} = OpPhi {{%\w+}} %7 %12 [[phi]] [[merge]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 1
          %6 = OpTypePointer Function %5
          %7 = OpConstant %5 0
          %8 = OpTypeBool
          %9 = OpConstant %5 10
         %10 = OpConstant %5 5
         %11 = OpConstant %5 1
          %2 = OpFunction %3 None %4
         %12 = OpLabel
         %13 = OpINotEqual %8 %7 %7
               OpSelectionMerge %14 None
               OpBranchConditional %13 %15 %14
         %15 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %17 = OpPhi %5 %7 %15 %18 %19
               OpLoopMerge %20 %19 None
               OpBranch %21
         %21 = OpLabel
         %22 = OpSLessThan %8 %17 %9
               OpBranchConditional %22 %23 %20
         %23 = OpLabel
         %24 = OpSGreaterThan %8 %17 %10
               OpSelectionMerge %25 None
               OpBranchConditional %24 %26 %25
         %26 = OpLabel
               OpBranch %20
         %25 = OpLabel
               OpBranch %19
         %19 = OpLabel
         %18 = OpIAdd %5 %17 %11
               OpBranch %16
         %20 = OpLabel
               OpBranch %14
         %14 = OpLabel
         %27 = OpPhi %5 %7 %12 %17 %20
         %28 = OpINotEqual %8 %27 %7
               OpSelectionMerge %29 None
               OpBranchConditional %28 %30 %29
         %30 = OpLabel
               OpBranch %29
         %29 = OpLabel
         %31 = OpPhi %5 %27 %14 %11 %30
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

  Loop* loop = ld[19];
  EXPECT_FALSE(loop->IsLCSSA());
  LoopUtils Util(context.get(), loop);
  Util.MakeLoopClosedSSA();
  EXPECT_TRUE(loop->IsLCSSA());
  Match(text, context.get());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
void main() {
  int i = 0;
  for (; i < 10; i++) {}
  for (int j = i; j < 10;) { j = i + j; }
}
*/
TEST_F(LCSSATest, LCSSAUseInNonEligiblePhi) {
  const std::string text = R"(
; CHECK: %12 = OpLabel
; CHECK-NEXT: [[def_to_close:%\w+]] = OpPhi {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}} [[continue:%\w+]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: [[closing_phi:%\w+]] = OpPhi {{%\w+}} [[def_to_close]] %17
; CHECK: %16 = OpLabel
; CHECK-NEXT: [[use_in_phi:%\w+]] = OpPhi {{%\w+}} %21 %22 [[closing_phi]] [[merge]]
; CHECK: OpIAdd {{%\w+}} [[closing_phi]] [[use_in_phi]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
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
               OpLoopMerge %16 %15 None
               OpBranch %17
         %17 = OpLabel
         %18 = OpSLessThan %9 %13 %8
               OpBranchConditional %18 %19 %16
         %19 = OpLabel
               OpBranch %15
         %15 = OpLabel
         %14 = OpIAdd %5 %13 %10
               OpBranch %12
         %16 = OpLabel
         %20 = OpPhi %5 %13 %17 %21 %22
               OpLoopMerge %23 %22 None
               OpBranch %24
         %24 = OpLabel
         %25 = OpSLessThan %9 %20 %8
               OpBranchConditional %25 %26 %23
         %26 = OpLabel
         %21 = OpIAdd %5 %13 %20
               OpBranch %22
         %22 = OpLabel
               OpBranch %16
         %23 = OpLabel
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

  Loop* loop = ld[12];
  EXPECT_FALSE(loop->IsLCSSA());
  LoopUtils Util(context.get(), loop);
  Util.MakeLoopClosedSSA();
  EXPECT_TRUE(loop->IsLCSSA());
  Match(text, context.get());
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
