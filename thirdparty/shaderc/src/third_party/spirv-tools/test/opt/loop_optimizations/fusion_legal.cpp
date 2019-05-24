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

#include "effcee/effcee.h"
#include "gmock/gmock.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/loop_fusion.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using FusionLegalTest = PassTest<::testing::Test>;

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

void Match(const std::string& checks, IRContext* context) {
  // Silence unused warnings with !defined(SPIRV_EFFCE)
  (void)checks;

  std::vector<uint32_t> bin;
  context->module()->ToBinary(&bin, true);
  EXPECT_TRUE(Validate(bin));
  std::string assembly;
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_2);
  EXPECT_TRUE(
      tools.Disassemble(bin, &assembly, SPV_BINARY_TO_TEXT_OPTION_NO_HEADER))
      << "Disassembling failed for shader:\n"
      << assembly << std::endl;
  auto match_result = effcee::Match(assembly, checks);
  EXPECT_EQ(effcee::Result::Status::Ok, match_result.status())
      << match_result.message() << "\nChecking result:\n"
      << assembly;
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  // No dependence, legal
  for (int i = 0; i < 10; i++) {
    a[i] = a[i]*2;
  }
  for (int i = 0; i < 10; i++) {
    b[i] = b[i]+2;
  }
}

*/
TEST_F(FusionLegalTest, DifferentArraysInLoops) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %34 "i"
               OpName %42 "b"
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
         %28 = OpConstant %6 2
         %32 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %51 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %51 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpAccessChain %7 %23 %51
         %27 = OpLoad %6 %26
         %29 = OpIMul %6 %27 %28
         %30 = OpAccessChain %7 %23 %51
               OpStore %30 %29
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %51 %32
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %52 = OpPhi %6 %9 %12 %50 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %52 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpAccessChain %7 %42 %52
         %46 = OpLoad %6 %45
         %47 = OpIAdd %6 %46 %28
         %48 = OpAccessChain %7 %42 %52
               OpStore %48 %47
               OpBranch %38
         %38 = OpLabel
         %50 = OpIAdd %6 %52 %32
               OpStore %34 %50
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

  LoopFusion fusion(context.get(), loops[0], loops[1]);

  EXPECT_TRUE(fusion.AreCompatible());
  EXPECT_TRUE(fusion.IsLegal());

  fusion.Fuse();

  std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
)";

  Match(checks, context.get());
  auto& ld_final = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld_final.NumLoops(), 1u);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Only loads to the same array, legal
  for (int i = 0; i < 10; i++) {
    b[i] = a[i]*2;
  }
  for (int i = 0; i < 10; i++) {
    c[i] = a[i]+2;
  }
}

*/
TEST_F(FusionLegalTest, OnlyLoadsToSameArray) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "b"
               OpName %25 "a"
               OpName %35 "i"
               OpName %43 "c"
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
         %29 = OpConstant %6 2
         %33 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %25 = OpVariable %22 Function
         %35 = OpVariable %7 Function
         %43 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %52 = OpPhi %6 %9 %5 %34 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %52 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpAccessChain %7 %25 %52
         %28 = OpLoad %6 %27
         %30 = OpIMul %6 %28 %29
         %31 = OpAccessChain %7 %23 %52
               OpStore %31 %30
               OpBranch %13
         %13 = OpLabel
         %34 = OpIAdd %6 %52 %33
               OpStore %8 %34
               OpBranch %10
         %12 = OpLabel
               OpStore %35 %9
               OpBranch %36
         %36 = OpLabel
         %53 = OpPhi %6 %9 %12 %51 %39
               OpLoopMerge %38 %39 None
               OpBranch %40
         %40 = OpLabel
         %42 = OpSLessThan %17 %53 %16
               OpBranchConditional %42 %37 %38
         %37 = OpLabel
         %46 = OpAccessChain %7 %25 %53
         %47 = OpLoad %6 %46
         %48 = OpIAdd %6 %47 %29
         %49 = OpAccessChain %7 %43 %53
               OpStore %49 %48
               OpBranch %39
         %39 = OpLabel
         %51 = OpIAdd %6 %53 %33
               OpStore %35 %51
               OpBranch %36
         %38 = OpLabel
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
  EXPECT_TRUE(fusion.IsLegal());

  fusion.Fuse();

  std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
)";

  Match(checks, context.get());
  auto& ld_final = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld_final.NumLoops(), 1u);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  // No loop-carried dependences, legal
  for (int i = 0; i < 10; i++) {
    a[i] = a[i]*2;
  }
  for (int i = 0; i < 10; i++) {
    b[i] = a[i]+2;
  }
}

*/
TEST_F(FusionLegalTest, NoLoopCarriedDependences) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %34 "i"
               OpName %42 "b"
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
         %28 = OpConstant %6 2
         %32 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %51 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %51 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpAccessChain %7 %23 %51
         %27 = OpLoad %6 %26
         %29 = OpIMul %6 %27 %28
         %30 = OpAccessChain %7 %23 %51
               OpStore %30 %29
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %51 %32
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %52 = OpPhi %6 %9 %12 %50 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %52 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpAccessChain %7 %23 %52
         %46 = OpLoad %6 %45
         %47 = OpIAdd %6 %46 %28
         %48 = OpAccessChain %7 %42 %52
               OpStore %48 %47
               OpBranch %38
         %38 = OpLabel
         %50 = OpIAdd %6 %52 %32
               OpStore %34 %50
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

  LoopFusion fusion(context.get(), loops[0], loops[1]);

  EXPECT_TRUE(fusion.AreCompatible());
  EXPECT_TRUE(fusion.IsLegal());

  fusion.Fuse();

  std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
)";

  Match(checks, context.get());
  auto& ld_final = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld_final.NumLoops(), 1u);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Parallelism inhibiting, but legal.
  for (int i = 0; i < 10; i++) {
    a[i] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    c[i] = a[i] + c[i-1];
  }
}

*/
TEST_F(FusionLegalTest, ExistingLoopCarriedDependence) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %25 "b"
               OpName %34 "i"
               OpName %42 "c"
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
         %29 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %25 = OpVariable %22 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %55 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %55 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpAccessChain %7 %25 %55
         %28 = OpLoad %6 %27
         %30 = OpIAdd %6 %28 %29
         %31 = OpAccessChain %7 %23 %55
               OpStore %31 %30
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %55 %29
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %56 = OpPhi %6 %9 %12 %54 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %56 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpAccessChain %7 %23 %56
         %46 = OpLoad %6 %45
         %48 = OpISub %6 %56 %29
         %49 = OpAccessChain %7 %42 %48
         %50 = OpLoad %6 %49
         %51 = OpIAdd %6 %46 %50
         %52 = OpAccessChain %7 %42 %56
               OpStore %52 %51
               OpBranch %38
         %38 = OpLabel
         %54 = OpIAdd %6 %56 %29
               OpStore %34 %54
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

  LoopFusion fusion(context.get(), loops[0], loops[1]);

  EXPECT_TRUE(fusion.AreCompatible());
  EXPECT_TRUE(fusion.IsLegal());

  fusion.Fuse();

  std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[I_1:%\w+]] = OpISub {{%\w+}} [[PHI]] {{%\w+}}
CHECK-NEXT: [[LOAD_2:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[I_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_2]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
)";

  Match(checks, context.get());
  auto& ld_final = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld_final.NumLoops(), 1u);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Creates a loop-carried dependence, but negative, so legal
  for (int i = 0; i < 10; i++) {
    a[i+1] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    c[i] = a[i] + 2;
  }
}

*/
TEST_F(FusionLegalTest, NegativeDistanceCreatedRAW) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %27 "b"
               OpName %35 "i"
               OpName %43 "c"
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
         %25 = OpConstant %6 1
         %48 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %27 = OpVariable %22 Function
         %35 = OpVariable %7 Function
         %43 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %53 = OpPhi %6 %9 %5 %34 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %53 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpIAdd %6 %53 %25
         %29 = OpAccessChain %7 %27 %53
         %30 = OpLoad %6 %29
         %31 = OpIAdd %6 %30 %25
         %32 = OpAccessChain %7 %23 %26
               OpStore %32 %31
               OpBranch %13
         %13 = OpLabel
         %34 = OpIAdd %6 %53 %25
               OpStore %8 %34
               OpBranch %10
         %12 = OpLabel
               OpStore %35 %9
               OpBranch %36
         %36 = OpLabel
         %54 = OpPhi %6 %9 %12 %52 %39
               OpLoopMerge %38 %39 None
               OpBranch %40
         %40 = OpLabel
         %42 = OpSLessThan %17 %54 %16
               OpBranchConditional %42 %37 %38
         %37 = OpLabel
         %46 = OpAccessChain %7 %23 %54
         %47 = OpLoad %6 %46
         %49 = OpIAdd %6 %47 %48
         %50 = OpAccessChain %7 %43 %54
               OpStore %50 %49
               OpBranch %39
         %39 = OpLabel
         %52 = OpIAdd %6 %54 %25
               OpStore %35 %52
               OpBranch %36
         %38 = OpLabel
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[0], loops[1]);

    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();

    std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[I_1:%\w+]] = OpIAdd {{%\w+}} [[PHI]] {{%\w+}}
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[I_1]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
    )";

    Match(checks, context.get());
  }

  {
    auto& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 1u);
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Legal
  for (int i = 0; i < 10; i++) {
    a[i+1] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    c[i] = a[i+1] + 2;
  }
}

*/
TEST_F(FusionLegalTest, NoLoopCarriedDependencesAdjustedIndex) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %27 "b"
               OpName %35 "i"
               OpName %43 "c"
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
         %25 = OpConstant %6 1
         %49 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %27 = OpVariable %22 Function
         %35 = OpVariable %7 Function
         %43 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %54 = OpPhi %6 %9 %5 %34 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %54 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpIAdd %6 %54 %25
         %29 = OpAccessChain %7 %27 %54
         %30 = OpLoad %6 %29
         %31 = OpIAdd %6 %30 %25
         %32 = OpAccessChain %7 %23 %26
               OpStore %32 %31
               OpBranch %13
         %13 = OpLabel
         %34 = OpIAdd %6 %54 %25
               OpStore %8 %34
               OpBranch %10
         %12 = OpLabel
               OpStore %35 %9
               OpBranch %36
         %36 = OpLabel
         %55 = OpPhi %6 %9 %12 %53 %39
               OpLoopMerge %38 %39 None
               OpBranch %40
         %40 = OpLabel
         %42 = OpSLessThan %17 %55 %16
               OpBranchConditional %42 %37 %38
         %37 = OpLabel
         %46 = OpIAdd %6 %55 %25
         %47 = OpAccessChain %7 %23 %46
         %48 = OpLoad %6 %47
         %50 = OpIAdd %6 %48 %49
         %51 = OpAccessChain %7 %43 %55
               OpStore %51 %50
               OpBranch %39
         %39 = OpLabel
         %53 = OpIAdd %6 %55 %25
               OpStore %35 %53
               OpBranch %36
         %38 = OpLabel
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
  EXPECT_TRUE(fusion.IsLegal());

  fusion.Fuse();

  std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[I_1:%\w+]] = OpIAdd {{%\w+}} [[PHI]] {{%\w+}}
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[I_1]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[I_1:%\w+]] = OpIAdd {{%\w+}} [[PHI]] {{%\w+}}
CHECK-NEXT: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[I_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
)";

  Match(checks, context.get());
  auto& ld_final = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld_final.NumLoops(), 1u);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Legal, independent locations in |a|, SIV
  for (int i = 0; i < 10; i++) {
    a[2*i+1] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    c[i] = a[2*i] + 2;
  }
}

*/
TEST_F(FusionLegalTest, IndependentSIV) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %29 "b"
               OpName %37 "i"
               OpName %45 "c"
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
         %24 = OpConstant %6 2
         %27 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %29 = OpVariable %22 Function
         %37 = OpVariable %7 Function
         %45 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %55 = OpPhi %6 %9 %5 %36 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %55 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpIMul %6 %24 %55
         %28 = OpIAdd %6 %26 %27
         %31 = OpAccessChain %7 %29 %55
         %32 = OpLoad %6 %31
         %33 = OpIAdd %6 %32 %27
         %34 = OpAccessChain %7 %23 %28
               OpStore %34 %33
               OpBranch %13
         %13 = OpLabel
         %36 = OpIAdd %6 %55 %27
               OpStore %8 %36
               OpBranch %10
         %12 = OpLabel
               OpStore %37 %9
               OpBranch %38
         %38 = OpLabel
         %56 = OpPhi %6 %9 %12 %54 %41
               OpLoopMerge %40 %41 None
               OpBranch %42
         %42 = OpLabel
         %44 = OpSLessThan %17 %56 %16
               OpBranchConditional %44 %39 %40
         %39 = OpLabel
         %48 = OpIMul %6 %24 %56
         %49 = OpAccessChain %7 %23 %48
         %50 = OpLoad %6 %49
         %51 = OpIAdd %6 %50 %24
         %52 = OpAccessChain %7 %45 %56
               OpStore %52 %51
               OpBranch %41
         %41 = OpLabel
         %54 = OpIAdd %6 %56 %27
               OpStore %37 %54
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

  LoopFusion fusion(context.get(), loops[0], loops[1]);

  EXPECT_TRUE(fusion.AreCompatible());
  EXPECT_TRUE(fusion.IsLegal());

  fusion.Fuse();

  std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[I_2:%\w+]] = OpIMul {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: [[I_2_1:%\w+]] = OpIAdd {{%\w+}} [[I_2]] {{%\w+}}
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[I_2_1]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[I_2:%\w+]] = OpIMul {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[I_2]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
)";

  Match(checks, context.get());
  auto& ld_final = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld_final.NumLoops(), 1u);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Legal, independent locations in |a|, ZIV
  for (int i = 0; i < 10; i++) {
    a[1] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    c[i] = a[9] + 2;
  }
}

*/
TEST_F(FusionLegalTest, IndependentZIV) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %25 "b"
               OpName %33 "i"
               OpName %41 "c"
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
         %24 = OpConstant %6 1
         %43 = OpConstant %6 9
         %46 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %25 = OpVariable %22 Function
         %33 = OpVariable %7 Function
         %41 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %51 = OpPhi %6 %9 %5 %32 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %51 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpAccessChain %7 %25 %51
         %28 = OpLoad %6 %27
         %29 = OpIAdd %6 %28 %24
         %30 = OpAccessChain %7 %23 %24
               OpStore %30 %29
               OpBranch %13
         %13 = OpLabel
         %32 = OpIAdd %6 %51 %24
               OpStore %8 %32
               OpBranch %10
         %12 = OpLabel
               OpStore %33 %9
               OpBranch %34
         %34 = OpLabel
         %52 = OpPhi %6 %9 %12 %50 %37
               OpLoopMerge %36 %37 None
               OpBranch %38
         %38 = OpLabel
         %40 = OpSLessThan %17 %52 %16
               OpBranchConditional %40 %35 %36
         %35 = OpLabel
         %44 = OpAccessChain %7 %23 %43
         %45 = OpLoad %6 %44
         %47 = OpIAdd %6 %45 %46
         %48 = OpAccessChain %7 %41 %52
               OpStore %48 %47
               OpBranch %37
         %37 = OpLabel
         %50 = OpIAdd %6 %52 %24
               OpStore %33 %50
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
  EXPECT_TRUE(fusion.IsLegal());

  fusion.Fuse();

  std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK-NOT: OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK: OpStore
CHECK-NOT: OpPhi
CHECK-NOT: OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK: OpLoad
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
)";

  Match(checks, context.get());
  auto& ld_final = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld_final.NumLoops(), 1u);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[20] a;
  int[10] b;
  int[10] c;
  // Legal, non-overlapping sections in |a|
  for (int i = 0; i < 10; i++) {
    a[i] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    c[i] = a[i+10] + 2;
  }
}

*/
TEST_F(FusionLegalTest, NonOverlappingAccesses) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %28 "b"
               OpName %37 "i"
               OpName %45 "c"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %19 = OpTypeInt 32 0
         %20 = OpConstant %19 20
         %21 = OpTypeArray %6 %20
         %22 = OpTypePointer Function %21
         %25 = OpConstant %19 10
         %26 = OpTypeArray %6 %25
         %27 = OpTypePointer Function %26
         %32 = OpConstant %6 1
         %51 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %28 = OpVariable %27 Function
         %37 = OpVariable %7 Function
         %45 = OpVariable %27 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %56 = OpPhi %6 %9 %5 %36 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %56 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %30 = OpAccessChain %7 %28 %56
         %31 = OpLoad %6 %30
         %33 = OpIAdd %6 %31 %32
         %34 = OpAccessChain %7 %23 %56
               OpStore %34 %33
               OpBranch %13
         %13 = OpLabel
         %36 = OpIAdd %6 %56 %32
               OpStore %8 %36
               OpBranch %10
         %12 = OpLabel
               OpStore %37 %9
               OpBranch %38
         %38 = OpLabel
         %57 = OpPhi %6 %9 %12 %55 %41
               OpLoopMerge %40 %41 None
               OpBranch %42
         %42 = OpLabel
         %44 = OpSLessThan %17 %57 %16
               OpBranchConditional %44 %39 %40
         %39 = OpLabel
         %48 = OpIAdd %6 %57 %16
         %49 = OpAccessChain %7 %23 %48
         %50 = OpLoad %6 %49
         %52 = OpIAdd %6 %50 %51
         %53 = OpAccessChain %7 %45 %57
               OpStore %53 %52
               OpBranch %41
         %41 = OpLabel
         %55 = OpIAdd %6 %57 %32
               OpStore %37 %55
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

  LoopFusion fusion(context.get(), loops[0], loops[1]);

  EXPECT_TRUE(fusion.AreCompatible());
  EXPECT_TRUE(fusion.IsLegal());

  fusion.Fuse();

  std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NOT: OpPhi
CHECK: [[I_10:%\w+]] = OpIAdd {{%\w+}} [[PHI]] {{%\w+}}
CHECK-NEXT: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[I_10]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
)";

  Match(checks, context.get());

  auto& ld_final = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld_final.NumLoops(), 1u);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Legal, 3 adjacent loops
  for (int i = 0; i < 10; i++) {
    a[i] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    c[i] = a[i] + 2;
  }
  for (int i = 0; i < 10; i++) {
    b[i] = c[i] + 10;
  }
}

*/
TEST_F(FusionLegalTest, AdjacentLoops) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %25 "b"
               OpName %34 "i"
               OpName %42 "c"
               OpName %52 "i"
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
         %29 = OpConstant %6 1
         %47 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %25 = OpVariable %22 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %22 Function
         %52 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %68 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %68 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpAccessChain %7 %25 %68
         %28 = OpLoad %6 %27
         %30 = OpIAdd %6 %28 %29
         %31 = OpAccessChain %7 %23 %68
               OpStore %31 %30
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %68 %29
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %69 = OpPhi %6 %9 %12 %51 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %69 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpAccessChain %7 %23 %69
         %46 = OpLoad %6 %45
         %48 = OpIAdd %6 %46 %47
         %49 = OpAccessChain %7 %42 %69
               OpStore %49 %48
               OpBranch %38
         %38 = OpLabel
         %51 = OpIAdd %6 %69 %29
               OpStore %34 %51
               OpBranch %35
         %37 = OpLabel
               OpStore %52 %9
               OpBranch %53
         %53 = OpLabel
         %70 = OpPhi %6 %9 %37 %67 %56
               OpLoopMerge %55 %56 None
               OpBranch %57
         %57 = OpLabel
         %59 = OpSLessThan %17 %70 %16
               OpBranchConditional %59 %54 %55
         %54 = OpLabel
         %62 = OpAccessChain %7 %42 %70
         %63 = OpLoad %6 %62
         %64 = OpIAdd %6 %63 %16
         %65 = OpAccessChain %7 %25 %70
               OpStore %65 %64
               OpBranch %56
         %56 = OpLabel
         %67 = OpIAdd %6 %70 %29
               OpStore %52 %67
               OpBranch %53
         %55 = OpLabel
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 3u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[1], loops[2]);

    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();
  }

  std::string checks = R"(
CHECK: [[PHI_0:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK: [[PHI_1:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_1]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_2:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_2]]
CHECK: [[STORE_2:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_2]]
    )";

  Match(checks, context.get());

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[0], loops[1]);

    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();
  }

  std::string checks_ = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_2:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_2]]
CHECK: [[STORE_2:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_2]]
    )";

  Match(checks_, context.get());

  auto& ld_final = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld_final.NumLoops(), 1u);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10][10] a;
  int[10][10] b;
  int[10][10] c;
  // Legal inner loop fusion
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      c[i][j] = a[i][j] + 2;
    }
    for (int j = 0; j < 10; j++) {
      b[i][j] = c[i][j] + 10;
    }
  }
}

*/
TEST_F(FusionLegalTest, InnerLoopFusion) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %19 "j"
               OpName %32 "c"
               OpName %35 "a"
               OpName %46 "j"
               OpName %54 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %27 = OpTypeInt 32 0
         %28 = OpConstant %27 10
         %29 = OpTypeArray %6 %28
         %30 = OpTypeArray %29 %28
         %31 = OpTypePointer Function %30
         %40 = OpConstant %6 2
         %44 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %31 Function
         %35 = OpVariable %31 Function
         %46 = OpVariable %7 Function
         %54 = OpVariable %31 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %67 = OpPhi %6 %9 %5 %66 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %67 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
         %68 = OpPhi %6 %9 %11 %45 %23
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %26 = OpSLessThan %17 %68 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
         %38 = OpAccessChain %7 %35 %67 %68
         %39 = OpLoad %6 %38
         %41 = OpIAdd %6 %39 %40
         %42 = OpAccessChain %7 %32 %67 %68
               OpStore %42 %41
               OpBranch %23
         %23 = OpLabel
         %45 = OpIAdd %6 %68 %44
               OpStore %19 %45
               OpBranch %20
         %22 = OpLabel
               OpStore %46 %9
               OpBranch %47
         %47 = OpLabel
         %69 = OpPhi %6 %9 %22 %64 %50
               OpLoopMerge %49 %50 None
               OpBranch %51
         %51 = OpLabel
         %53 = OpSLessThan %17 %69 %16
               OpBranchConditional %53 %48 %49
         %48 = OpLabel
         %59 = OpAccessChain %7 %32 %67 %69
         %60 = OpLoad %6 %59
         %61 = OpIAdd %6 %60 %16
         %62 = OpAccessChain %7 %54 %67 %69
               OpStore %62 %61
               OpBranch %50
         %50 = OpLabel
         %64 = OpIAdd %6 %69 %44
               OpStore %46 %64
               OpBranch %47
         %49 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %66 = OpIAdd %6 %67 %44
               OpStore %8 %66
               OpBranch %10
         %12 = OpLabel
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

  {
    LoopFusion fusion(context.get(), loop_0, loop_1);
    EXPECT_FALSE(fusion.AreCompatible());
  }

  {
    LoopFusion fusion(context.get(), loop_0, loop_2);
    EXPECT_FALSE(fusion.AreCompatible());
  }

  {
    LoopFusion fusion(context.get(), loop_1, loop_2);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();
  }

  std::string checks = R"(
CHECK: [[PHI_0:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[PHI_1:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_1]]
    )";

  Match(checks, context.get());

  auto& ld_final = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld_final.NumLoops(), 2u);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 12
#version 440 core
void main() {
  int[10][10] a;
  int[10][10] b;
  int[10][10] c;
  // Legal both
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      c[i][j] = a[i][j] + 2;
    }
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      b[i][j] = c[i][j] + 10;
    }
  }
}

*/
TEST_F(FusionLegalTest, OuterAndInnerLoop) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %19 "j"
               OpName %32 "c"
               OpName %35 "a"
               OpName %48 "i"
               OpName %56 "j"
               OpName %64 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %27 = OpTypeInt 32 0
         %28 = OpConstant %27 10
         %29 = OpTypeArray %6 %28
         %30 = OpTypeArray %29 %28
         %31 = OpTypePointer Function %30
         %40 = OpConstant %6 2
         %44 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %31 Function
         %35 = OpVariable %31 Function
         %48 = OpVariable %7 Function
         %56 = OpVariable %7 Function
         %64 = OpVariable %31 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %77 = OpPhi %6 %9 %5 %47 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %77 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
         %81 = OpPhi %6 %9 %11 %45 %23
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %26 = OpSLessThan %17 %81 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
         %38 = OpAccessChain %7 %35 %77 %81
         %39 = OpLoad %6 %38
         %41 = OpIAdd %6 %39 %40
         %42 = OpAccessChain %7 %32 %77 %81
               OpStore %42 %41
               OpBranch %23
         %23 = OpLabel
         %45 = OpIAdd %6 %81 %44
               OpStore %19 %45
               OpBranch %20
         %22 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %47 = OpIAdd %6 %77 %44
               OpStore %8 %47
               OpBranch %10
         %12 = OpLabel
               OpStore %48 %9
               OpBranch %49
         %49 = OpLabel
         %78 = OpPhi %6 %9 %12 %76 %52
               OpLoopMerge %51 %52 None
               OpBranch %53
         %53 = OpLabel
         %55 = OpSLessThan %17 %78 %16
               OpBranchConditional %55 %50 %51
         %50 = OpLabel
               OpStore %56 %9
               OpBranch %57
         %57 = OpLabel
         %79 = OpPhi %6 %9 %50 %74 %60
               OpLoopMerge %59 %60 None
               OpBranch %61
         %61 = OpLabel
         %63 = OpSLessThan %17 %79 %16
               OpBranchConditional %63 %58 %59
         %58 = OpLabel
         %69 = OpAccessChain %7 %32 %78 %79
         %70 = OpLoad %6 %69
         %71 = OpIAdd %6 %70 %16
         %72 = OpAccessChain %7 %64 %78 %79
               OpStore %72 %71
               OpBranch %60
         %60 = OpLabel
         %74 = OpIAdd %6 %79 %44
               OpStore %56 %74
               OpBranch %57
         %59 = OpLabel
               OpBranch %52
         %52 = OpLabel
         %76 = OpIAdd %6 %78 %44
               OpStore %48 %76
               OpBranch %49
         %51 = OpLabel
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 4u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    auto loop_0 = loops[0];
    auto loop_1 = loops[1];
    auto loop_2 = loops[2];
    auto loop_3 = loops[3];

    {
      LoopFusion fusion(context.get(), loop_0, loop_1);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_1, loop_2);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_2, loop_3);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_1, loop_3);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_0, loop_2);
      EXPECT_TRUE(fusion.AreCompatible());
      EXPECT_TRUE(fusion.IsLegal());
      fusion.Fuse();
    }

    std::string checks = R"(
CHECK: [[PHI_0:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[PHI_1:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK: [[PHI_2:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_2]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_2]]
CHECK-NEXT: OpStore [[STORE_1]]
    )";

    Match(checks, context.get());
  }

  {
    auto& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 3u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();
    auto loop_0 = loops[0];
    auto loop_1 = loops[1];
    auto loop_2 = loops[2];

    {
      LoopFusion fusion(context.get(), loop_0, loop_1);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_0, loop_2);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_1, loop_2);
      EXPECT_TRUE(fusion.AreCompatible());
      EXPECT_TRUE(fusion.IsLegal());
      fusion.Fuse();
    }

    std::string checks = R"(
CHECK: [[PHI_0:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[PHI_1:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_1]]
    )";

    Match(checks, context.get());
  }

  {
    auto& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10][10] a;
  int[10][10] b;
  int[10][10] c;
  // Legal both, more complex
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      if (i % 2 == 0 && j % 2 == 0) {
        c[i][j] = a[i][j] + 2;
      }
    }
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      b[i][j] = c[i][j] + 10;
    }
  }
}

*/
TEST_F(FusionLegalTest, OuterAndInnerLoopMoreComplex) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %19 "j"
               OpName %44 "c"
               OpName %47 "a"
               OpName %59 "i"
               OpName %67 "j"
               OpName %75 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %28 = OpConstant %6 2
         %39 = OpTypeInt 32 0
         %40 = OpConstant %39 10
         %41 = OpTypeArray %6 %40
         %42 = OpTypeArray %41 %40
         %43 = OpTypePointer Function %42
         %55 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %44 = OpVariable %43 Function
         %47 = OpVariable %43 Function
         %59 = OpVariable %7 Function
         %67 = OpVariable %7 Function
         %75 = OpVariable %43 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %88 = OpPhi %6 %9 %5 %58 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %88 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
         %92 = OpPhi %6 %9 %11 %56 %23
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %26 = OpSLessThan %17 %92 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
         %29 = OpSMod %6 %88 %28
         %30 = OpIEqual %17 %29 %9
               OpSelectionMerge %32 None
               OpBranchConditional %30 %31 %32
         %31 = OpLabel
         %34 = OpSMod %6 %92 %28
         %35 = OpIEqual %17 %34 %9
               OpBranch %32
         %32 = OpLabel
         %36 = OpPhi %17 %30 %21 %35 %31
               OpSelectionMerge %38 None
               OpBranchConditional %36 %37 %38
         %37 = OpLabel
         %50 = OpAccessChain %7 %47 %88 %92
         %51 = OpLoad %6 %50
         %52 = OpIAdd %6 %51 %28
         %53 = OpAccessChain %7 %44 %88 %92
               OpStore %53 %52
               OpBranch %38
         %38 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %56 = OpIAdd %6 %92 %55
               OpStore %19 %56
               OpBranch %20
         %22 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %58 = OpIAdd %6 %88 %55
               OpStore %8 %58
               OpBranch %10
         %12 = OpLabel
               OpStore %59 %9
               OpBranch %60
         %60 = OpLabel
         %89 = OpPhi %6 %9 %12 %87 %63
               OpLoopMerge %62 %63 None
               OpBranch %64
         %64 = OpLabel
         %66 = OpSLessThan %17 %89 %16
               OpBranchConditional %66 %61 %62
         %61 = OpLabel
               OpStore %67 %9
               OpBranch %68
         %68 = OpLabel
         %90 = OpPhi %6 %9 %61 %85 %71
               OpLoopMerge %70 %71 None
               OpBranch %72
         %72 = OpLabel
         %74 = OpSLessThan %17 %90 %16
               OpBranchConditional %74 %69 %70
         %69 = OpLabel
         %80 = OpAccessChain %7 %44 %89 %90
         %81 = OpLoad %6 %80
         %82 = OpIAdd %6 %81 %16
         %83 = OpAccessChain %7 %75 %89 %90
               OpStore %83 %82
               OpBranch %71
         %71 = OpLabel
         %85 = OpIAdd %6 %90 %55
               OpStore %67 %85
               OpBranch %68
         %70 = OpLabel
               OpBranch %63
         %63 = OpLabel
         %87 = OpIAdd %6 %89 %55
               OpStore %59 %87
               OpBranch %60
         %62 = OpLabel
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 4u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    auto loop_0 = loops[0];
    auto loop_1 = loops[1];
    auto loop_2 = loops[2];
    auto loop_3 = loops[3];

    {
      LoopFusion fusion(context.get(), loop_0, loop_1);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_1, loop_2);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_2, loop_3);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_1, loop_3);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_0, loop_2);
      EXPECT_TRUE(fusion.AreCompatible());
      EXPECT_TRUE(fusion.IsLegal());
      fusion.Fuse();
    }

    std::string checks = R"(
CHECK: [[PHI_0:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[PHI_1:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: OpPhi
CHECK-NEXT: OpSelectionMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK: [[PHI_2:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_2]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_2]]
CHECK-NEXT: OpStore [[STORE_1]]
    )";

    Match(checks, context.get());
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 3u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    auto loop_0 = loops[0];
    auto loop_1 = loops[1];
    auto loop_2 = loops[2];

    {
      LoopFusion fusion(context.get(), loop_0, loop_1);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_0, loop_2);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_1, loop_2);
      EXPECT_TRUE(fusion.AreCompatible());
      EXPECT_TRUE(fusion.IsLegal());
      fusion.Fuse();
    }

    std::string checks = R"(
CHECK: [[PHI_0:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[PHI_1:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: OpPhi
CHECK-NEXT: OpSelectionMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_1]]
    )";

    Match(checks, context.get());
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10][10] a;
  int[10][10] b;
  int[10][10] c;
  // Outer would have been illegal to fuse, but since written
  // like this, inner loop fusion is legal.
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      c[i][j] = a[i][j] + 2;
    }
    for (int j = 0; j < 10; j++) {
      b[i][j] = c[i+1][j] + 10;
    }
  }
}

*/
TEST_F(FusionLegalTest, InnerWithExistingDependenceOnOuter) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %19 "j"
               OpName %32 "c"
               OpName %35 "a"
               OpName %46 "j"
               OpName %54 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %27 = OpTypeInt 32 0
         %28 = OpConstant %27 10
         %29 = OpTypeArray %6 %28
         %30 = OpTypeArray %29 %28
         %31 = OpTypePointer Function %30
         %40 = OpConstant %6 2
         %44 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %31 Function
         %35 = OpVariable %31 Function
         %46 = OpVariable %7 Function
         %54 = OpVariable %31 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %68 = OpPhi %6 %9 %5 %67 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %68 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
         %69 = OpPhi %6 %9 %11 %45 %23
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %26 = OpSLessThan %17 %69 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
         %38 = OpAccessChain %7 %35 %68 %69
         %39 = OpLoad %6 %38
         %41 = OpIAdd %6 %39 %40
         %42 = OpAccessChain %7 %32 %68 %69
               OpStore %42 %41
               OpBranch %23
         %23 = OpLabel
         %45 = OpIAdd %6 %69 %44
               OpStore %19 %45
               OpBranch %20
         %22 = OpLabel
               OpStore %46 %9
               OpBranch %47
         %47 = OpLabel
         %70 = OpPhi %6 %9 %22 %65 %50
               OpLoopMerge %49 %50 None
               OpBranch %51
         %51 = OpLabel
         %53 = OpSLessThan %17 %70 %16
               OpBranchConditional %53 %48 %49
         %48 = OpLabel
         %58 = OpIAdd %6 %68 %44
         %60 = OpAccessChain %7 %32 %58 %70
         %61 = OpLoad %6 %60
         %62 = OpIAdd %6 %61 %16
         %63 = OpAccessChain %7 %54 %68 %70
               OpStore %63 %62
               OpBranch %50
         %50 = OpLabel
         %65 = OpIAdd %6 %70 %44
               OpStore %46 %65
               OpBranch %47
         %49 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %67 = OpIAdd %6 %68 %44
               OpStore %8 %67
               OpBranch %10
         %12 = OpLabel
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 3u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    auto loop_0 = loops[0];
    auto loop_1 = loops[1];
    auto loop_2 = loops[2];

    {
      LoopFusion fusion(context.get(), loop_0, loop_1);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_0, loop_2);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_1, loop_2);
      EXPECT_TRUE(fusion.AreCompatible());
      EXPECT_TRUE(fusion.IsLegal());

      fusion.Fuse();
    }
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    std::string checks = R"(
CHECK: [[PHI_0:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[PHI_1:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[I_1:%\w+]] = OpIAdd {{%\w+}} [[PHI_0]] {{%\w+}}
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[I_1]] [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_1]]
    )";

    Match(checks, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // One dimensional arrays. Legal, outer dist 0, inner independent.
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      c[i] = a[j] + 2;
    }
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      b[j] = c[i] + 10;
    }
  }
}

*/
TEST_F(FusionLegalTest, OuterAndInnerLoopOneDimArrays) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %19 "j"
               OpName %31 "c"
               OpName %33 "a"
               OpName %45 "i"
               OpName %53 "j"
               OpName %61 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %27 = OpTypeInt 32 0
         %28 = OpConstant %27 10
         %29 = OpTypeArray %6 %28
         %30 = OpTypePointer Function %29
         %37 = OpConstant %6 2
         %41 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %31 = OpVariable %30 Function
         %33 = OpVariable %30 Function
         %45 = OpVariable %7 Function
         %53 = OpVariable %7 Function
         %61 = OpVariable %30 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %72 = OpPhi %6 %9 %5 %44 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %72 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
         %76 = OpPhi %6 %9 %11 %42 %23
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %26 = OpSLessThan %17 %76 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
         %35 = OpAccessChain %7 %33 %76
         %36 = OpLoad %6 %35
         %38 = OpIAdd %6 %36 %37
         %39 = OpAccessChain %7 %31 %72
               OpStore %39 %38
               OpBranch %23
         %23 = OpLabel
         %42 = OpIAdd %6 %76 %41
               OpStore %19 %42
               OpBranch %20
         %22 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %44 = OpIAdd %6 %72 %41
               OpStore %8 %44
               OpBranch %10
         %12 = OpLabel
               OpStore %45 %9
               OpBranch %46
         %46 = OpLabel
         %73 = OpPhi %6 %9 %12 %71 %49
               OpLoopMerge %48 %49 None
               OpBranch %50
         %50 = OpLabel
         %52 = OpSLessThan %17 %73 %16
               OpBranchConditional %52 %47 %48
         %47 = OpLabel
               OpStore %53 %9
               OpBranch %54
         %54 = OpLabel
         %74 = OpPhi %6 %9 %47 %69 %57
               OpLoopMerge %56 %57 None
               OpBranch %58
         %58 = OpLabel
         %60 = OpSLessThan %17 %74 %16
               OpBranchConditional %60 %55 %56
         %55 = OpLabel
         %64 = OpAccessChain %7 %31 %73
         %65 = OpLoad %6 %64
         %66 = OpIAdd %6 %65 %16
         %67 = OpAccessChain %7 %61 %74
               OpStore %67 %66
               OpBranch %57
         %57 = OpLabel
         %69 = OpIAdd %6 %74 %41
               OpStore %53 %69
               OpBranch %54
         %56 = OpLabel
               OpBranch %49
         %49 = OpLabel
         %71 = OpIAdd %6 %73 %41
               OpStore %45 %71
               OpBranch %46
         %48 = OpLabel
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 4u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    auto loop_0 = loops[0];
    auto loop_1 = loops[1];
    auto loop_2 = loops[2];
    auto loop_3 = loops[3];

    {
      LoopFusion fusion(context.get(), loop_0, loop_1);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_1, loop_2);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_2, loop_3);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_0, loop_2);
      EXPECT_TRUE(fusion.AreCompatible());
      EXPECT_TRUE(fusion.IsLegal());
      fusion.Fuse();
    }

    std::string checks = R"(
CHECK: [[PHI_0:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[PHI_1:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK: [[PHI_2:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_2]]
CHECK-NEXT: OpStore [[STORE_1]]
    )";

    Match(checks, context.get());
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 3u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    auto loop_0 = loops[0];
    auto loop_1 = loops[1];
    auto loop_2 = loops[2];

    {
      LoopFusion fusion(context.get(), loop_0, loop_1);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_0, loop_2);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      LoopFusion fusion(context.get(), loop_1, loop_2);
      EXPECT_TRUE(fusion.AreCompatible());
      EXPECT_TRUE(fusion.IsLegal());

      fusion.Fuse();
    }

    std::string checks = R"(
CHECK: [[PHI_0:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[PHI_1:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_1]]
    )";

    Match(checks, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Legal, creates a loop-carried dependence, but has negative distance
  for (int i = 0; i < 10; i++) {
    c[i] = a[i+1] + 1;
  }
  for (int i = 0; i < 10; i++) {
    a[i] = c[i] + 2;
  }
}

*/
TEST_F(FusionLegalTest, NegativeDistanceCreatedWAR) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "c"
               OpName %25 "a"
               OpName %35 "i"
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
         %27 = OpConstant %6 1
         %47 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %25 = OpVariable %22 Function
         %35 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %52 = OpPhi %6 %9 %5 %34 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %52 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %28 = OpIAdd %6 %52 %27
         %29 = OpAccessChain %7 %25 %28
         %30 = OpLoad %6 %29
         %31 = OpIAdd %6 %30 %27
         %32 = OpAccessChain %7 %23 %52
               OpStore %32 %31
               OpBranch %13
         %13 = OpLabel
         %34 = OpIAdd %6 %52 %27
               OpStore %8 %34
               OpBranch %10
         %12 = OpLabel
               OpStore %35 %9
               OpBranch %36
         %36 = OpLabel
         %53 = OpPhi %6 %9 %12 %51 %39
               OpLoopMerge %38 %39 None
               OpBranch %40
         %40 = OpLabel
         %42 = OpSLessThan %17 %53 %16
               OpBranchConditional %42 %37 %38
         %37 = OpLabel
         %45 = OpAccessChain %7 %23 %53
         %46 = OpLoad %6 %45
         %48 = OpIAdd %6 %46 %47
         %49 = OpAccessChain %7 %25 %53
               OpStore %49 %48
               OpBranch %39
         %39 = OpLabel
         %51 = OpIAdd %6 %53 %27
               OpStore %35 %51
               OpBranch %36
         %38 = OpLabel
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();

    std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK: [[I_1:%\w+]] = OpIAdd {{%\w+}} [[PHI]] {{%\w+}}
CHECK-NEXT: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[I_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
    )";

    Match(checks, context.get());
  }

  {
    auto& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 1u);
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Legal, creates a loop-carried dependence, but has negative distance
  for (int i = 0; i < 10; i++) {
    a[i+1] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    a[i] = c[i+1] + 2;
  }
}

*/
TEST_F(FusionLegalTest, NegativeDistanceCreatedWAW) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %27 "b"
               OpName %35 "i"
               OpName %44 "c"
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
         %25 = OpConstant %6 1
         %49 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %27 = OpVariable %22 Function
         %35 = OpVariable %7 Function
         %44 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %54 = OpPhi %6 %9 %5 %34 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %54 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpIAdd %6 %54 %25
         %29 = OpAccessChain %7 %27 %54
         %30 = OpLoad %6 %29
         %31 = OpIAdd %6 %30 %25
         %32 = OpAccessChain %7 %23 %26
               OpStore %32 %31
               OpBranch %13
         %13 = OpLabel
         %34 = OpIAdd %6 %54 %25
               OpStore %8 %34
               OpBranch %10
         %12 = OpLabel
               OpStore %35 %9
               OpBranch %36
         %36 = OpLabel
         %55 = OpPhi %6 %9 %12 %53 %39
               OpLoopMerge %38 %39 None
               OpBranch %40
         %40 = OpLabel
         %42 = OpSLessThan %17 %55 %16
               OpBranchConditional %42 %37 %38
         %37 = OpLabel
         %46 = OpIAdd %6 %55 %25
         %47 = OpAccessChain %7 %44 %46
         %48 = OpLoad %6 %47
         %50 = OpIAdd %6 %48 %49
         %51 = OpAccessChain %7 %23 %55
               OpStore %51 %50
               OpBranch %39
         %39 = OpLabel
         %53 = OpIAdd %6 %55 %25
               OpStore %35 %53
               OpBranch %36
         %38 = OpLabel
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 1u);

    std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[I_1:%\w+]] = OpIAdd {{%\w+}} [[PHI]] {{%\w+}}
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[I_1]]
CHECK-NEXT: OpStore
CHECK-NOT: OpPhi
CHECK: [[I_1:%\w+]] = OpIAdd {{%\w+}} [[PHI]] {{%\w+}}
CHECK-NEXT: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[I_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
    )";

    Match(checks, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Legal, no loop-carried dependence
  for (int i = 0; i < 10; i++) {
    a[i] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    a[i] = c[i+1] + 2;
  }
}

*/
TEST_F(FusionLegalTest, NoLoopCarriedDependencesWAW) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %25 "b"
               OpName %34 "i"
               OpName %43 "c"
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
         %29 = OpConstant %6 1
         %48 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %25 = OpVariable %22 Function
         %34 = OpVariable %7 Function
         %43 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %53 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %53 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpAccessChain %7 %25 %53
         %28 = OpLoad %6 %27
         %30 = OpIAdd %6 %28 %29
         %31 = OpAccessChain %7 %23 %53
               OpStore %31 %30
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %53 %29
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %54 = OpPhi %6 %9 %12 %52 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %54 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpIAdd %6 %54 %29
         %46 = OpAccessChain %7 %43 %45
         %47 = OpLoad %6 %46
         %49 = OpIAdd %6 %47 %48
         %50 = OpAccessChain %7 %23 %54
               OpStore %50 %49
               OpBranch %38
         %38 = OpLabel
         %52 = OpIAdd %6 %54 %29
               OpStore %34 %52
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 1u);

    std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[I_1:%\w+]] = OpIAdd {{%\w+}} [[PHI]] {{%\w+}}
CHECK-NEXT: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[I_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
    )";

    Match(checks, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10][10] a;
  int[10][10] b;
  int[10][10] c;
  // Legal outer. Continue and break are fine if nested in inner loops
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      if (j % 2 == 0) {
        c[i][j] = a[i][j] + 2;
      } else {
        continue;
      }
    }
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      if (j % 2 == 0) {
        b[i][j] = c[i][j] + 10;
      } else {
        break;
      }
    }
  }
}

*/
TEST_F(FusionLegalTest, OuterloopWithBreakContinueInInner) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %19 "j"
               OpName %38 "c"
               OpName %41 "a"
               OpName %55 "i"
               OpName %63 "j"
               OpName %76 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %28 = OpConstant %6 2
         %33 = OpTypeInt 32 0
         %34 = OpConstant %33 10
         %35 = OpTypeArray %6 %34
         %36 = OpTypeArray %35 %34
         %37 = OpTypePointer Function %36
         %51 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %38 = OpVariable %37 Function
         %41 = OpVariable %37 Function
         %55 = OpVariable %7 Function
         %63 = OpVariable %7 Function
         %76 = OpVariable %37 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %91 = OpPhi %6 %9 %5 %54 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %91 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
         %96 = OpPhi %6 %9 %11 %52 %23
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %26 = OpSLessThan %17 %96 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
         %29 = OpSMod %6 %96 %28
         %30 = OpIEqual %17 %29 %9
               OpSelectionMerge %23 None
               OpBranchConditional %30 %31 %48
         %31 = OpLabel
         %44 = OpAccessChain %7 %41 %91 %96
         %45 = OpLoad %6 %44
         %46 = OpIAdd %6 %45 %28
         %47 = OpAccessChain %7 %38 %91 %96
               OpStore %47 %46
               OpBranch %32
         %48 = OpLabel
               OpBranch %23
         %32 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %52 = OpIAdd %6 %96 %51
               OpStore %19 %52
               OpBranch %20
         %22 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %54 = OpIAdd %6 %91 %51
               OpStore %8 %54
               OpBranch %10
         %12 = OpLabel
               OpStore %55 %9
               OpBranch %56
         %56 = OpLabel
         %92 = OpPhi %6 %9 %12 %90 %59
               OpLoopMerge %58 %59 None
               OpBranch %60
         %60 = OpLabel
         %62 = OpSLessThan %17 %92 %16
               OpBranchConditional %62 %57 %58
         %57 = OpLabel
               OpStore %63 %9
               OpBranch %64
         %64 = OpLabel
         %93 = OpPhi %6 %9 %57 %88 %67
               OpLoopMerge %66 %67 None
               OpBranch %68
         %68 = OpLabel
         %70 = OpSLessThan %17 %93 %16
               OpBranchConditional %70 %65 %66
         %65 = OpLabel
         %72 = OpSMod %6 %93 %28
         %73 = OpIEqual %17 %72 %9
               OpSelectionMerge %75 None
               OpBranchConditional %73 %74 %66
         %74 = OpLabel
         %81 = OpAccessChain %7 %38 %92 %93
         %82 = OpLoad %6 %81
         %83 = OpIAdd %6 %82 %16
         %84 = OpAccessChain %7 %76 %92 %93
               OpStore %84 %83
               OpBranch %75
         %75 = OpLabel
               OpBranch %67
         %67 = OpLabel
         %88 = OpIAdd %6 %93 %51
               OpStore %63 %88
               OpBranch %64
         %66 = OpLabel
               OpBranch %59
         %59 = OpLabel
         %90 = OpIAdd %6 %92 %51
               OpStore %55 %90
               OpBranch %56
         %58 = OpLabel
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 4u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[0], loops[2]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 3u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[1], loops[2]);
    EXPECT_FALSE(fusion.AreCompatible());

    std::string checks = R"(
CHECK: [[PHI_0:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[PHI_1:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK: [[PHI_2:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_2]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]] [[PHI_2]]
CHECK-NEXT: OpStore [[STORE_1]]
      )";

    Match(checks, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// j loop preheader removed manually
#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int i = 0;
  int j = 0;
  // No loop-carried dependences, legal
  for (; i < 10; i++) {
    a[i] = a[i]*2;
  }
  for (; j < 10; j++) {
    b[j] = a[j]+2;
  }
}

*/
TEST_F(FusionLegalTest, DifferentArraysInLoopsNoPreheader) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %10 "j"
               OpName %24 "a"
               OpName %42 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 10
         %18 = OpTypeBool
         %20 = OpTypeInt 32 0
         %21 = OpConstant %20 10
         %22 = OpTypeArray %6 %21
         %23 = OpTypePointer Function %22
         %29 = OpConstant %6 2
         %33 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %24 = OpVariable %23 Function
         %42 = OpVariable %23 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
         %51 = OpPhi %6 %9 %5 %34 %14
               OpLoopMerge %35 %14 None
               OpBranch %15
         %15 = OpLabel
         %19 = OpSLessThan %18 %51 %17
               OpBranchConditional %19 %12 %35
         %12 = OpLabel
         %27 = OpAccessChain %7 %24 %51
         %28 = OpLoad %6 %27
         %30 = OpIMul %6 %28 %29
         %31 = OpAccessChain %7 %24 %51
               OpStore %31 %30
               OpBranch %14
         %14 = OpLabel
         %34 = OpIAdd %6 %51 %33
               OpStore %8 %34
               OpBranch %11
         %35 = OpLabel
         %52 = OpPhi %6 %9 %15 %50 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %18 %52 %17
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpAccessChain %7 %24 %52
         %46 = OpLoad %6 %45
         %47 = OpIAdd %6 %46 %29
         %48 = OpAccessChain %7 %42 %52
               OpStore %48 %47
               OpBranch %38
         %38 = OpLabel
         %50 = OpIAdd %6 %52 %33
               OpStore %10 %50
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    {
      LoopFusion fusion(context.get(), loops[0], loops[1]);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    ld.CreatePreHeaderBlocksIfMissing();

    {
      LoopFusion fusion(context.get(), loops[0], loops[1]);
      EXPECT_TRUE(fusion.AreCompatible());
      EXPECT_TRUE(fusion.IsLegal());

      fusion.Fuse();
    }
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 1u);

    std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
      )";

    Match(checks, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// j & k loop preheaders removed manually
#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int i = 0;
  int j = 0;
  int k = 0;
  // No loop-carried dependences, legal
  for (; i < 10; i++) {
    a[i] = a[i]*2;
  }
  for (; j < 10; j++) {
    b[j] = a[j]+2;
  }
  for (; k < 10; k++) {
    a[k] = a[k]*2;
  }
}

*/
TEST_F(FusionLegalTest, AdjacentLoopsNoPreheaders) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %10 "j"
               OpName %11 "k"
               OpName %25 "a"
               OpName %43 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %18 = OpConstant %6 10
         %19 = OpTypeBool
         %21 = OpTypeInt 32 0
         %22 = OpConstant %21 10
         %23 = OpTypeArray %6 %22
         %24 = OpTypePointer Function %23
         %30 = OpConstant %6 2
         %34 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %11 = OpVariable %7 Function
         %25 = OpVariable %24 Function
         %43 = OpVariable %24 Function
               OpStore %8 %9
               OpStore %10 %9
               OpStore %11 %9
               OpBranch %12
         %12 = OpLabel
         %67 = OpPhi %6 %9 %5 %35 %15
               OpLoopMerge %36 %15 None
               OpBranch %16
         %16 = OpLabel
         %20 = OpSLessThan %19 %67 %18
               OpBranchConditional %20 %13 %36
         %13 = OpLabel
         %28 = OpAccessChain %7 %25 %67
         %29 = OpLoad %6 %28
         %31 = OpIMul %6 %29 %30
         %32 = OpAccessChain %7 %25 %67
               OpStore %32 %31
               OpBranch %15
         %15 = OpLabel
         %35 = OpIAdd %6 %67 %34
               OpStore %8 %35
               OpBranch %12
         %36 = OpLabel
         %68 = OpPhi %6 %9 %16 %51 %39
               OpLoopMerge %52 %39 None
               OpBranch %40
         %40 = OpLabel
         %42 = OpSLessThan %19 %68 %18
               OpBranchConditional %42 %37 %52
         %37 = OpLabel
         %46 = OpAccessChain %7 %25 %68
         %47 = OpLoad %6 %46
         %48 = OpIAdd %6 %47 %30
         %49 = OpAccessChain %7 %43 %68
               OpStore %49 %48
               OpBranch %39
         %39 = OpLabel
         %51 = OpIAdd %6 %68 %34
               OpStore %10 %51
               OpBranch %36
         %52 = OpLabel
         %70 = OpPhi %6 %9 %40 %66 %55
               OpLoopMerge %54 %55 None
               OpBranch %56
         %56 = OpLabel
         %58 = OpSLessThan %19 %70 %18
               OpBranchConditional %58 %53 %54
         %53 = OpLabel
         %61 = OpAccessChain %7 %25 %70
         %62 = OpLoad %6 %61
         %63 = OpIMul %6 %62 %30
         %64 = OpAccessChain %7 %25 %70
               OpStore %64 %63
               OpBranch %55
         %55 = OpLabel
         %66 = OpIAdd %6 %70 %34
               OpStore %11 %66
               OpBranch %52
         %54 = OpLabel
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 3u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    {
      LoopFusion fusion(context.get(), loops[0], loops[1]);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    ld.CreatePreHeaderBlocksIfMissing();

    {
      LoopFusion fusion(context.get(), loops[0], loops[1]);
      EXPECT_TRUE(fusion.AreCompatible());
      EXPECT_TRUE(fusion.IsLegal());

      fusion.Fuse();
    }
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    std::string checks = R"(
CHECK: [[PHI_0:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_0]]
CHECK-NEXT: OpStore [[STORE_1]]
CHECK: [[PHI_1:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_2:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_1]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_2]]
CHECK: [[STORE_2:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI_1]]
CHECK-NEXT: OpStore [[STORE_2]]
      )";

    Match(checks, context.get());

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 1u);

    std::string checks = R"(
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_0]]
CHECK: [[STORE_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_2:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpLoad {{%\w+}} [[LOAD_2]]
CHECK: [[STORE_2:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_2]]
      )";

    Match(checks, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;

  int sum_0 = 0;
  int sum_1 = 0;

  // No loop-carried dependences, legal
  for (int i = 0; i < 10; i++) {
    sum_0 += a[i];
  }
  for (int j = 0; j < 10; j++) {
    sum_1 += b[j];
  }

  int total = sum_0 + sum_1;
}

*/
TEST_F(FusionLegalTest, IndependentReductions) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "sum_0"
               OpName %10 "sum_1"
               OpName %11 "i"
               OpName %25 "a"
               OpName %34 "j"
               OpName %42 "b"
               OpName %50 "total"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %18 = OpConstant %6 10
         %19 = OpTypeBool
         %21 = OpTypeInt 32 0
         %22 = OpConstant %21 10
         %23 = OpTypeArray %6 %22
         %24 = OpTypePointer Function %23
         %32 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %11 = OpVariable %7 Function
         %25 = OpVariable %24 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %24 Function
         %50 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %9
               OpStore %11 %9
               OpBranch %12
         %12 = OpLabel
         %57 = OpPhi %6 %9 %5 %30 %15
         %54 = OpPhi %6 %9 %5 %33 %15
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
         %20 = OpSLessThan %19 %54 %18
               OpBranchConditional %20 %13 %14
         %13 = OpLabel
         %27 = OpAccessChain %7 %25 %54
         %28 = OpLoad %6 %27
         %30 = OpIAdd %6 %57 %28
               OpStore %8 %30
               OpBranch %15
         %15 = OpLabel
         %33 = OpIAdd %6 %54 %32
               OpStore %11 %33
               OpBranch %12
         %14 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %58 = OpPhi %6 %9 %14 %47 %38
         %55 = OpPhi %6 %9 %14 %49 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %19 %55 %18
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %44 = OpAccessChain %7 %42 %55
         %45 = OpLoad %6 %44
         %47 = OpIAdd %6 %58 %45
               OpStore %10 %47
               OpBranch %38
         %38 = OpLabel
         %49 = OpIAdd %6 %55 %32
               OpStore %34 %49
               OpBranch %35
         %37 = OpLabel
         %53 = OpIAdd %6 %57 %58
               OpStore %50 %53
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 1u);

    std::string checks = R"(
CHECK: [[SUM_0:%\w+]] = OpPhi
CHECK-NEXT: [[SUM_1:%\w+]] = OpPhi
CHECK-NEXT: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: [[LOAD_RES_0:%\w+]] = OpLoad {{%\w+}} [[LOAD_0]]
CHECK-NEXT: [[ADD_RES_0:%\w+]] = OpIAdd {{%\w+}} [[SUM_0]] [[LOAD_RES_0]]
CHECK-NEXT: OpStore {{%\w+}} [[ADD_RES_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: [[LOAD_RES_1:%\w+]] = OpLoad {{%\w+}} [[LOAD_1]]
CHECK-NEXT: [[ADD_RES_1:%\w+]] = OpIAdd {{%\w+}} [[SUM_1]] [[LOAD_RES_1]]
CHECK-NEXT: OpStore {{%\w+}} [[ADD_RES_1]]
      )";

    Match(checks, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;

  int sum_0 = 0;
  int sum_1 = 0;

  // No loop-carried dependences, legal
  for (int i = 0; i < 10; i++) {
    sum_0 += a[i];
  }
  for (int j = 0; j < 10; j++) {
    sum_1 += b[j];
  }

  int total = sum_0 + sum_1;
}

*/
TEST_F(FusionLegalTest, IndependentReductionsOneLCSSA) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "sum_0"
               OpName %10 "sum_1"
               OpName %11 "i"
               OpName %25 "a"
               OpName %34 "j"
               OpName %42 "b"
               OpName %50 "total"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %18 = OpConstant %6 10
         %19 = OpTypeBool
         %21 = OpTypeInt 32 0
         %22 = OpConstant %21 10
         %23 = OpTypeArray %6 %22
         %24 = OpTypePointer Function %23
         %32 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %11 = OpVariable %7 Function
         %25 = OpVariable %24 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %24 Function
         %50 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %9
               OpStore %11 %9
               OpBranch %12
         %12 = OpLabel
         %57 = OpPhi %6 %9 %5 %30 %15
         %54 = OpPhi %6 %9 %5 %33 %15
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
         %20 = OpSLessThan %19 %54 %18
               OpBranchConditional %20 %13 %14
         %13 = OpLabel
         %27 = OpAccessChain %7 %25 %54
         %28 = OpLoad %6 %27
         %30 = OpIAdd %6 %57 %28
               OpStore %8 %30
               OpBranch %15
         %15 = OpLabel
         %33 = OpIAdd %6 %54 %32
               OpStore %11 %33
               OpBranch %12
         %14 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %58 = OpPhi %6 %9 %14 %47 %38
         %55 = OpPhi %6 %9 %14 %49 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %19 %55 %18
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %44 = OpAccessChain %7 %42 %55
         %45 = OpLoad %6 %44
         %47 = OpIAdd %6 %58 %45
               OpStore %10 %47
               OpBranch %38
         %38 = OpLabel
         %49 = OpIAdd %6 %55 %32
               OpStore %34 %49
               OpBranch %35
         %37 = OpLabel
         %53 = OpIAdd %6 %57 %58
               OpStore %50 %53
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopUtils utils_0(context.get(), loops[0]);
    utils_0.MakeLoopClosedSSA();

    LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 1u);

    std::string checks = R"(
CHECK: [[SUM_0:%\w+]] = OpPhi
CHECK-NEXT: [[SUM_1:%\w+]] = OpPhi
CHECK-NEXT: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: [[LOAD_RES_0:%\w+]] = OpLoad {{%\w+}} [[LOAD_0]]
CHECK-NEXT: [[ADD_RES_0:%\w+]] = OpIAdd {{%\w+}} [[SUM_0]] [[LOAD_RES_0]]
CHECK-NEXT: OpStore {{%\w+}} [[ADD_RES_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: [[LOAD_RES_1:%\w+]] = OpLoad {{%\w+}} [[LOAD_1]]
CHECK-NEXT: [[ADD_RES_1:%\w+]] = OpIAdd {{%\w+}} [[SUM_1]] [[LOAD_RES_1]]
CHECK-NEXT: OpStore {{%\w+}} [[ADD_RES_1]]
      )";

    Match(checks, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;

  int sum_0 = 0;
  int sum_1 = 0;

  // No loop-carried dependences, legal
  for (int i = 0; i < 10; i++) {
    sum_0 += a[i];
  }
  for (int j = 0; j < 10; j++) {
    sum_1 += b[j];
  }

  int total = sum_0 + sum_1;
}

*/
TEST_F(FusionLegalTest, IndependentReductionsBothLCSSA) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "sum_0"
               OpName %10 "sum_1"
               OpName %11 "i"
               OpName %25 "a"
               OpName %34 "j"
               OpName %42 "b"
               OpName %50 "total"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %18 = OpConstant %6 10
         %19 = OpTypeBool
         %21 = OpTypeInt 32 0
         %22 = OpConstant %21 10
         %23 = OpTypeArray %6 %22
         %24 = OpTypePointer Function %23
         %32 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %11 = OpVariable %7 Function
         %25 = OpVariable %24 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %24 Function
         %50 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %9
               OpStore %11 %9
               OpBranch %12
         %12 = OpLabel
         %57 = OpPhi %6 %9 %5 %30 %15
         %54 = OpPhi %6 %9 %5 %33 %15
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
         %20 = OpSLessThan %19 %54 %18
               OpBranchConditional %20 %13 %14
         %13 = OpLabel
         %27 = OpAccessChain %7 %25 %54
         %28 = OpLoad %6 %27
         %30 = OpIAdd %6 %57 %28
               OpStore %8 %30
               OpBranch %15
         %15 = OpLabel
         %33 = OpIAdd %6 %54 %32
               OpStore %11 %33
               OpBranch %12
         %14 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %58 = OpPhi %6 %9 %14 %47 %38
         %55 = OpPhi %6 %9 %14 %49 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %19 %55 %18
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %44 = OpAccessChain %7 %42 %55
         %45 = OpLoad %6 %44
         %47 = OpIAdd %6 %58 %45
               OpStore %10 %47
               OpBranch %38
         %38 = OpLabel
         %49 = OpIAdd %6 %55 %32
               OpStore %34 %49
               OpBranch %35
         %37 = OpLabel
         %53 = OpIAdd %6 %57 %58
               OpStore %50 %53
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopUtils utils_0(context.get(), loops[0]);
    utils_0.MakeLoopClosedSSA();
    LoopUtils utils_1(context.get(), loops[1]);
    utils_1.MakeLoopClosedSSA();

    LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 1u);

    std::string checks = R"(
CHECK: [[SUM_0:%\w+]] = OpPhi
CHECK-NEXT: [[SUM_1:%\w+]] = OpPhi
CHECK-NEXT: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: [[LOAD_RES_0:%\w+]] = OpLoad {{%\w+}} [[LOAD_0]]
CHECK-NEXT: [[ADD_RES_0:%\w+]] = OpIAdd {{%\w+}} [[SUM_0]] [[LOAD_RES_0]]
CHECK-NEXT: OpStore {{%\w+}} [[ADD_RES_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: [[LOAD_RES_1:%\w+]] = OpLoad {{%\w+}} [[LOAD_1]]
CHECK-NEXT: [[ADD_RES_1:%\w+]] = OpIAdd {{%\w+}} [[SUM_1]] [[LOAD_RES_1]]
CHECK-NEXT: OpStore {{%\w+}} [[ADD_RES_1]]
      )";

    Match(checks, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;

  int sum_0 = 0;

  // No loop-carried dependences, legal
  for (int i = 0; i < 10; i++) {
    sum_0 += a[i];
  }
  for (int j = 0; j < 10; j++) {
    a[j] = b[j];
  }
}

*/
TEST_F(FusionLegalTest, LoadStoreReductionAndNonLoopCarriedDependence) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "sum_0"
               OpName %10 "i"
               OpName %24 "a"
               OpName %33 "j"
               OpName %42 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 10
         %18 = OpTypeBool
         %20 = OpTypeInt 32 0
         %21 = OpConstant %20 10
         %22 = OpTypeArray %6 %21
         %23 = OpTypePointer Function %22
         %31 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %24 = OpVariable %23 Function
         %33 = OpVariable %7 Function
         %42 = OpVariable %23 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
         %51 = OpPhi %6 %9 %5 %29 %14
         %49 = OpPhi %6 %9 %5 %32 %14
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %19 = OpSLessThan %18 %49 %17
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
         %26 = OpAccessChain %7 %24 %49
         %27 = OpLoad %6 %26
         %29 = OpIAdd %6 %51 %27
               OpStore %8 %29
               OpBranch %14
         %14 = OpLabel
         %32 = OpIAdd %6 %49 %31
               OpStore %10 %32
               OpBranch %11
         %13 = OpLabel
               OpStore %33 %9
               OpBranch %34
         %34 = OpLabel
         %50 = OpPhi %6 %9 %13 %48 %37
               OpLoopMerge %36 %37 None
               OpBranch %38
         %38 = OpLabel
         %40 = OpSLessThan %18 %50 %17
               OpBranchConditional %40 %35 %36
         %35 = OpLabel
         %44 = OpAccessChain %7 %42 %50
         %45 = OpLoad %6 %44
         %46 = OpAccessChain %7 %24 %50
               OpStore %46 %45
               OpBranch %37
         %37 = OpLabel
         %48 = OpIAdd %6 %50 %31
               OpStore %33 %48
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    // TODO: Loop descriptor doesn't return induction variables but all OpPhi
    // in the header and LoopDependenceAnalysis falls over.
    // EXPECT_TRUE(fusion.IsLegal());

    // fusion.Fuse();
  }

  {
    // LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    // EXPECT_EQ(ld.NumLoops(), 1u);

    //       std::string checks = R"(
    // CHECK: [[SUM_0:%\w+]] = OpPhi
    // CHECK-NEXT: [[PHI:%\w+]] = OpPhi
    // CHECK-NEXT: OpLoopMerge
    // CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
    // CHECK-NEXT: [[LOAD_RES_0:%\w+]] = OpLoad {{%\w+}} [[LOAD_0]]
    // CHECK-NEXT: [[ADD_RES_0:%\w+]] = OpIAdd {{%\w+}} [[SUM_0]] [[LOAD_RES_0]]
    // CHECK-NEXT: OpStore {{%\w+}} [[ADD_RES_0]]
    // CHECK-NOT: OpPhi
    // CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
    // CHECK-NEXT: [[LOAD_RES_1:%\w+]] = OpLoad {{%\w+}} [[LOAD_1]]
    // CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
    // CHECK-NEXT: OpStore [[STORE_1]] [[LOAD_RES_1]]
    //       )";

    // Match(checks, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
int x;
void main() {
  int[10] a;
  int[10] b;

  // Legal.
  for (int i = 0; i < 10; i++) {
    x += a[i];
  }
  for (int j = 0; j < 10; j++) {
    b[j] = b[j]+1;
  }
}

*/
TEST_F(FusionLegalTest, ReductionAndNonLoopCarriedDependence) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %20 "x"
               OpName %25 "a"
               OpName %34 "j"
               OpName %42 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %19 = OpTypePointer Private %6
         %20 = OpVariable %19 Private
         %21 = OpTypeInt 32 0
         %22 = OpConstant %21 10
         %23 = OpTypeArray %6 %22
         %24 = OpTypePointer Function %23
         %32 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %25 = OpVariable %24 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %24 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %51 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %51 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpAccessChain %7 %25 %51
         %28 = OpLoad %6 %27
         %29 = OpLoad %6 %20
         %30 = OpIAdd %6 %29 %28
               OpStore %20 %30
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %51 %32
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %52 = OpPhi %6 %9 %12 %50 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %52 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpAccessChain %7 %42 %52
         %46 = OpLoad %6 %45
         %47 = OpIAdd %6 %46 %32
         %48 = OpAccessChain %7 %42 %52
               OpStore %48 %47
               OpBranch %38
         %38 = OpLabel
         %50 = OpIAdd %6 %52 %32
               OpStore %34 %50
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 1u);

    std::string checks = R"(
CHECK: OpName [[X:%\w+]] "x"
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[LOAD_0:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: [[LOAD_RES_0:%\w+]] = OpLoad {{%\w+}} [[LOAD_0]]
CHECK-NEXT: [[X_LOAD:%\w+]] = OpLoad {{%\w+}} [[X]]
CHECK-NEXT: [[ADD_RES_0:%\w+]] = OpIAdd {{%\w+}} [[X_LOAD]] [[LOAD_RES_0]]
CHECK-NEXT: OpStore [[X]] [[ADD_RES_0]]
CHECK-NOT: OpPhi
CHECK: [[LOAD_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: {{%\w+}} = OpLoad {{%\w+}} [[LOAD_1]]
CHECK: [[STORE_1:%\w+]] = OpAccessChain {{%\w+}} {{%\w+}} [[PHI]]
CHECK-NEXT: OpStore [[STORE_1]]
      )";

    Match(checks, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
struct TestStruct {
  int[10] a;
  int b;
};

void main() {
  TestStruct test_0;
  TestStruct test_1;
  TestStruct test_2;

  test_1.b = 2;

  for (int i = 0; i < 10; i++) {
    test_0.a[i] = i;
  }
  for (int j = 0; j < 10; j++) {
    test_2 = test_1;
  }
}

*/
TEST_F(FusionLegalTest, ArrayInStruct) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %10 "TestStruct"
               OpMemberName %10 0 "a"
               OpMemberName %10 1 "b"
               OpName %12 "test_1"
               OpName %17 "i"
               OpName %28 "test_0"
               OpName %34 "j"
               OpName %42 "test_2"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeInt 32 0
          %8 = OpConstant %7 10
          %9 = OpTypeArray %6 %8
         %10 = OpTypeStruct %9 %6
         %11 = OpTypePointer Function %10
         %13 = OpConstant %6 1
         %14 = OpConstant %6 2
         %15 = OpTypePointer Function %6
         %18 = OpConstant %6 0
         %25 = OpConstant %6 10
         %26 = OpTypeBool
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %12 = OpVariable %11 Function
         %17 = OpVariable %15 Function
         %28 = OpVariable %11 Function
         %34 = OpVariable %15 Function
         %42 = OpVariable %11 Function
         %16 = OpAccessChain %15 %12 %13
               OpStore %16 %14
               OpStore %17 %18
               OpBranch %19
         %19 = OpLabel
         %46 = OpPhi %6 %18 %5 %33 %22
               OpLoopMerge %21 %22 None
               OpBranch %23
         %23 = OpLabel
         %27 = OpSLessThan %26 %46 %25
               OpBranchConditional %27 %20 %21
         %20 = OpLabel
         %31 = OpAccessChain %15 %28 %18 %46
               OpStore %31 %46
               OpBranch %22
         %22 = OpLabel
         %33 = OpIAdd %6 %46 %13
               OpStore %17 %33
               OpBranch %19
         %21 = OpLabel
               OpStore %34 %18
               OpBranch %35
         %35 = OpLabel
         %47 = OpPhi %6 %18 %21 %45 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %26 %47 %25
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %43 = OpLoad %10 %12
               OpStore %42 %43
               OpBranch %38
         %38 = OpLabel
         %45 = OpIAdd %6 %47 %13
               OpStore %34 %45
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

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInBinaryLayoutOrder();

    LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_TRUE(fusion.IsLegal());

    fusion.Fuse();
  }

  {
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 1u);

    // clang-format off
        std::string checks = R"(
CHECK: OpName [[TEST_1:%\w+]] "test_1"
CHECK: OpName [[TEST_0:%\w+]] "test_0"
CHECK: OpName [[TEST_2:%\w+]] "test_2"
CHECK: [[PHI:%\w+]] = OpPhi
CHECK-NEXT: OpLoopMerge
CHECK: [[TEST_0_STORE:%\w+]] = OpAccessChain {{%\w+}} [[TEST_0]] {{%\w+}} {{%\w+}}
CHECK-NEXT: OpStore [[TEST_0_STORE]] [[PHI]]
CHECK-NOT: OpPhi
CHECK: [[TEST_1_LOAD:%\w+]] = OpLoad {{%\w+}} [[TEST_1]]
CHECK: OpStore [[TEST_2]] [[TEST_1_LOAD]]
      )";
    // clang-format on

    Match(checks, context.get());
  }
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
