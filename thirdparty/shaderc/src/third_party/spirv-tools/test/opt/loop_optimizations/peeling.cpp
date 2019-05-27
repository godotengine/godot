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
#include "source/opt/ir_builder.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/loop_peeling.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using PeelingTest = PassTest<::testing::Test>;

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

First test:
#version 330 core
void main() {
  for(int i = 0; i < 10; ++i) {
    if (i < 4)
      break;
  }
}

Second test (with a common sub-expression elimination):
#version 330 core
void main() {
  for(int i = 0; i + 1 < 10; ++i) {
  }
}

Third test:
#version 330 core
void main() {
  int a[10];
  for (int i = 0; a[i] != 0; i++) {}
}

Forth test:
#version 330 core
void main() {
  for (long i = 0; i < 10; i++) {}
}

Fifth test:
#version 330 core
void main() {
  for (float i = 0; i < 10; i++) {}
}

Sixth test:
#version 450
layout(location = 0)out float o;
void main() {
  o = 0.0;
  for( int i = 0; true; i++ ) {
    o += 1.0;
    if (i > 10) break;
  }
}
*/
TEST_F(PeelingTest, CannotPeel) {
  // Build the given SPIR-V program in |text|, take the first loop in the first
  // function and test that it is not peelable. |loop_count_id| is the id
  // representing the loop count, if equals to 0, then the function build a 10
  // constant as loop count.
  auto test_cannot_peel = [](const std::string& text, uint32_t loop_count_id) {
    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    Function& f = *module->begin();
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    Instruction* loop_count = nullptr;
    if (loop_count_id) {
      loop_count = context->get_def_use_mgr()->GetDef(loop_count_id);
    } else {
      InstructionBuilder builder(context.get(), &*f.begin());
      // Exit condition.
      loop_count = builder.GetSintConstant(10);
    }

    LoopPeeling peel(&*ld.begin(), loop_count);
    EXPECT_FALSE(peel.CanPeelLoop());
  };
  {
    SCOPED_TRACE("loop with break");

    const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
      %int_4 = OpConstant %int 4
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %28 = OpPhi %int %int_0 %5 %27 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %bool %28 %int_10
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %21 = OpSLessThan %bool %28 %int_4
               OpSelectionMerge %23 None
               OpBranchConditional %21 %22 %23
         %22 = OpLabel
               OpBranch %12
         %23 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %27 = OpIAdd %int %28 %int_1
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
    test_cannot_peel(text, 0);
  }

  {
    SCOPED_TRACE("Ambiguous iterator update");

    const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %23 = OpPhi %int %int_0 %5 %17 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %17 = OpIAdd %int %23 %int_1
         %20 = OpSLessThan %bool %17 %int_10
               OpBranchConditional %20 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

    test_cannot_peel(text, 0);
  }

  {
    SCOPED_TRACE("No loop static bounds");

    const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
               OpName %i "i"
               OpName %a "a"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
       %uint = OpTypeInt 32 0
    %uint_10 = OpConstant %uint 10
%_arr_int_uint_10 = OpTypeArray %int %uint_10
%_ptr_Function__arr_int_uint_10 = OpTypePointer Function %_arr_int_uint_10
       %bool = OpTypeBool
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
          %i = OpVariable %_ptr_Function_int Function
          %a = OpVariable %_ptr_Function__arr_int_uint_10 Function
               OpStore %i %int_0
               OpBranch %10
         %10 = OpLabel
         %28 = OpPhi %int %int_0 %5 %27 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %21 = OpAccessChain %_ptr_Function_int %a %28
         %22 = OpLoad %int %21
         %24 = OpINotEqual %bool %22 %int_0
               OpBranchConditional %24 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %27 = OpIAdd %int %28 %int_1
               OpStore %i %27
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

    test_cannot_peel(text, 22);
  }
  {
    SCOPED_TRACE("Int 64 type for conditions");

    const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginLowerLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %4 "i"
          %6 = OpTypeVoid
          %3 = OpTypeFunction %6
          %7 = OpTypeInt 64 1
          %8 = OpTypePointer Function %7
          %9 = OpConstant %7 0
         %15 = OpConstant %7 10
         %16 = OpTypeBool
         %17 = OpConstant %7 1
          %2 = OpFunction %6 None %3
          %5 = OpLabel
          %4 = OpVariable %8 Function
               OpStore %4 %9
               OpBranch %10
         %10 = OpLabel
         %22 = OpPhi %7 %9 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %16 %22 %15
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpIAdd %7 %22 %17
               OpStore %4 %21
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
    // %15 is a constant for a 64 int. Currently rejected.
    test_cannot_peel(text, 15);
  }
  {
    SCOPED_TRACE("Float type for conditions");

    const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginLowerLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %4 "i"
          %6 = OpTypeVoid
          %3 = OpTypeFunction %6
          %7 = OpTypeFloat 32
          %8 = OpTypePointer Function %7
          %9 = OpConstant %7 0
         %15 = OpConstant %7 10
         %16 = OpTypeBool
         %17 = OpConstant %7 1
          %2 = OpFunction %6 None %3
          %5 = OpLabel
          %4 = OpVariable %8 Function
               OpStore %4 %9
               OpBranch %10
         %10 = OpLabel
         %22 = OpPhi %7 %9 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpFOrdLessThan %16 %22 %15
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpFAdd %7 %22 %17
               OpStore %4 %21
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
    // %15 is a constant for a float. Currently rejected.
    test_cannot_peel(text, 15);
  }
  {
    SCOPED_TRACE("Side effect before exit");

    const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %o
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %o "o"
               OpName %i "i"
               OpDecorate %o Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_ptr_Output_float = OpTypePointer Output %float
          %o = OpVariable %_ptr_Output_float Output
    %float_0 = OpConstant %float 0
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
    %float_1 = OpConstant %float 1
     %int_10 = OpConstant %int 10
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
          %i = OpVariable %_ptr_Function_int Function
               OpStore %o %float_0
               OpStore %i %int_0
               OpBranch %14
         %14 = OpLabel
         %33 = OpPhi %int %int_0 %5 %32 %17
               OpLoopMerge %16 %17 None
               OpBranch %15
         %15 = OpLabel
         %22 = OpLoad %float %o
         %23 = OpFAdd %float %22 %float_1
               OpStore %o %23
         %26 = OpSGreaterThan %bool %33 %int_10
               OpSelectionMerge %28 None
               OpBranchConditional %26 %27 %28
         %27 = OpLabel
               OpBranch %16
         %28 = OpLabel
               OpBranch %17
         %17 = OpLabel
         %32 = OpIAdd %int %33 %int_1
               OpStore %i %32
               OpBranch %14
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
    test_cannot_peel(text, 0);
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
void main() {
  int i = 0;
  for (; i < 10; i++) {}
}
*/
TEST_F(PeelingTest, SimplePeeling) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %22 = OpPhi %int %int_0 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %bool %22 %int_10
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpIAdd %int %22 %int_1
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  // Peel before.
  {
    SCOPED_TRACE("Peel before");

    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    Function& f = *module->begin();
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    InstructionBuilder builder(context.get(), &*f.begin());
    // Exit condition.
    Instruction* ten_cst = builder.GetSintConstant(10);

    LoopPeeling peel(&*ld.begin(), ten_cst);
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelBefore(2);

    const std::string check = R"(
CHECK: [[CST_TEN:%\w+]] = OpConstant {{%\w+}} 10
CHECK: [[CST_TWO:%\w+]] = OpConstant {{%\w+}} 2
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK: [[MIN_LOOP_COUNT:%\w+]] = OpSLessThan {{%\w+}} [[CST_TWO]] [[CST_TEN]]
CHECK-NEXT: [[LOOP_COUNT:%\w+]] = OpSelect {{%\w+}} [[MIN_LOOP_COUNT]] [[CST_TWO]] [[CST_TEN]]
CHECK:      [[BEFORE_LOOP:%\w+]] = OpLabel
CHECK-NEXT: [[DUMMY_IT:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[DUMMY_IT_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: [[i:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE]]
CHECK-NEXT: OpLoopMerge [[AFTER_LOOP_PREHEADER:%\w+]] [[BE]] None
CHECK:      [[COND_BLOCK:%\w+]] = OpLabel
CHECK-NEXT: OpSLessThan
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpSLessThan {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] {{%\w+}} [[AFTER_LOOP_PREHEADER]]
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[i]]
CHECK-NEXT: [[DUMMY_IT_1]] = OpIAdd {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: OpBranch [[BEFORE_LOOP]]

CHECK: [[AFTER_LOOP_PREHEADER]] = OpLabel
CHECK-NEXT: OpSelectionMerge [[IF_MERGE:%\w+]]
CHECK-NEXT: OpBranchConditional [[MIN_LOOP_COUNT]] [[AFTER_LOOP:%\w+]] [[IF_MERGE]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} {{%\w+}} {{%\w+}} [[i]] [[AFTER_LOOP_PREHEADER]]
CHECK-NEXT: OpLoopMerge
)";

    Match(check, context.get());
  }

  // Peel after.
  {
    SCOPED_TRACE("Peel after");

    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    Function& f = *module->begin();
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    InstructionBuilder builder(context.get(), &*f.begin());
    // Exit condition.
    Instruction* ten_cst = builder.GetSintConstant(10);

    LoopPeeling peel(&*ld.begin(), ten_cst);
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelAfter(2);

    const std::string check = R"(
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK:      [[MIN_LOOP_COUNT:%\w+]] = OpSLessThan {{%\w+}}
CHECK-NEXT: OpSelectionMerge [[IF_MERGE:%\w+]]
CHECK-NEXT: OpBranchConditional [[MIN_LOOP_COUNT]] [[BEFORE_LOOP:%\w+]] [[IF_MERGE]]
CHECK:      [[BEFORE_LOOP]] = OpLabel
CHECK-NEXT: [[DUMMY_IT:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[DUMMY_IT_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: [[I:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE]]
CHECK-NEXT: OpLoopMerge [[BEFORE_LOOP_MERGE:%\w+]] [[BE]] None
CHECK:      [[COND_BLOCK:%\w+]] = OpLabel
CHECK-NEXT: OpSLessThan
CHECK-NEXT: [[TMP:%\w+]] = OpIAdd {{%\w+}} [[DUMMY_IT]] {{%\w+}}
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpSLessThan {{%\w+}} [[TMP]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] {{%\w+}} [[BEFORE_LOOP_MERGE]]
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[I]]
CHECK-NEXT: [[DUMMY_IT_1]] = OpIAdd {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: OpBranch [[BEFORE_LOOP]]

CHECK:      [[IF_MERGE]] = OpLabel
CHECK-NEXT: [[TMP:%\w+]] = OpPhi {{%\w+}} [[I]] [[BEFORE_LOOP_MERGE]]
CHECK-NEXT: OpBranch [[AFTER_LOOP:%\w+]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} {{%\w+}} {{%\w+}} [[TMP]] [[IF_MERGE]]
CHECK-NEXT: OpLoopMerge

)";

    Match(check, context.get());
  }

  // Same as above, but reuse the induction variable.
  // Peel before.
  {
    SCOPED_TRACE("Peel before with IV reuse");

    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    Function& f = *module->begin();
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    InstructionBuilder builder(context.get(), &*f.begin());
    // Exit condition.
    Instruction* ten_cst = builder.GetSintConstant(10);

    LoopPeeling peel(&*ld.begin(), ten_cst,
                     context->get_def_use_mgr()->GetDef(22));
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelBefore(2);

    const std::string check = R"(
CHECK: [[CST_TEN:%\w+]] = OpConstant {{%\w+}} 10
CHECK: [[CST_TWO:%\w+]] = OpConstant {{%\w+}} 2
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK: [[MIN_LOOP_COUNT:%\w+]] = OpSLessThan {{%\w+}} [[CST_TWO]] [[CST_TEN]]
CHECK-NEXT: [[LOOP_COUNT:%\w+]] = OpSelect {{%\w+}} [[MIN_LOOP_COUNT]] [[CST_TWO]] [[CST_TEN]]
CHECK:      [[BEFORE_LOOP:%\w+]] = OpLabel
CHECK-NEXT: [[i:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: OpLoopMerge [[AFTER_LOOP_PREHEADER:%\w+]] [[BE]] None
CHECK:      [[COND_BLOCK:%\w+]] = OpLabel
CHECK-NEXT: OpSLessThan
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpSLessThan {{%\w+}} [[i]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] {{%\w+}} [[AFTER_LOOP_PREHEADER]]
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[i]]
CHECK-NEXT: OpBranch [[BEFORE_LOOP]]

CHECK: [[AFTER_LOOP_PREHEADER]] = OpLabel
CHECK-NEXT: OpSelectionMerge [[IF_MERGE:%\w+]]
CHECK-NEXT: OpBranchConditional [[MIN_LOOP_COUNT]] [[AFTER_LOOP:%\w+]] [[IF_MERGE]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} {{%\w+}} {{%\w+}} [[i]] [[AFTER_LOOP_PREHEADER]]
CHECK-NEXT: OpLoopMerge
)";

    Match(check, context.get());
  }

  // Peel after.
  {
    SCOPED_TRACE("Peel after IV reuse");

    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    Function& f = *module->begin();
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    InstructionBuilder builder(context.get(), &*f.begin());
    // Exit condition.
    Instruction* ten_cst = builder.GetSintConstant(10);

    LoopPeeling peel(&*ld.begin(), ten_cst,
                     context->get_def_use_mgr()->GetDef(22));
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelAfter(2);

    const std::string check = R"(
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK:      [[MIN_LOOP_COUNT:%\w+]] = OpSLessThan {{%\w+}}
CHECK-NEXT: OpSelectionMerge [[IF_MERGE:%\w+]]
CHECK-NEXT: OpBranchConditional [[MIN_LOOP_COUNT]] [[BEFORE_LOOP:%\w+]] [[IF_MERGE]]
CHECK:      [[BEFORE_LOOP]] = OpLabel
CHECK-NEXT: [[I:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: OpLoopMerge [[BEFORE_LOOP_MERGE:%\w+]] [[BE]] None
CHECK:      [[COND_BLOCK:%\w+]] = OpLabel
CHECK-NEXT: OpSLessThan
CHECK-NEXT: [[TMP:%\w+]] = OpIAdd {{%\w+}} [[I]] {{%\w+}}
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpSLessThan {{%\w+}} [[TMP]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] {{%\w+}} [[BEFORE_LOOP_MERGE]]
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[I]]
CHECK-NEXT: OpBranch [[BEFORE_LOOP]]

CHECK:      [[IF_MERGE]] = OpLabel
CHECK-NEXT: [[TMP:%\w+]] = OpPhi {{%\w+}} [[I]] [[BEFORE_LOOP_MERGE]]
CHECK-NEXT: OpBranch [[AFTER_LOOP:%\w+]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} {{%\w+}} {{%\w+}} [[TMP]] [[IF_MERGE]]
CHECK-NEXT: OpLoopMerge

)";

    Match(check, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
void main() {
  int a[10];
  int n = a[0];
  for(int i = 0; i < n; ++i) {}
}
*/
TEST_F(PeelingTest, PeelingUncountable) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
               OpName %a "a"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
       %uint = OpTypeInt 32 0
    %uint_10 = OpConstant %uint 10
%_arr_int_uint_10 = OpTypeArray %int %uint_10
%_ptr_Function__arr_int_uint_10 = OpTypePointer Function %_arr_int_uint_10
      %int_0 = OpConstant %int 0
       %bool = OpTypeBool
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
          %a = OpVariable %_ptr_Function__arr_int_uint_10 Function
         %15 = OpAccessChain %_ptr_Function_int %a %int_0
         %16 = OpLoad %int %15
               OpBranch %18
         %18 = OpLabel
         %30 = OpPhi %int %int_0 %5 %29 %21
               OpLoopMerge %20 %21 None
               OpBranch %22
         %22 = OpLabel
         %26 = OpSLessThan %bool %30 %16
               OpBranchConditional %26 %19 %20
         %19 = OpLabel
               OpBranch %21
         %21 = OpLabel
         %29 = OpIAdd %int %30 %int_1
               OpBranch %18
         %20 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  // Peel before.
  {
    SCOPED_TRACE("Peel before");

    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    Function& f = *module->begin();
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    Instruction* loop_count = context->get_def_use_mgr()->GetDef(16);
    EXPECT_EQ(loop_count->opcode(), SpvOpLoad);

    LoopPeeling peel(&*ld.begin(), loop_count);
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelBefore(1);

    const std::string check = R"(
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK:      [[LOOP_COUNT:%\w+]] = OpLoad
CHECK:      [[MIN_LOOP_COUNT:%\w+]] = OpSLessThan {{%\w+}} {{%\w+}} [[LOOP_COUNT]]
CHECK-NEXT: [[LOOP_COUNT:%\w+]] = OpSelect {{%\w+}} [[MIN_LOOP_COUNT]] {{%\w+}} [[LOOP_COUNT]]
CHECK:      [[BEFORE_LOOP:%\w+]] = OpLabel
CHECK-NEXT: [[DUMMY_IT:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[DUMMY_IT_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: [[i:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE]]
CHECK-NEXT: OpLoopMerge [[AFTER_LOOP_PREHEADER:%\w+]] [[BE]] None
CHECK:      [[COND_BLOCK:%\w+]] = OpLabel
CHECK-NEXT: OpSLessThan
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpSLessThan {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] {{%\w+}} [[AFTER_LOOP_PREHEADER]]
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[i]]
CHECK-NEXT: [[DUMMY_IT_1]] = OpIAdd {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: OpBranch [[BEFORE_LOOP]]

CHECK: [[AFTER_LOOP_PREHEADER]] = OpLabel
CHECK-NEXT: OpSelectionMerge [[IF_MERGE:%\w+]]
CHECK-NEXT: OpBranchConditional [[MIN_LOOP_COUNT]] [[AFTER_LOOP:%\w+]] [[IF_MERGE]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} {{%\w+}} {{%\w+}} [[i]] [[AFTER_LOOP_PREHEADER]]
CHECK-NEXT: OpLoopMerge
)";

    Match(check, context.get());
  }

  // Peel after.
  {
    SCOPED_TRACE("Peel after");

    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    Function& f = *module->begin();
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    Instruction* loop_count = context->get_def_use_mgr()->GetDef(16);
    EXPECT_EQ(loop_count->opcode(), SpvOpLoad);

    LoopPeeling peel(&*ld.begin(), loop_count);
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelAfter(1);

    const std::string check = R"(
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK:      [[MIN_LOOP_COUNT:%\w+]] = OpSLessThan {{%\w+}}
CHECK-NEXT: OpSelectionMerge [[IF_MERGE:%\w+]]
CHECK-NEXT: OpBranchConditional [[MIN_LOOP_COUNT]] [[BEFORE_LOOP:%\w+]] [[IF_MERGE]]
CHECK:      [[BEFORE_LOOP]] = OpLabel
CHECK-NEXT: [[DUMMY_IT:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[DUMMY_IT_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: [[I:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE]]
CHECK-NEXT: OpLoopMerge [[BEFORE_LOOP_MERGE:%\w+]] [[BE]] None
CHECK:      [[COND_BLOCK:%\w+]] = OpLabel
CHECK-NEXT: OpSLessThan
CHECK-NEXT: [[TMP:%\w+]] = OpIAdd {{%\w+}} [[DUMMY_IT]] {{%\w+}}
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpSLessThan {{%\w+}} [[TMP]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] {{%\w+}} [[BEFORE_LOOP_MERGE]]
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[I]]
CHECK-NEXT: [[DUMMY_IT_1]] = OpIAdd {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: OpBranch [[BEFORE_LOOP]]

CHECK:      [[IF_MERGE]] = OpLabel
CHECK-NEXT: [[TMP:%\w+]] = OpPhi {{%\w+}} [[I]] [[BEFORE_LOOP_MERGE]]
CHECK-NEXT: OpBranch [[AFTER_LOOP:%\w+]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} {{%\w+}} {{%\w+}} [[TMP]] [[IF_MERGE]]
CHECK-NEXT: OpLoopMerge

)";

    Match(check, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
void main() {
  int i = 0;
  do {
    i++;
  } while (i < 10);
}
*/
TEST_F(PeelingTest, DoWhilePeeling) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %21 = OpPhi %int %int_0 %5 %16 %13
               OpLoopMerge %12 %13 None
               OpBranch %11
         %11 = OpLabel
         %16 = OpIAdd %int %21 %int_1
               OpBranch %13
         %13 = OpLabel
         %20 = OpSLessThan %bool %16 %int_10
               OpBranchConditional %20 %10 %12
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  // Peel before.
  {
    SCOPED_TRACE("Peel before");

    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    Function& f = *module->begin();
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);
    InstructionBuilder builder(context.get(), &*f.begin());
    // Exit condition.
    Instruction* ten_cst = builder.GetUintConstant(10);

    LoopPeeling peel(&*ld.begin(), ten_cst);
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelBefore(2);

    const std::string check = R"(
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK:      [[MIN_LOOP_COUNT:%\w+]] = OpULessThan {{%\w+}}
CHECK-NEXT: [[LOOP_COUNT:%\w+]] = OpSelect {{%\w+}} [[MIN_LOOP_COUNT]]
CHECK:      [[BEFORE_LOOP:%\w+]] = OpLabel
CHECK-NEXT: [[DUMMY_IT:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[DUMMY_IT_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: [[i:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE]]
CHECK-NEXT: OpLoopMerge [[AFTER_LOOP_PREHEADER:%\w+]] [[BE]] None
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[i]]
CHECK:      [[BE]] = OpLabel
CHECK:      [[DUMMY_IT_1]] = OpIAdd {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpULessThan {{%\w+}} [[DUMMY_IT_1]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] [[BEFORE_LOOP]] [[AFTER_LOOP_PREHEADER]]

CHECK:      [[AFTER_LOOP_PREHEADER]] = OpLabel
CHECK-NEXT: OpSelectionMerge [[IF_MERGE:%\w+]]
CHECK-NEXT: OpBranchConditional [[MIN_LOOP_COUNT]] [[AFTER_LOOP:%\w+]] [[IF_MERGE]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} {{%\w+}} {{%\w+}} [[I_1]] [[AFTER_LOOP_PREHEADER]]
CHECK-NEXT: OpLoopMerge
)";

    Match(check, context.get());
  }

  // Peel after.
  {
    SCOPED_TRACE("Peel after");

    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    Function& f = *module->begin();
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    InstructionBuilder builder(context.get(), &*f.begin());
    // Exit condition.
    Instruction* ten_cst = builder.GetUintConstant(10);

    LoopPeeling peel(&*ld.begin(), ten_cst);
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelAfter(2);

    const std::string check = R"(
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK:      [[MIN_LOOP_COUNT:%\w+]] = OpULessThan {{%\w+}}
CHECK-NEXT: OpSelectionMerge [[IF_MERGE:%\w+]]
CHECK-NEXT: OpBranchConditional [[MIN_LOOP_COUNT]] [[BEFORE_LOOP:%\w+]] [[IF_MERGE]]
CHECK:      [[BEFORE_LOOP]] = OpLabel
CHECK-NEXT: [[DUMMY_IT:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[DUMMY_IT_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: [[I:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE]]
CHECK-NEXT: OpLoopMerge [[BEFORE_LOOP_MERGE:%\w+]] [[BE]] None
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[I]]
CHECK:      [[BE]] = OpLabel
CHECK:      [[DUMMY_IT_1]] = OpIAdd {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: [[EXIT_VAL:%\w+]] = OpIAdd {{%\w+}} [[DUMMY_IT_1]]
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpULessThan {{%\w+}} [[EXIT_VAL]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] [[BEFORE_LOOP]] [[BEFORE_LOOP_MERGE]]

CHECK:      [[IF_MERGE]] = OpLabel
CHECK-NEXT: [[TMP:%\w+]] = OpPhi {{%\w+}} [[I_1]] [[BEFORE_LOOP_MERGE]]
CHECK-NEXT: OpBranch [[AFTER_LOOP:%\w+]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} {{%\w+}} {{%\w+}} [[TMP]] [[IF_MERGE]]
CHECK-NEXT: OpLoopMerge
)";

    Match(check, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
void main() {
  int a[10];
  int n = a[0];
  for(int i = 0; i < n; ++i) {}
}
*/
TEST_F(PeelingTest, PeelingLoopWithStore) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %o %n
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %o "o"
               OpName %end "end"
               OpName %n "n"
               OpName %i "i"
               OpDecorate %o Location 0
               OpDecorate %n Flat
               OpDecorate %n Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_ptr_Output_float = OpTypePointer Output %float
          %o = OpVariable %_ptr_Output_float Output
    %float_0 = OpConstant %float 0
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Input_int = OpTypePointer Input %int
          %n = OpVariable %_ptr_Input_int Input
      %int_0 = OpConstant %int 0
       %bool = OpTypeBool
    %float_1 = OpConstant %float 1
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
        %end = OpVariable %_ptr_Function_int Function
          %i = OpVariable %_ptr_Function_int Function
               OpStore %o %float_0
         %15 = OpLoad %int %n
               OpStore %end %15
               OpStore %i %int_0
               OpBranch %18
         %18 = OpLabel
         %33 = OpPhi %int %int_0 %5 %32 %21
               OpLoopMerge %20 %21 None
               OpBranch %22
         %22 = OpLabel
         %26 = OpSLessThan %bool %33 %15
               OpBranchConditional %26 %19 %20
         %19 = OpLabel
         %28 = OpLoad %float %o
         %29 = OpFAdd %float %28 %float_1
               OpStore %o %29
               OpBranch %21
         %21 = OpLabel
         %32 = OpIAdd %int %33 %int_1
               OpStore %i %32
               OpBranch %18
         %20 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  // Peel before.
  {
    SCOPED_TRACE("Peel before");

    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    Function& f = *module->begin();
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    Instruction* loop_count = context->get_def_use_mgr()->GetDef(15);
    EXPECT_EQ(loop_count->opcode(), SpvOpLoad);

    LoopPeeling peel(&*ld.begin(), loop_count);
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelBefore(1);

    const std::string check = R"(
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK:      [[LOOP_COUNT:%\w+]] = OpLoad
CHECK:      [[MIN_LOOP_COUNT:%\w+]] = OpSLessThan {{%\w+}} {{%\w+}} [[LOOP_COUNT]]
CHECK-NEXT: [[LOOP_COUNT:%\w+]] = OpSelect {{%\w+}} [[MIN_LOOP_COUNT]] {{%\w+}} [[LOOP_COUNT]]
CHECK:      [[BEFORE_LOOP:%\w+]] = OpLabel
CHECK-NEXT: [[DUMMY_IT:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[DUMMY_IT_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: [[i:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE]]
CHECK-NEXT: OpLoopMerge [[AFTER_LOOP_PREHEADER:%\w+]] [[BE]] None
CHECK:      [[COND_BLOCK:%\w+]] = OpLabel
CHECK-NEXT: OpSLessThan
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpSLessThan {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] {{%\w+}} [[AFTER_LOOP_PREHEADER]]
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[i]]
CHECK:      [[DUMMY_IT_1]] = OpIAdd {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: OpBranch [[BEFORE_LOOP]]

CHECK: [[AFTER_LOOP_PREHEADER]] = OpLabel
CHECK-NEXT: OpSelectionMerge [[IF_MERGE:%\w+]]
CHECK-NEXT: OpBranchConditional [[MIN_LOOP_COUNT]] [[AFTER_LOOP:%\w+]] [[IF_MERGE]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} {{%\w+}} {{%\w+}} [[i]] [[AFTER_LOOP_PREHEADER]]
CHECK-NEXT: OpLoopMerge
)";

    Match(check, context.get());
  }

  // Peel after.
  {
    SCOPED_TRACE("Peel after");

    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    Function& f = *module->begin();
    LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    Instruction* loop_count = context->get_def_use_mgr()->GetDef(15);
    EXPECT_EQ(loop_count->opcode(), SpvOpLoad);

    LoopPeeling peel(&*ld.begin(), loop_count);
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelAfter(1);

    const std::string check = R"(
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK:      [[MIN_LOOP_COUNT:%\w+]] = OpSLessThan {{%\w+}}
CHECK-NEXT: OpSelectionMerge [[IF_MERGE:%\w+]]
CHECK-NEXT: OpBranchConditional [[MIN_LOOP_COUNT]] [[BEFORE_LOOP:%\w+]] [[IF_MERGE]]
CHECK:      [[BEFORE_LOOP]] = OpLabel
CHECK-NEXT: [[DUMMY_IT:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[DUMMY_IT_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: [[I:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE]]
CHECK-NEXT: OpLoopMerge [[BEFORE_LOOP_MERGE:%\w+]] [[BE]] None
CHECK:      [[COND_BLOCK:%\w+]] = OpLabel
CHECK-NEXT: OpSLessThan
CHECK-NEXT: [[TMP:%\w+]] = OpIAdd {{%\w+}} [[DUMMY_IT]] {{%\w+}}
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpSLessThan {{%\w+}} [[TMP]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] {{%\w+}} [[BEFORE_LOOP_MERGE]]
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[I]]
CHECK:      [[DUMMY_IT_1]] = OpIAdd {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: OpBranch [[BEFORE_LOOP]]

CHECK:      [[IF_MERGE]] = OpLabel
CHECK-NEXT: [[TMP:%\w+]] = OpPhi {{%\w+}} [[I]] [[BEFORE_LOOP_MERGE]]
CHECK-NEXT: OpBranch [[AFTER_LOOP:%\w+]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} {{%\w+}} {{%\w+}} [[TMP]] [[IF_MERGE]]
CHECK-NEXT: OpLoopMerge

)";

    Match(check, context.get());
  }
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
