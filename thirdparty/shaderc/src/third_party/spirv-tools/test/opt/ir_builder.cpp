// Copyright (c) 2018 Google Inc.
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
#include <memory>
#include <string>
#include <vector>

#include "effcee/effcee.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/basic_block.h"
#include "source/opt/build_module.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/type_manager.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace opt {
namespace {

using Analysis = IRContext::Analysis;
using IRBuilderTest = ::testing::Test;

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
      tools.Disassemble(bin, &assembly, SpirvTools::kDefaultDisassembleOption))
      << "Disassembling failed for shader:\n"
      << assembly << std::endl;
  auto match_result = effcee::Match(assembly, original);
  EXPECT_EQ(effcee::Result::Status::Ok, match_result.status())
      << match_result.message() << "\nChecking result:\n"
      << assembly;
}

TEST_F(IRBuilderTest, TestInsnAddition) {
  const std::string text = R"(
; CHECK: %18 = OpLabel
; CHECK: OpPhi %int %int_0 %14
; CHECK: OpPhi %bool %16 %14
; CHECK: OpBranch %17
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %4 "i"
               OpName %3 "c"
               OpDecorate %3 Location 0
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeInt 32 1
          %8 = OpTypePointer Function %7
          %9 = OpConstant %7 0
         %10 = OpTypeBool
         %11 = OpTypeFloat 32
         %12 = OpTypeVector %11 4
         %13 = OpTypePointer Output %12
          %3 = OpVariable %13 Output
          %2 = OpFunction %5 None %6
         %14 = OpLabel
          %4 = OpVariable %8 Function
               OpStore %4 %9
         %15 = OpLoad %7 %4
         %16 = OpINotEqual %10 %15 %9
               OpSelectionMerge %17 None
               OpBranchConditional %16 %18 %17
         %18 = OpLabel
               OpBranch %17
         %17 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  {
    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);

    BasicBlock* bb = context->cfg()->block(18);

    // Build managers.
    context->get_def_use_mgr();
    context->get_instr_block(nullptr);

    InstructionBuilder builder(context.get(), &*bb->begin());
    Instruction* phi1 = builder.AddPhi(7, {9, 14});
    Instruction* phi2 = builder.AddPhi(10, {16, 14});

    // Make sure the InstructionBuilder did not update the def/use manager.
    EXPECT_EQ(context->get_def_use_mgr()->GetDef(phi1->result_id()), nullptr);
    EXPECT_EQ(context->get_def_use_mgr()->GetDef(phi2->result_id()), nullptr);
    EXPECT_EQ(context->get_instr_block(phi1), nullptr);
    EXPECT_EQ(context->get_instr_block(phi2), nullptr);

    Match(text, context.get());
  }

  {
    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);

    // Build managers.
    context->get_def_use_mgr();
    context->get_instr_block(nullptr);

    BasicBlock* bb = context->cfg()->block(18);
    InstructionBuilder builder(
        context.get(), &*bb->begin(),
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    Instruction* phi1 = builder.AddPhi(7, {9, 14});
    Instruction* phi2 = builder.AddPhi(10, {16, 14});

    // Make sure InstructionBuilder updated the def/use manager
    EXPECT_NE(context->get_def_use_mgr()->GetDef(phi1->result_id()), nullptr);
    EXPECT_NE(context->get_def_use_mgr()->GetDef(phi2->result_id()), nullptr);
    EXPECT_NE(context->get_instr_block(phi1), nullptr);
    EXPECT_NE(context->get_instr_block(phi2), nullptr);

    Match(text, context.get());
  }
}

TEST_F(IRBuilderTest, TestCondBranchAddition) {
  const std::string text = R"(
; CHECK: %main = OpFunction %void None %6
; CHECK-NEXT: %15 = OpLabel
; CHECK-NEXT: OpSelectionMerge %13 None
; CHECK-NEXT: OpBranchConditional %true %14 %13
; CHECK-NEXT: %14 = OpLabel
; CHECK-NEXT: OpBranch %13
; CHECK-NEXT: %13 = OpLabel
; CHECK-NEXT: OpReturn
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %4 "i"
               OpName %3 "c"
               OpDecorate %3 Location 0
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeBool
          %8 = OpTypePointer Private %7
          %9 = OpConstantTrue %7
         %10 = OpTypeFloat 32
         %11 = OpTypeVector %10 4
         %12 = OpTypePointer Output %11
          %3 = OpVariable %12 Output
          %4 = OpVariable %8 Private
          %2 = OpFunction %5 None %6
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  {
    std::unique_ptr<IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);

    Function& fn = *context->module()->begin();

    BasicBlock& bb_merge = *fn.begin();

    // TODO(1841): Handle id overflow.
    fn.begin().InsertBefore(std::unique_ptr<BasicBlock>(
        new BasicBlock(std::unique_ptr<Instruction>(new Instruction(
            context.get(), SpvOpLabel, 0, context->TakeNextId(), {})))));
    BasicBlock& bb_true = *fn.begin();
    {
      InstructionBuilder builder(context.get(), &*bb_true.begin());
      builder.AddBranch(bb_merge.id());
    }

    // TODO(1841): Handle id overflow.
    fn.begin().InsertBefore(std::unique_ptr<BasicBlock>(
        new BasicBlock(std::unique_ptr<Instruction>(new Instruction(
            context.get(), SpvOpLabel, 0, context->TakeNextId(), {})))));
    BasicBlock& bb_cond = *fn.begin();

    InstructionBuilder builder(context.get(), &bb_cond);
    // This also test consecutive instruction insertion: merge selection +
    // branch.
    builder.AddConditionalBranch(9, bb_true.id(), bb_merge.id(), bb_merge.id());

    Match(text, context.get());
  }
}

TEST_F(IRBuilderTest, AddSelect) {
  const std::string text = R"(
; CHECK: [[bool:%\w+]] = OpTypeBool
; CHECK: [[uint:%\w+]] = OpTypeInt 32 0
; CHECK: [[true:%\w+]] = OpConstantTrue [[bool]]
; CHECK: [[u0:%\w+]] = OpConstant [[uint]] 0
; CHECK: [[u1:%\w+]] = OpConstant [[uint]] 1
; CHECK: OpSelect [[uint]] [[true]] [[u0]] [[u1]]
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
%1 = OpTypeVoid
%2 = OpTypeBool
%3 = OpTypeInt 32 0
%4 = OpConstantTrue %2
%5 = OpConstant %3 0
%6 = OpConstant %3 1
%7 = OpTypeFunction %1
%8 = OpFunction %1 None %7
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  EXPECT_NE(nullptr, context);

  InstructionBuilder builder(context.get(),
                             &*context->module()->begin()->begin()->begin());
  EXPECT_NE(nullptr, builder.AddSelect(3u, 4u, 5u, 6u));

  Match(text, context.get());
}

TEST_F(IRBuilderTest, AddCompositeConstruct) {
  const std::string text = R"(
; CHECK: [[uint:%\w+]] = OpTypeInt
; CHECK: [[u0:%\w+]] = OpConstant [[uint]] 0
; CHECK: [[u1:%\w+]] = OpConstant [[uint]] 1
; CHECK: [[struct:%\w+]] = OpTypeStruct [[uint]] [[uint]] [[uint]] [[uint]]
; CHECK: OpCompositeConstruct [[struct]] [[u0]] [[u1]] [[u1]] [[u0]]
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpConstant %2 1
%5 = OpTypeStruct %2 %2 %2 %2
%6 = OpTypeFunction %1
%7 = OpFunction %1 None %6
%8 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  EXPECT_NE(nullptr, context);

  InstructionBuilder builder(context.get(),
                             &*context->module()->begin()->begin()->begin());
  std::vector<uint32_t> ids = {3u, 4u, 4u, 3u};
  EXPECT_NE(nullptr, builder.AddCompositeConstruct(5u, ids));

  Match(text, context.get());
}

TEST_F(IRBuilderTest, ConstantAdder) {
  const std::string text = R"(
; CHECK: [[uint:%\w+]] = OpTypeInt 32 0
; CHECK: OpConstant [[uint]] 13
; CHECK: [[sint:%\w+]] = OpTypeInt 32 1
; CHECK: OpConstant [[sint]] -1
; CHECK: OpConstant [[uint]] 1
; CHECK: OpConstant [[sint]] 34
; CHECK: OpConstant [[uint]] 0
; CHECK: OpConstant [[sint]] 0
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  EXPECT_NE(nullptr, context);

  InstructionBuilder builder(context.get(),
                             &*context->module()->begin()->begin()->begin());
  EXPECT_NE(nullptr, builder.GetUintConstant(13));
  EXPECT_NE(nullptr, builder.GetSintConstant(-1));

  // Try adding the same constants again to make sure they aren't added.
  EXPECT_NE(nullptr, builder.GetUintConstant(13));
  EXPECT_NE(nullptr, builder.GetSintConstant(-1));

  // Try adding different constants to make sure the type is reused.
  EXPECT_NE(nullptr, builder.GetUintConstant(1));
  EXPECT_NE(nullptr, builder.GetSintConstant(34));

  // Try adding 0 as both signed and unsigned.
  EXPECT_NE(nullptr, builder.GetUintConstant(0));
  EXPECT_NE(nullptr, builder.GetSintConstant(0));

  Match(text, context.get());
}

TEST_F(IRBuilderTest, ConstantAdderTypeAlreadyExists) {
  const std::string text = R"(
; CHECK: OpConstant %uint 13
; CHECK: OpConstant %int -1
; CHECK: OpConstant %uint 1
; CHECK: OpConstant %int 34
; CHECK: OpConstant %uint 0
; CHECK: OpConstant %int 0
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%4 = OpTypeFunction %1
%5 = OpFunction %1 None %4
%6 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  EXPECT_NE(nullptr, context);

  InstructionBuilder builder(context.get(),
                             &*context->module()->begin()->begin()->begin());
  Instruction* const_1 = builder.GetUintConstant(13);
  Instruction* const_2 = builder.GetSintConstant(-1);

  EXPECT_NE(nullptr, const_1);
  EXPECT_NE(nullptr, const_2);

  // Try adding the same constants again to make sure they aren't added.
  EXPECT_EQ(const_1, builder.GetUintConstant(13));
  EXPECT_EQ(const_2, builder.GetSintConstant(-1));

  Instruction* const_3 = builder.GetUintConstant(1);
  Instruction* const_4 = builder.GetSintConstant(34);

  // Try adding different constants to make sure the type is reused.
  EXPECT_NE(nullptr, const_3);
  EXPECT_NE(nullptr, const_4);

  Instruction* const_5 = builder.GetUintConstant(0);
  Instruction* const_6 = builder.GetSintConstant(0);

  // Try adding 0 as both signed and unsigned.
  EXPECT_NE(nullptr, const_5);
  EXPECT_NE(nullptr, const_6);

  // They have the same value but different types so should be unique.
  EXPECT_NE(const_5, const_6);

  // Check the types are correct.
  uint32_t type_id_unsigned = const_1->GetSingleWordOperand(0);
  uint32_t type_id_signed = const_2->GetSingleWordOperand(0);

  EXPECT_NE(type_id_unsigned, type_id_signed);

  EXPECT_EQ(const_3->GetSingleWordOperand(0), type_id_unsigned);
  EXPECT_EQ(const_5->GetSingleWordOperand(0), type_id_unsigned);

  EXPECT_EQ(const_4->GetSingleWordOperand(0), type_id_signed);
  EXPECT_EQ(const_6->GetSingleWordOperand(0), type_id_signed);

  Match(text, context.get());
}

TEST_F(IRBuilderTest, AccelerationStructureNV) {
  const std::string text = R"(
; CHECK: OpTypeAccelerationStructureNV
OpCapability Shader
OpCapability RayTracingNV
OpExtension "SPV_NV_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %8 "main"
OpExecutionMode %8 OriginUpperLeft
%1 = OpTypeVoid
%2 = OpTypeBool
%3 = OpTypeAccelerationStructureNV
%7 = OpTypeFunction %1
%8 = OpFunction %1 None %7
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  EXPECT_NE(nullptr, context);

  InstructionBuilder builder(context.get(),
                             &*context->module()->begin()->begin()->begin());
  Match(text, context.get());
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
