// Copyright (c) 2016 Google Inc.
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
#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"
#include "spirv-tools/libspirv.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace opt {
namespace {

using spvtest::MakeInstruction;
using ::testing::Eq;
using DescriptorTypeTest = PassTest<::testing::Test>;
using OpaqueTypeTest = PassTest<::testing::Test>;
using GetBaseTest = PassTest<::testing::Test>;
using ValidBasePointerTest = PassTest<::testing::Test>;

TEST(InstructionTest, CreateTrivial) {
  Instruction empty;
  EXPECT_EQ(SpvOpNop, empty.opcode());
  EXPECT_EQ(0u, empty.type_id());
  EXPECT_EQ(0u, empty.result_id());
  EXPECT_EQ(0u, empty.NumOperands());
  EXPECT_EQ(0u, empty.NumOperandWords());
  EXPECT_EQ(0u, empty.NumInOperandWords());
  EXPECT_EQ(empty.cend(), empty.cbegin());
  EXPECT_EQ(empty.end(), empty.begin());
}

TEST(InstructionTest, CreateWithOpcodeAndNoOperands) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction inst(&context, SpvOpReturn);
  EXPECT_EQ(SpvOpReturn, inst.opcode());
  EXPECT_EQ(0u, inst.type_id());
  EXPECT_EQ(0u, inst.result_id());
  EXPECT_EQ(0u, inst.NumOperands());
  EXPECT_EQ(0u, inst.NumOperandWords());
  EXPECT_EQ(0u, inst.NumInOperandWords());
  EXPECT_EQ(inst.cend(), inst.cbegin());
  EXPECT_EQ(inst.end(), inst.begin());
}

// The words for an OpTypeInt for 32-bit signed integer resulting in Id 44.
uint32_t kSampleInstructionWords[] = {(4 << 16) | uint32_t(SpvOpTypeInt), 44,
                                      32, 1};
// The operands that would be parsed from kSampleInstructionWords
spv_parsed_operand_t kSampleParsedOperands[] = {
    {1, 1, SPV_OPERAND_TYPE_RESULT_ID, SPV_NUMBER_NONE, 0},
    {2, 1, SPV_OPERAND_TYPE_LITERAL_INTEGER, SPV_NUMBER_UNSIGNED_INT, 32},
    {3, 1, SPV_OPERAND_TYPE_LITERAL_INTEGER, SPV_NUMBER_UNSIGNED_INT, 1},
};

// A valid parse of kSampleParsedOperands.
spv_parsed_instruction_t kSampleParsedInstruction = {kSampleInstructionWords,
                                                     uint16_t(4),
                                                     uint16_t(SpvOpTypeInt),
                                                     SPV_EXT_INST_TYPE_NONE,
                                                     0,   // type id
                                                     44,  // result id
                                                     kSampleParsedOperands,
                                                     3};

// The words for an OpAccessChain instruction.
uint32_t kSampleAccessChainInstructionWords[] = {
    (7 << 16) | uint32_t(SpvOpAccessChain), 100, 101, 102, 103, 104, 105};

// The operands that would be parsed from kSampleAccessChainInstructionWords.
spv_parsed_operand_t kSampleAccessChainOperands[] = {
    {1, 1, SPV_OPERAND_TYPE_RESULT_ID, SPV_NUMBER_NONE, 0},
    {2, 1, SPV_OPERAND_TYPE_TYPE_ID, SPV_NUMBER_NONE, 0},
    {3, 1, SPV_OPERAND_TYPE_ID, SPV_NUMBER_NONE, 0},
    {4, 1, SPV_OPERAND_TYPE_ID, SPV_NUMBER_NONE, 0},
    {5, 1, SPV_OPERAND_TYPE_ID, SPV_NUMBER_NONE, 0},
    {6, 1, SPV_OPERAND_TYPE_ID, SPV_NUMBER_NONE, 0},
};

// A valid parse of kSampleAccessChainInstructionWords
spv_parsed_instruction_t kSampleAccessChainInstruction = {
    kSampleAccessChainInstructionWords,
    uint16_t(7),
    uint16_t(SpvOpAccessChain),
    SPV_EXT_INST_TYPE_NONE,
    100,  // type id
    101,  // result id
    kSampleAccessChainOperands,
    6};

// The words for an OpControlBarrier instruction.
uint32_t kSampleControlBarrierInstructionWords[] = {
    (4 << 16) | uint32_t(SpvOpControlBarrier), 100, 101, 102};

// The operands that would be parsed from kSampleControlBarrierInstructionWords.
spv_parsed_operand_t kSampleControlBarrierOperands[] = {
    {1, 1, SPV_OPERAND_TYPE_SCOPE_ID, SPV_NUMBER_NONE, 0},  // Execution
    {2, 1, SPV_OPERAND_TYPE_SCOPE_ID, SPV_NUMBER_NONE, 0},  // Memory
    {3, 1, SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID, SPV_NUMBER_NONE,
     0},  // Semantics
};

// A valid parse of kSampleControlBarrierInstructionWords
spv_parsed_instruction_t kSampleControlBarrierInstruction = {
    kSampleControlBarrierInstructionWords,
    uint16_t(4),
    uint16_t(SpvOpControlBarrier),
    SPV_EXT_INST_TYPE_NONE,
    0,  // type id
    0,  // result id
    kSampleControlBarrierOperands,
    3};

TEST(InstructionTest, CreateWithOpcodeAndOperands) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction inst(&context, kSampleParsedInstruction);
  EXPECT_EQ(SpvOpTypeInt, inst.opcode());
  EXPECT_EQ(0u, inst.type_id());
  EXPECT_EQ(44u, inst.result_id());
  EXPECT_EQ(3u, inst.NumOperands());
  EXPECT_EQ(3u, inst.NumOperandWords());
  EXPECT_EQ(2u, inst.NumInOperandWords());
}

TEST(InstructionTest, GetOperand) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction inst(&context, kSampleParsedInstruction);
  EXPECT_THAT(inst.GetOperand(0).words, Eq(std::vector<uint32_t>{44}));
  EXPECT_THAT(inst.GetOperand(1).words, Eq(std::vector<uint32_t>{32}));
  EXPECT_THAT(inst.GetOperand(2).words, Eq(std::vector<uint32_t>{1}));
}

TEST(InstructionTest, GetInOperand) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction inst(&context, kSampleParsedInstruction);
  EXPECT_THAT(inst.GetInOperand(0).words, Eq(std::vector<uint32_t>{32}));
  EXPECT_THAT(inst.GetInOperand(1).words, Eq(std::vector<uint32_t>{1}));
}

TEST(InstructionTest, OperandConstIterators) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction inst(&context, kSampleParsedInstruction);
  // Spot check iteration across operands.
  auto cbegin = inst.cbegin();
  auto cend = inst.cend();
  EXPECT_NE(cend, inst.cbegin());

  auto citer = inst.cbegin();
  for (int i = 0; i < 3; ++i, ++citer) {
    const auto& operand = *citer;
    EXPECT_THAT(operand.type, Eq(kSampleParsedOperands[i].type));
    EXPECT_THAT(operand.words,
                Eq(std::vector<uint32_t>{kSampleInstructionWords[i + 1]}));
    EXPECT_NE(cend, citer);
  }
  EXPECT_EQ(cend, citer);

  // Check that cbegin and cend have not changed.
  EXPECT_EQ(cbegin, inst.cbegin());
  EXPECT_EQ(cend, inst.cend());

  // Check arithmetic.
  const Operand& operand2 = *(inst.cbegin() + 2);
  EXPECT_EQ(SPV_OPERAND_TYPE_LITERAL_INTEGER, operand2.type);
}

TEST(InstructionTest, OperandIterators) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction inst(&context, kSampleParsedInstruction);
  // Spot check iteration across operands, with mutable iterators.
  auto begin = inst.begin();
  auto end = inst.end();
  EXPECT_NE(end, inst.begin());

  auto iter = inst.begin();
  for (int i = 0; i < 3; ++i, ++iter) {
    const auto& operand = *iter;
    EXPECT_THAT(operand.type, Eq(kSampleParsedOperands[i].type));
    EXPECT_THAT(operand.words,
                Eq(std::vector<uint32_t>{kSampleInstructionWords[i + 1]}));
    EXPECT_NE(end, iter);
  }
  EXPECT_EQ(end, iter);

  // Check that begin and end have not changed.
  EXPECT_EQ(begin, inst.begin());
  EXPECT_EQ(end, inst.end());

  // Check arithmetic.
  Operand& operand2 = *(inst.begin() + 2);
  EXPECT_EQ(SPV_OPERAND_TYPE_LITERAL_INTEGER, operand2.type);

  // Check mutation through an iterator.
  operand2.type = SPV_OPERAND_TYPE_TYPE_ID;
  EXPECT_EQ(SPV_OPERAND_TYPE_TYPE_ID, (*(inst.cbegin() + 2)).type);
}

TEST(InstructionTest, ForInIdStandardIdTypes) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction inst(&context, kSampleAccessChainInstruction);

  std::vector<uint32_t> ids;
  inst.ForEachInId([&ids](const uint32_t* idptr) { ids.push_back(*idptr); });
  EXPECT_THAT(ids, Eq(std::vector<uint32_t>{102, 103, 104, 105}));

  ids.clear();
  inst.ForEachInId([&ids](uint32_t* idptr) { ids.push_back(*idptr); });
  EXPECT_THAT(ids, Eq(std::vector<uint32_t>{102, 103, 104, 105}));
}

TEST(InstructionTest, ForInIdNonstandardIdTypes) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction inst(&context, kSampleControlBarrierInstruction);

  std::vector<uint32_t> ids;
  inst.ForEachInId([&ids](const uint32_t* idptr) { ids.push_back(*idptr); });
  EXPECT_THAT(ids, Eq(std::vector<uint32_t>{100, 101, 102}));

  ids.clear();
  inst.ForEachInId([&ids](uint32_t* idptr) { ids.push_back(*idptr); });
  EXPECT_THAT(ids, Eq(std::vector<uint32_t>{100, 101, 102}));
}

TEST(InstructionTest, UniqueIds) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction inst1(&context);
  Instruction inst2(&context);
  EXPECT_NE(inst1.unique_id(), inst2.unique_id());
}

TEST(InstructionTest, CloneUniqueIdDifferent) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction inst(&context);
  std::unique_ptr<Instruction> clone(inst.Clone(&context));
  EXPECT_EQ(inst.context(), clone->context());
  EXPECT_NE(inst.unique_id(), clone->unique_id());
}

TEST(InstructionTest, CloneDifferentContext) {
  IRContext c1(SPV_ENV_UNIVERSAL_1_2, nullptr);
  IRContext c2(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction inst(&c1);
  std::unique_ptr<Instruction> clone(inst.Clone(&c2));
  EXPECT_EQ(&c1, inst.context());
  EXPECT_EQ(&c2, clone->context());
  EXPECT_NE(&c1, &c2);
}

TEST(InstructionTest, CloneDifferentContextDifferentUniqueId) {
  IRContext c1(SPV_ENV_UNIVERSAL_1_2, nullptr);
  IRContext c2(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction inst(&c1);
  Instruction other(&c2);
  std::unique_ptr<Instruction> clone(inst.Clone(&c2));
  EXPECT_EQ(&c2, clone->context());
  EXPECT_NE(other.unique_id(), clone->unique_id());
}

TEST(InstructionTest, EqualsEqualsOperator) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction i1(&context);
  Instruction i2(&context);
  std::unique_ptr<Instruction> clone(i1.Clone(&context));
  EXPECT_TRUE(i1 == i1);
  EXPECT_FALSE(i1 == i2);
  EXPECT_FALSE(i1 == *clone);
  EXPECT_FALSE(i2 == *clone);
}

TEST(InstructionTest, LessThanOperator) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);
  Instruction i1(&context);
  Instruction i2(&context);
  std::unique_ptr<Instruction> clone(i1.Clone(&context));
  EXPECT_TRUE(i1 < i2);
  EXPECT_TRUE(i1 < *clone);
  EXPECT_TRUE(i2 < *clone);
}

TEST_F(DescriptorTypeTest, StorageImage) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeImage %6 2D 0 0 0 2 R32f
          %8 = OpTypePointer UniformConstant %7
          %3 = OpVariable %8 UniformConstant
          %2 = OpFunction %4 None %5
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* type = context->get_def_use_mgr()->GetDef(8);
  EXPECT_TRUE(type->IsVulkanStorageImage());
  EXPECT_FALSE(type->IsVulkanSampledImage());
  EXPECT_FALSE(type->IsVulkanStorageTexelBuffer());
  EXPECT_FALSE(type->IsVulkanStorageBuffer());
  EXPECT_FALSE(type->IsVulkanUniformBuffer());

  Instruction* variable = context->get_def_use_mgr()->GetDef(3);
  EXPECT_FALSE(variable->IsReadOnlyVariable());
}

TEST_F(DescriptorTypeTest, SampledImage) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeImage %6 2D 0 0 0 1 Unknown
          %8 = OpTypePointer UniformConstant %7
          %3 = OpVariable %8 UniformConstant
          %2 = OpFunction %4 None %5
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* type = context->get_def_use_mgr()->GetDef(8);
  EXPECT_FALSE(type->IsVulkanStorageImage());
  EXPECT_TRUE(type->IsVulkanSampledImage());
  EXPECT_FALSE(type->IsVulkanStorageTexelBuffer());
  EXPECT_FALSE(type->IsVulkanStorageBuffer());
  EXPECT_FALSE(type->IsVulkanUniformBuffer());

  Instruction* variable = context->get_def_use_mgr()->GetDef(3);
  EXPECT_TRUE(variable->IsReadOnlyVariable());
}

TEST_F(DescriptorTypeTest, StorageTexelBuffer) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeImage %6 Buffer 0 0 0 2 R32f
          %8 = OpTypePointer UniformConstant %7
          %3 = OpVariable %8 UniformConstant
          %2 = OpFunction %4 None %5
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* type = context->get_def_use_mgr()->GetDef(8);
  EXPECT_FALSE(type->IsVulkanStorageImage());
  EXPECT_FALSE(type->IsVulkanSampledImage());
  EXPECT_TRUE(type->IsVulkanStorageTexelBuffer());
  EXPECT_FALSE(type->IsVulkanStorageBuffer());
  EXPECT_FALSE(type->IsVulkanUniformBuffer());

  Instruction* variable = context->get_def_use_mgr()->GetDef(3);
  EXPECT_FALSE(variable->IsReadOnlyVariable());
}

TEST_F(DescriptorTypeTest, StorageBuffer) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
               OpDecorate %9 BufferBlock
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypeRuntimeArray %7
          %9 = OpTypeStruct %8
         %10 = OpTypePointer Uniform %9
          %3 = OpVariable %10 Uniform
          %2 = OpFunction %4 None %5
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* type = context->get_def_use_mgr()->GetDef(10);
  EXPECT_FALSE(type->IsVulkanStorageImage());
  EXPECT_FALSE(type->IsVulkanSampledImage());
  EXPECT_FALSE(type->IsVulkanStorageTexelBuffer());
  EXPECT_TRUE(type->IsVulkanStorageBuffer());
  EXPECT_FALSE(type->IsVulkanUniformBuffer());

  Instruction* variable = context->get_def_use_mgr()->GetDef(3);
  EXPECT_FALSE(variable->IsReadOnlyVariable());
}

TEST_F(DescriptorTypeTest, UniformBuffer) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
               OpDecorate %9 Block
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypeRuntimeArray %7
          %9 = OpTypeStruct %8
         %10 = OpTypePointer Uniform %9
          %3 = OpVariable %10 Uniform
          %2 = OpFunction %4 None %5
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* type = context->get_def_use_mgr()->GetDef(10);
  EXPECT_FALSE(type->IsVulkanStorageImage());
  EXPECT_FALSE(type->IsVulkanSampledImage());
  EXPECT_FALSE(type->IsVulkanStorageTexelBuffer());
  EXPECT_FALSE(type->IsVulkanStorageBuffer());
  EXPECT_TRUE(type->IsVulkanUniformBuffer());

  Instruction* variable = context->get_def_use_mgr()->GetDef(3);
  EXPECT_TRUE(variable->IsReadOnlyVariable());
}

TEST_F(DescriptorTypeTest, NonWritableIsReadOnly) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
               OpDecorate %9 BufferBlock
               OpDecorate %3 NonWritable
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypeRuntimeArray %7
          %9 = OpTypeStruct %8
         %10 = OpTypePointer Uniform %9
          %3 = OpVariable %10 Uniform
          %2 = OpFunction %4 None %5
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* variable = context->get_def_use_mgr()->GetDef(3);
  EXPECT_TRUE(variable->IsReadOnlyVariable());
}

TEST_F(OpaqueTypeTest, BaseOpaqueTypesShader) {
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
          %6 = OpTypeImage %5 2D 1 0 0 1 Unknown
          %7 = OpTypeSampler
          %8 = OpTypeSampledImage %6
          %9 = OpTypeRuntimeArray %5
          %2 = OpFunction %3 None %4
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* image_type = context->get_def_use_mgr()->GetDef(6);
  EXPECT_TRUE(image_type->IsOpaqueType());
  Instruction* sampler_type = context->get_def_use_mgr()->GetDef(7);
  EXPECT_TRUE(sampler_type->IsOpaqueType());
  Instruction* sampled_image_type = context->get_def_use_mgr()->GetDef(8);
  EXPECT_TRUE(sampled_image_type->IsOpaqueType());
  Instruction* runtime_array_type = context->get_def_use_mgr()->GetDef(9);
  EXPECT_TRUE(runtime_array_type->IsOpaqueType());
  Instruction* float_type = context->get_def_use_mgr()->GetDef(5);
  EXPECT_FALSE(float_type->IsOpaqueType());
  Instruction* void_type = context->get_def_use_mgr()->GetDef(3);
  EXPECT_FALSE(void_type->IsOpaqueType());
}

TEST_F(OpaqueTypeTest, OpaqueStructTypes) {
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
          %6 = OpTypeRuntimeArray %5
          %7 = OpTypeStruct %6 %6
          %8 = OpTypeStruct %5 %6
          %9 = OpTypeStruct %6 %5
         %10 = OpTypeStruct %7
          %2 = OpFunction %3 None %4
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  for (int i = 7; i <= 10; i++) {
    Instruction* type = context->get_def_use_mgr()->GetDef(i);
    EXPECT_TRUE(type->IsOpaqueType());
  }
}

TEST_F(GetBaseTest, SampleImage) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 2
          %8 = OpTypeVector %6 4
          %9 = OpConstant %6 0
         %10 = OpConstantComposite %7 %9 %9
         %11 = OpTypeImage %6 2D 0 0 0 1 R32f
         %12 = OpTypePointer UniformConstant %11
          %3 = OpVariable %12 UniformConstant
         %13 = OpTypeSampledImage %11
         %14 = OpTypeSampler
         %15 = OpTypePointer UniformConstant %14
         %16 = OpVariable %15 UniformConstant
          %2 = OpFunction %4 None %5
         %17 = OpLabel
         %18 = OpLoad %11 %3
         %19 = OpLoad %14 %16
         %20 = OpSampledImage %13 %18 %19
         %21 = OpImageSampleImplicitLod %8 %20 %10
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* load = context->get_def_use_mgr()->GetDef(21);
  Instruction* base = context->get_def_use_mgr()->GetDef(20);
  EXPECT_TRUE(load->GetBaseAddress() == base);
}

TEST_F(GetBaseTest, PtrAccessChain) {
  const std::string text = R"(
               OpCapability VariablePointers
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "PSMain" %2
               OpExecutionMode %1 OriginUpperLeft
       %void = OpTypeVoid
          %4 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
        %int = OpTypeInt 32 8388353
      %int_0 = OpConstant %int 0
%_ptr_Function_v4float = OpTypePointer Function %v4float
          %2 = OpVariable %_ptr_Function_v4float Input
          %1 = OpFunction %void None %4
         %10 = OpLabel
         %11 = OpPtrAccessChain %_ptr_Function_v4float %2 %int_0
         %12 = OpLoad %v4float %11
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* load = context->get_def_use_mgr()->GetDef(12);
  Instruction* base = context->get_def_use_mgr()->GetDef(2);
  EXPECT_TRUE(load->GetBaseAddress() == base);
}

TEST_F(GetBaseTest, ImageRead) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeInt 32 0
          %7 = OpTypeVector %6 2
          %8 = OpConstant %6 0
          %9 = OpConstantComposite %7 %8 %8
         %10 = OpTypeImage %6 2D 0 0 0 2 R32f
         %11 = OpTypePointer UniformConstant %10
          %3 = OpVariable %11 UniformConstant
          %2 = OpFunction %4 None %5
         %12 = OpLabel
         %13 = OpLoad %10 %3
         %14 = OpImageRead %6 %13 %9
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* load = context->get_def_use_mgr()->GetDef(14);
  Instruction* base = context->get_def_use_mgr()->GetDef(13);
  EXPECT_TRUE(load->GetBaseAddress() == base);
}

TEST_F(ValidBasePointerTest, OpSelectBadNoVariablePointersStorageBuffer) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer StorageBuffer %3
%5 = OpVariable %4 StorageBuffer
%6 = OpTypeFunction %2
%7 = OpTypeBool
%8 = OpConstantTrue %7
%1 = OpFunction %2 None %6
%9 = OpLabel
%10 = OpSelect %4 %8 %5 %5
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* select = context->get_def_use_mgr()->GetDef(10);
  EXPECT_NE(select, nullptr);
  EXPECT_FALSE(select->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpSelectBadNoVariablePointers) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointersStorageBuffer
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer Workgroup %3
%5 = OpVariable %4 Workgroup
%6 = OpTypeFunction %2
%7 = OpTypeBool
%8 = OpConstantTrue %7
%1 = OpFunction %2 None %6
%9 = OpLabel
%10 = OpSelect %4 %8 %5 %5
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* select = context->get_def_use_mgr()->GetDef(10);
  EXPECT_NE(select, nullptr);
  EXPECT_FALSE(select->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpSelectGoodVariablePointersStorageBuffer) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointersStorageBuffer
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer StorageBuffer %3
%5 = OpVariable %4 StorageBuffer
%6 = OpTypeFunction %2
%7 = OpTypeBool
%8 = OpConstantTrue %7
%1 = OpFunction %2 None %6
%9 = OpLabel
%10 = OpSelect %4 %8 %5 %5
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* select = context->get_def_use_mgr()->GetDef(10);
  EXPECT_NE(select, nullptr);
  EXPECT_TRUE(select->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpSelectGoodVariablePointers) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer Workgroup %3
%5 = OpVariable %4 Workgroup
%6 = OpTypeFunction %2
%7 = OpTypeBool
%8 = OpConstantTrue %7
%1 = OpFunction %2 None %6
%9 = OpLabel
%10 = OpSelect %4 %8 %5 %5
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* select = context->get_def_use_mgr()->GetDef(10);
  EXPECT_NE(select, nullptr);
  EXPECT_TRUE(select->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpConstantNullBadNoVariablePointersStorageBuffer) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer StorageBuffer %3
%5 = OpConstantNull %4
%6 = OpTypeFunction %2
%1 = OpFunction %2 None %6
%7 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* null_inst = context->get_def_use_mgr()->GetDef(5);
  EXPECT_NE(null_inst, nullptr);
  EXPECT_FALSE(null_inst->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpConstantNullBadNoVariablePointers) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointersStorageBuffer
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer Workgroup %3
%5 = OpConstantNull %4
%6 = OpTypeFunction %2
%1 = OpFunction %2 None %6
%7 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* null_inst = context->get_def_use_mgr()->GetDef(5);
  EXPECT_NE(null_inst, nullptr);
  EXPECT_FALSE(null_inst->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpConstantNullGoodVariablePointersStorageBuffer) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointersStorageBuffer
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer StorageBuffer %3
%5 = OpConstantNull %4
%6 = OpTypeFunction %2
%1 = OpFunction %2 None %6
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* null_inst = context->get_def_use_mgr()->GetDef(5);
  EXPECT_NE(null_inst, nullptr);
  EXPECT_TRUE(null_inst->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpConstantNullGoodVariablePointers) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer Workgroup %3
%5 = OpConstantNull %4
%6 = OpTypeFunction %2
%1 = OpFunction %2 None %6
%7 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* null_inst = context->get_def_use_mgr()->GetDef(5);
  EXPECT_NE(null_inst, nullptr);
  EXPECT_TRUE(null_inst->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpPhiBadNoVariablePointersStorageBuffer) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer StorageBuffer %3
%5 = OpVariable %4 StorageBuffer
%6 = OpTypeFunction %2
%1 = OpFunction %2 None %6
%7 = OpLabel
OpBranch %8
%8 = OpLabel
%9 = OpPhi %4 %5 %7
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* phi = context->get_def_use_mgr()->GetDef(9);
  EXPECT_NE(phi, nullptr);
  EXPECT_FALSE(phi->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpPhiBadNoVariablePointers) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointersStorageBuffer
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer Workgroup %3
%5 = OpVariable %4 Workgroup
%6 = OpTypeFunction %2
%1 = OpFunction %2 None %6
%7 = OpLabel
OpBranch %8
%8 = OpLabel
%9 = OpPhi %4 %5 %7
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* phi = context->get_def_use_mgr()->GetDef(9);
  EXPECT_NE(phi, nullptr);
  EXPECT_FALSE(phi->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpPhiGoodVariablePointersStorageBuffer) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointersStorageBuffer
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer StorageBuffer %3
%5 = OpVariable %4 StorageBuffer
%6 = OpTypeFunction %2
%1 = OpFunction %2 None %6
%7 = OpLabel
OpBranch %8
%8 = OpLabel
%9 = OpPhi %4 %5 %7
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* phi = context->get_def_use_mgr()->GetDef(9);
  EXPECT_NE(phi, nullptr);
  EXPECT_TRUE(phi->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpPhiGoodVariablePointers) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer Workgroup %3
%5 = OpVariable %4 Workgroup
%6 = OpTypeFunction %2
%1 = OpFunction %2 None %6
%7 = OpLabel
OpBranch %8
%8 = OpLabel
%9 = OpPhi %4 %5 %7
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* phi = context->get_def_use_mgr()->GetDef(9);
  EXPECT_NE(phi, nullptr);
  EXPECT_TRUE(phi->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpFunctionCallBadNoVariablePointersStorageBuffer) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer StorageBuffer %3
%5 = OpConstantNull %4
%6 = OpTypeFunction %2
%7 = OpTypeFunction %4
%1 = OpFunction %2 None %6
%8 = OpLabel
%9 = OpFunctionCall %4 %10
OpReturn
OpFunctionEnd
%10 = OpFunction %4 None %7
%11 = OpLabel
OpReturnValue %5
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* null_inst = context->get_def_use_mgr()->GetDef(9);
  EXPECT_NE(null_inst, nullptr);
  EXPECT_FALSE(null_inst->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpFunctionCallBadNoVariablePointers) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointersStorageBuffer
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer Workgroup %3
%5 = OpConstantNull %4
%6 = OpTypeFunction %2
%7 = OpTypeFunction %4
%1 = OpFunction %2 None %6
%8 = OpLabel
%9 = OpFunctionCall %4 %10
OpReturn
OpFunctionEnd
%10 = OpFunction %4 None %7
%11 = OpLabel
OpReturnValue %5
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* null_inst = context->get_def_use_mgr()->GetDef(9);
  EXPECT_NE(null_inst, nullptr);
  EXPECT_FALSE(null_inst->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpFunctionCallGoodVariablePointersStorageBuffer) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointersStorageBuffer
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer StorageBuffer %3
%5 = OpConstantNull %4
%6 = OpTypeFunction %2
%7 = OpTypeFunction %4
%1 = OpFunction %2 None %6
%8 = OpLabel
%9 = OpFunctionCall %4 %10
OpReturn
OpFunctionEnd
%10 = OpFunction %4 None %7
%11 = OpLabel
OpReturnValue %5
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* null_inst = context->get_def_use_mgr()->GetDef(9);
  EXPECT_NE(null_inst, nullptr);
  EXPECT_TRUE(null_inst->IsValidBasePointer());
}

TEST_F(ValidBasePointerTest, OpFunctionCallGoodVariablePointers) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer Workgroup %3
%5 = OpConstantNull %4
%6 = OpTypeFunction %2
%7 = OpTypeFunction %4
%1 = OpFunction %2 None %6
%8 = OpLabel
%9 = OpFunctionCall %4 %10
OpReturn
OpFunctionEnd
%10 = OpFunction %4 None %7
%11 = OpLabel
OpReturnValue %5
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, text);
  EXPECT_NE(context, nullptr);
  Instruction* null_inst = context->get_def_use_mgr()->GetDef(9);
  EXPECT_NE(null_inst, nullptr);
  EXPECT_TRUE(null_inst->IsValidBasePointer());
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
