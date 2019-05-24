// Copyright (c) 2017 Pierre Moreau
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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/opt/build_module.h"
#include "source/opt/decoration_manager.h"
#include "source/opt/ir_context.h"
#include "source/spirv_constant.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace opt {
namespace analysis {
namespace {

using spvtest::MakeVector;

class DecorationManagerTest : public ::testing::Test {
 public:
  DecorationManagerTest()
      : tools_(SPV_ENV_UNIVERSAL_1_2),
        context_(),
        consumer_([this](spv_message_level_t level, const char*,
                         const spv_position_t& position, const char* message) {
          if (!error_message_.empty()) error_message_ += "\n";
          switch (level) {
            case SPV_MSG_FATAL:
            case SPV_MSG_INTERNAL_ERROR:
            case SPV_MSG_ERROR:
              error_message_ += "ERROR";
              break;
            case SPV_MSG_WARNING:
              error_message_ += "WARNING";
              break;
            case SPV_MSG_INFO:
              error_message_ += "INFO";
              break;
            case SPV_MSG_DEBUG:
              error_message_ += "DEBUG";
              break;
          }
          error_message_ +=
              ": " + std::to_string(position.index) + ": " + message;
        }),
        disassemble_options_(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER),
        error_message_() {
    tools_.SetMessageConsumer(consumer_);
  }

  void TearDown() override { error_message_.clear(); }

  DecorationManager* GetDecorationManager(const std::string& text) {
    context_ = BuildModule(SPV_ENV_UNIVERSAL_1_2, consumer_, text);
    if (context_.get())
      return context_->get_decoration_mgr();
    else
      return nullptr;
  }

  // Disassembles |binary| and outputs the result in |text|. If |text| is a
  // null pointer, SPV_ERROR_INVALID_POINTER is returned.
  spv_result_t Disassemble(const std::vector<uint32_t>& binary,
                           std::string* text) {
    if (!text) return SPV_ERROR_INVALID_POINTER;
    return tools_.Disassemble(binary, text, disassemble_options_)
               ? SPV_SUCCESS
               : SPV_ERROR_INVALID_BINARY;
  }

  // Returns the accumulated error messages for the test.
  std::string GetErrorMessage() const { return error_message_; }

  std::string ToText(const std::vector<Instruction*>& inst) {
    std::vector<uint32_t> binary = {SpvMagicNumber, 0x10200, 0u, 2u, 0u};
    for (const Instruction* i : inst)
      i->ToBinaryWithoutAttachedDebugInsts(&binary);
    std::string text;
    Disassemble(binary, &text);
    return text;
  }

  std::string ModuleToText() {
    std::vector<uint32_t> binary;
    context_->module()->ToBinary(&binary, false);
    std::string text;
    Disassemble(binary, &text);
    return text;
  }

  spvtools::MessageConsumer GetConsumer() { return consumer_; }

 private:
  // An instance for calling SPIRV-Tools functionalities.
  spvtools::SpirvTools tools_;
  std::unique_ptr<IRContext> context_;
  spvtools::MessageConsumer consumer_;
  uint32_t disassemble_options_;
  std::string error_message_;
};

TEST_F(DecorationManagerTest,
       ComparingDecorationsWithDiffOpcodesDecorateDecorateId) {
  IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // This parameter can be interprated both as { SpvDecorationConstant }
  // and also as a list of IDs:  { 22 }
  const std::vector<uint32_t> param{SpvDecorationConstant};
  // OpDecorate %1 Constant
  Instruction inst1(
      &ir_context, SpvOpDecorate, 0u, 0u,
      {{SPV_OPERAND_TYPE_ID, {1u}}, {SPV_OPERAND_TYPE_DECORATION, param}});
  // OpDecorateId %1 %22   ; 'Constant' is decoration number 22
  Instruction inst2(
      &ir_context, SpvOpDecorateId, 0u, 0u,
      {{SPV_OPERAND_TYPE_ID, {1u}}, {SPV_OPERAND_TYPE_ID, param}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest,
       ComparingDecorationsWithDiffOpcodesDecorateDecorateString) {
  IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // This parameter can be interprated both as { SpvDecorationConstant }
  // and also as a null-terminated string with a single character with value 22.
  const std::vector<uint32_t> param{SpvDecorationConstant};
  // OpDecorate %1 Constant
  Instruction inst1(
      &ir_context, SpvOpDecorate, 0u, 0u,
      {{SPV_OPERAND_TYPE_ID, {1u}}, {SPV_OPERAND_TYPE_DECORATION, param}});
  // OpDecorateStringGOOGLE %1 !22
  Instruction inst2(
      &ir_context, SpvOpDecorateStringGOOGLE, 0u, 0u,
      {{SPV_OPERAND_TYPE_ID, {1u}}, {SPV_OPERAND_TYPE_LITERAL_STRING, param}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest, ComparingDecorationsWithDiffDecorateParam) {
  IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpDecorate %1 Constant
  Instruction inst1(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpDecorate %1 Restrict
  Instruction inst2(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationRestrict}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest, ComparingDecorationsWithDiffDecorateIdParam) {
  IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpDecorate %1 Constant
  Instruction inst1(
      &ir_context, SpvOpDecorateId, 0u, 0u,
      {{SPV_OPERAND_TYPE_ID, {1u}}, {SPV_OPERAND_TYPE_ID, {555}}});
  // OpDecorate %1 Restrict
  Instruction inst2(
      &ir_context, SpvOpDecorateId, 0u, 0u,
      {{SPV_OPERAND_TYPE_ID, {1u}}, {SPV_OPERAND_TYPE_ID, {666}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest, ComparingDecorationsWithDiffDecorateStringParam) {
  IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpDecorate %1 Constant
  Instruction inst1(&ir_context, SpvOpDecorateStringGOOGLE, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_LITERAL_STRING, MakeVector("Hello!")}});
  // OpDecorate %1 Restrict
  Instruction inst2(&ir_context, SpvOpDecorateStringGOOGLE, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_LITERAL_STRING, MakeVector("Hellx")}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest, ComparingSameDecorationsOnDiffTargetAllowed) {
  IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpDecorate %1 Constant
  Instruction inst1(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpDecorate %2 Constant
  Instruction inst2(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {2u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest, ComparingSameDecorationIdsOnDiffTargetAllowed) {
  IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  Instruction inst1(
      &ir_context, SpvOpDecorateId, 0u, 0u,
      {{SPV_OPERAND_TYPE_ID, {1u}}, {SPV_OPERAND_TYPE_DECORATION, {44}}});
  Instruction inst2(
      &ir_context, SpvOpDecorateId, 0u, 0u,
      {{SPV_OPERAND_TYPE_ID, {2u}}, {SPV_OPERAND_TYPE_DECORATION, {44}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest,
       ComparingSameDecorationStringsOnDiffTargetAllowed) {
  IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  Instruction inst1(&ir_context, SpvOpDecorateStringGOOGLE, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_LITERAL_STRING, MakeVector("hello")}});
  Instruction inst2(&ir_context, SpvOpDecorateStringGOOGLE, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {2u}},
                     {SPV_OPERAND_TYPE_LITERAL_STRING, MakeVector("hello")}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest, ComparingSameDecorationsOnDiffTargetDisallowed) {
  IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpDecorate %1 Constant
  Instruction inst1(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpDecorate %2 Constant
  Instruction inst2(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {2u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, false));
}

TEST_F(DecorationManagerTest, ComparingMemberDecorationsOnSameTypeDiffMember) {
  IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpMemberDecorate %1 0 Constant
  Instruction inst1(&ir_context, SpvOpMemberDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpMemberDecorate %1 1 Constant
  Instruction inst2(&ir_context, SpvOpMemberDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {1u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest,
       ComparingSameMemberDecorationsOnDiffTargetAllowed) {
  IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpMemberDecorate %1 0 Constant
  Instruction inst1(&ir_context, SpvOpMemberDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpMemberDecorate %2 0 Constant
  Instruction inst2(&ir_context, SpvOpMemberDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {2u}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest,
       ComparingSameMemberDecorationsOnDiffTargetDisallowed) {
  IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpMemberDecorate %1 0 Constant
  Instruction inst1(&ir_context, SpvOpMemberDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpMemberDecorate %2 0 Constant
  Instruction inst2(&ir_context, SpvOpMemberDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {2u}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, false));
}

TEST_F(DecorationManagerTest, RemoveDecorationFromVariable) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %2 Restrict
%2      = OpDecorationGroup
OpGroupDecorate %2 %1 %3
%4   = OpTypeInt 32 0
%1      = OpVariable %4 Uniform
%3      = OpVariable %4 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  decoManager->RemoveDecorationsFrom(1u);
  auto decorations = decoManager->GetDecorationsFor(1u, false);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decorations.empty());
  decorations = decoManager->GetDecorationsFor(3u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  const std::string expected_decorations = R"(OpDecorate %2 Restrict
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);

  const std::string expected_binary = R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %2 Restrict
%2 = OpDecorationGroup
OpGroupDecorate %2 %3
%4 = OpTypeInt 32 0
%1 = OpVariable %4 Uniform
%3 = OpVariable %4 Uniform
)";
  EXPECT_THAT(ModuleToText(), expected_binary);
}

TEST_F(DecorationManagerTest, RemoveDecorationStringFromVariable) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "hello world"
OpDecorate %2 Restrict
%2      = OpDecorationGroup
OpGroupDecorate %2 %1 %3
%4   = OpTypeInt 32 0
%1      = OpVariable %4 Uniform
%3      = OpVariable %4 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  decoManager->RemoveDecorationsFrom(1u);
  auto decorations = decoManager->GetDecorationsFor(1u, false);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decorations.empty());
  decorations = decoManager->GetDecorationsFor(3u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  const std::string expected_decorations = R"(OpDecorate %2 Restrict
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);

  const std::string expected_binary = R"(OpCapability Shader
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorate %2 Restrict
%2 = OpDecorationGroup
OpGroupDecorate %2 %3
%4 = OpTypeInt 32 0
%1 = OpVariable %4 Uniform
%3 = OpVariable %4 Uniform
)";
  EXPECT_THAT(ModuleToText(), expected_binary);
}

TEST_F(DecorationManagerTest, RemoveDecorationFromDecorationGroup) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %2 Restrict
%2      = OpDecorationGroup
OpGroupDecorate %2 %1 %3
%4   = OpTypeInt 32 0
%1      = OpVariable %4 Uniform
%3      = OpVariable %4 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  decoManager->RemoveDecorationsFrom(2u);
  auto decorations = decoManager->GetDecorationsFor(2u, false);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decorations.empty());
  decorations = decoManager->GetDecorationsFor(1u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  const std::string expected_decorations = R"(OpDecorate %1 Constant
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);
  decorations = decoManager->GetDecorationsFor(3u, false);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_THAT(ToText(decorations), "");

  const std::string expected_binary = R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
%2 = OpDecorationGroup
%4 = OpTypeInt 32 0
%1 = OpVariable %4 Uniform
%3 = OpVariable %4 Uniform
)";
  EXPECT_THAT(ModuleToText(), expected_binary);
}

TEST_F(DecorationManagerTest,
       RemoveDecorationFromDecorationGroupKeepDeadDecorations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %2 Restrict
%2      = OpDecorationGroup
OpGroupDecorate %2 %1
%3   = OpTypeInt 32 0
%1      = OpVariable %3 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  decoManager->RemoveDecorationsFrom(1u);
  auto decorations = decoManager->GetDecorationsFor(1u, false);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decorations.empty());
  decorations = decoManager->GetDecorationsFor(2u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  const std::string expected_decorations = R"(OpDecorate %2 Restrict
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);

  const std::string expected_binary = R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %2 Restrict
%2 = OpDecorationGroup
%3 = OpTypeInt 32 0
%1 = OpVariable %3 Uniform
)";
  EXPECT_THAT(ModuleToText(), expected_binary);
}

TEST_F(DecorationManagerTest, RemoveAllDecorationsAppliedByGroup) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %2 Restrict
%2      = OpDecorationGroup
OpGroupDecorate %2 %1
OpDecorate %3 BuiltIn VertexId
%3      = OpDecorationGroup
OpGroupDecorate %3 %1
%4      = OpTypeInt 32 0
%1      = OpVariable %4 Input
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  decoManager->RemoveDecorationsFrom(1u, [](const Instruction& inst) {
    return inst.opcode() == SpvOpDecorate &&
           inst.GetSingleWordInOperand(0u) == 3u;
  });
  auto decorations = decoManager->GetDecorationsFor(1u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  std::string expected_decorations = R"(OpDecorate %1 Constant
OpDecorate %2 Restrict
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);
  decorations = decoManager->GetDecorationsFor(2u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  expected_decorations = R"(OpDecorate %2 Restrict
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);

  const std::string expected_binary = R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %2 Restrict
%2 = OpDecorationGroup
OpGroupDecorate %2 %1
OpDecorate %3 BuiltIn VertexId
%3 = OpDecorationGroup
%4 = OpTypeInt 32 0
%1 = OpVariable %4 Input
)";
  EXPECT_THAT(ModuleToText(), expected_binary);
}

TEST_F(DecorationManagerTest, RemoveSomeDecorationsAppliedByGroup) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %2 Restrict
%2      = OpDecorationGroup
OpGroupDecorate %2 %1
OpDecorate %3 BuiltIn VertexId
OpDecorate %3 Invariant
%3      = OpDecorationGroup
OpGroupDecorate %3 %1
%uint   = OpTypeInt 32 0
%1      = OpVariable %uint Input
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  decoManager->RemoveDecorationsFrom(1u, [](const Instruction& inst) {
    return inst.opcode() == SpvOpDecorate &&
           inst.GetSingleWordInOperand(0u) == 3u &&
           inst.GetSingleWordInOperand(1u) == SpvDecorationBuiltIn;
  });
  auto decorations = decoManager->GetDecorationsFor(1u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  std::string expected_decorations = R"(OpDecorate %1 Constant
OpDecorate %1 Invariant
OpDecorate %2 Restrict
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);
  decorations = decoManager->GetDecorationsFor(2u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  expected_decorations = R"(OpDecorate %2 Restrict
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);

  const std::string expected_binary = R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %2 Restrict
%2 = OpDecorationGroup
OpGroupDecorate %2 %1
OpDecorate %3 BuiltIn VertexId
OpDecorate %3 Invariant
%3 = OpDecorationGroup
OpDecorate %1 Invariant
%4 = OpTypeInt 32 0
%1 = OpVariable %4 Input
)";
  EXPECT_THAT(ModuleToText(), expected_binary);
}

TEST_F(DecorationManagerTest, RemoveDecorationDecorate) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %1 Restrict
%2    = OpTypeInt 32 0
%1    = OpVariable %2 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  auto decorations = decoManager->GetDecorationsFor(1u, false);
  decoManager->RemoveDecoration(decorations.front());
  decorations = decoManager->GetDecorationsFor(1u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  const std::string expected_decorations = R"(OpDecorate %1 Restrict
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);
}

TEST_F(DecorationManagerTest, RemoveDecorationStringDecorate) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "foobar"
OpDecorate %1 Restrict
%2    = OpTypeInt 32 0
%1    = OpVariable %2 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  auto decorations = decoManager->GetDecorationsFor(1u, false);
  decoManager->RemoveDecoration(decorations.front());
  decorations = decoManager->GetDecorationsFor(1u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  const std::string expected_decorations = R"(OpDecorate %1 Restrict
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);
}

TEST_F(DecorationManagerTest, CloneDecorations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %2 Restrict
%2      = OpDecorationGroup
OpGroupDecorate %2 %1
OpDecorate %3 BuiltIn VertexId
OpDecorate %3 Invariant
%3      = OpDecorationGroup
OpGroupDecorate %3 %1
%4      = OpTypeInt 32 0
%1      = OpVariable %4 Input
%5      = OpVariable %4 Input
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");

  // Check cloning OpDecorate including group decorations.
  auto decorations = decoManager->GetDecorationsFor(5u, false);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decorations.empty());

  decoManager->CloneDecorations(1u, 5u);
  decorations = decoManager->GetDecorationsFor(5u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  std::string expected_decorations = R"(OpDecorate %5 Constant
OpDecorate %2 Restrict
OpDecorate %3 BuiltIn VertexId
OpDecorate %3 Invariant
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);

  // Check that bookkeeping for ID 2 remains the same.
  decorations = decoManager->GetDecorationsFor(2u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  expected_decorations = R"(OpDecorate %2 Restrict
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);

  const std::string expected_binary = R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %2 Restrict
%2 = OpDecorationGroup
OpGroupDecorate %2 %1 %5
OpDecorate %3 BuiltIn VertexId
OpDecorate %3 Invariant
%3 = OpDecorationGroup
OpGroupDecorate %3 %1 %5
OpDecorate %5 Constant
%4 = OpTypeInt 32 0
%1 = OpVariable %4 Input
%5 = OpVariable %4 Input
)";
  EXPECT_THAT(ModuleToText(), expected_binary);
}

TEST_F(DecorationManagerTest, CloneDecorationsStringAndId) {
  const std::string spirv = R"(OpCapability Shader
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "blah"
OpDecorateId %1 HlslCounterBufferGOOGLE %2
OpDecorate %1 Aliased
%3      = OpTypeInt 32 0
%4      = OpTypePointer Uniform %3
%1      = OpVariable %4 Uniform
%2      = OpVariable %4 Uniform
%5      = OpVariable %4 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");

  // Check cloning OpDecorate including group decorations.
  auto decorations = decoManager->GetDecorationsFor(5u, false);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decorations.empty());

  decoManager->CloneDecorations(1u, 5u);
  decorations = decoManager->GetDecorationsFor(5u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  std::string expected_decorations =
      R"(OpDecorateStringGOOGLE %5 HlslSemanticGOOGLE "blah"
OpDecorateId %5 HlslCounterBufferGOOGLE %2
OpDecorate %5 Aliased
)";
  EXPECT_THAT(ToText(decorations), expected_decorations);

  const std::string expected_binary = R"(OpCapability Shader
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "blah"
OpDecorateId %1 HlslCounterBufferGOOGLE %2
OpDecorate %1 Aliased
OpDecorateStringGOOGLE %5 HlslSemanticGOOGLE "blah"
OpDecorateId %5 HlslCounterBufferGOOGLE %2
OpDecorate %5 Aliased
%3 = OpTypeInt 32 0
%4 = OpTypePointer Uniform %3
%1 = OpVariable %4 Uniform
%2 = OpVariable %4 Uniform
%5 = OpVariable %4 Uniform
)";
  EXPECT_THAT(ModuleToText(), expected_binary);
}

TEST_F(DecorationManagerTest, CloneSomeDecorations) {
  const std::string spirv = R"(OpCapability Shader
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorate %1 RelaxedPrecision
OpDecorate %1 Restrict
%2 = OpTypeInt 32 0
%3 = OpTypePointer Function %2
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpFunction %4 None %5
%7 = OpLabel
%1 = OpVariable %3 Function
%8 = OpUndef %2
OpReturn
OpFunctionEnd
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_EQ(GetErrorMessage(), "");

  // Check cloning OpDecorate including group decorations.
  auto decorations = decoManager->GetDecorationsFor(8u, false);
  EXPECT_EQ(GetErrorMessage(), "");
  EXPECT_TRUE(decorations.empty());

  decoManager->CloneDecorations(1u, 8u, {SpvDecorationRelaxedPrecision});
  decorations = decoManager->GetDecorationsFor(8u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  std::string expected_decorations =
      R"(OpDecorate %8 RelaxedPrecision
)";
  EXPECT_EQ(ToText(decorations), expected_decorations);

  const std::string expected_binary = R"(OpCapability Shader
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorate %1 RelaxedPrecision
OpDecorate %1 Restrict
OpDecorate %8 RelaxedPrecision
%2 = OpTypeInt 32 0
%3 = OpTypePointer Function %2
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpFunction %4 None %5
%7 = OpLabel
%1 = OpVariable %3 Function
%8 = OpUndef %2
OpReturn
OpFunctionEnd
)";
  EXPECT_EQ(ModuleToText(), expected_binary);
}

// Test cloning decoration for an id that is decorated via a group decoration.
TEST_F(DecorationManagerTest, CloneSomeGroupDecorations) {
  const std::string spirv = R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 RelaxedPrecision
OpDecorate %1 Restrict
%1 = OpDecorationGroup
OpGroupDecorate %1 %2
%3 = OpTypeInt 32 0
%4 = OpTypePointer Function %3
%5 = OpTypeVoid
%6 = OpTypeFunction %5
%7 = OpFunction %5 None %6
%8 = OpLabel
%2 = OpVariable %4 Function
%9 = OpUndef %3
OpReturn
OpFunctionEnd
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_EQ(GetErrorMessage(), "");

  // Check cloning OpDecorate including group decorations.
  auto decorations = decoManager->GetDecorationsFor(9u, false);
  EXPECT_EQ(GetErrorMessage(), "");
  EXPECT_TRUE(decorations.empty());

  decoManager->CloneDecorations(2u, 9u, {SpvDecorationRelaxedPrecision});
  decorations = decoManager->GetDecorationsFor(9u, false);
  EXPECT_THAT(GetErrorMessage(), "");

  std::string expected_decorations =
      R"(OpDecorate %9 RelaxedPrecision
)";
  EXPECT_EQ(ToText(decorations), expected_decorations);

  const std::string expected_binary = R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 RelaxedPrecision
OpDecorate %1 Restrict
%1 = OpDecorationGroup
OpGroupDecorate %1 %2
OpDecorate %9 RelaxedPrecision
%3 = OpTypeInt 32 0
%4 = OpTypePointer Function %3
%5 = OpTypeVoid
%6 = OpTypeFunction %5
%7 = OpFunction %5 None %6
%8 = OpLabel
%2 = OpVariable %4 Function
%9 = OpUndef %3
OpReturn
OpFunctionEnd
)";
  EXPECT_EQ(ModuleToText(), expected_binary);
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsWithoutGroupsTrue) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Restrict
OpDecorate %2 Constant
OpDecorate %2 Restrict
OpDecorate %1 Constant
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsWithoutGroupsFalse) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Restrict
OpDecorate %2 Constant
OpDecorate %2 Restrict
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsIdWithoutGroupsTrue) {
  const std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorateId %1 AlignmentId %nine
OpDecorateId %3 MaxByteOffsetId %nine
OpDecorateId %3 AlignmentId %nine
OpDecorateId %1 MaxByteOffsetId %nine
%u32    = OpTypeInt 32 0
%nine   = OpConstant %u32 9
%1      = OpVariable %u32 Uniform
%3      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 3u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsIdWithoutGroupsFalse) {
  const std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorateId %1 AlignmentId %nine
OpDecorateId %2 MaxByteOffsetId %nine
OpDecorateId %2 AlignmentId %nine
%u32    = OpTypeInt 32 0
%nine   = OpConstant %u32 9
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsStringWithoutGroupsTrue) {
  const std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "hello"
OpDecorateStringGOOGLE %2 HlslSemanticGOOGLE "world"
OpDecorateStringGOOGLE %2 HlslSemanticGOOGLE "hello"
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "world"
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsStringWithoutGroupsFalse) {
  const std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "hello"
OpDecorateStringGOOGLE %2 HlslSemanticGOOGLE "world"
OpDecorateStringGOOGLE %2 HlslSemanticGOOGLE "hello"
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsWithGroupsTrue) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Restrict
OpDecorate %2 Constant
OpDecorate %1 Constant
OpDecorate %3 Restrict
%3 = OpDecorationGroup
OpGroupDecorate %3 %2
OpDecorate %4 Invariant
%4 = OpDecorationGroup
OpGroupDecorate %4 %1 %2
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsWithGroupsFalse) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Restrict
OpDecorate %2 Constant
OpDecorate %1 Constant
OpDecorate %4 Invariant
%4 = OpDecorationGroup
OpGroupDecorate %4 %1 %2
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsDuplicateDecorations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %2 Constant
OpDecorate %2 Constant
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsDifferentVariations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Location 0
OpDecorate %2 Location 1
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest,
       HaveTheSameDecorationsDuplicateMemberDecorations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpMemberDecorate %1 0 Location 0
OpMemberDecorate %2 0 Location 0
OpMemberDecorate %2 0 Location 0
%u32    = OpTypeInt 32 0
%1      = OpTypeStruct %u32 %u32
%2      = OpTypeStruct %u32 %u32
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest,
       HaveTheSameDecorationsDifferentMemberSameDecoration) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpMemberDecorate %1 0 Location 0
OpMemberDecorate %2 1 Location 0
%u32    = OpTypeInt 32 0
%1      = OpTypeStruct %u32 %u32
%2      = OpTypeStruct %u32 %u32
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsDifferentMemberVariations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpMemberDecorate %1 0 Location 0
OpMemberDecorate %2 0 Location 1
%u32    = OpTypeInt 32 0
%1      = OpTypeStruct %u32 %u32
%2      = OpTypeStruct %u32 %u32
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsDuplicateIdDecorations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorateId %1 AlignmentId %2
OpDecorateId %3 AlignmentId %2
OpDecorateId %3 AlignmentId %2
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%3      = OpVariable %u32 Uniform
%2      = OpSpecConstant %u32 0
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 3u));
}

TEST_F(DecorationManagerTest,
       HaveTheSameDecorationsDuplicateStringDecorations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "hello"
OpDecorateStringGOOGLE %2 HlslSemanticGOOGLE "hello"
OpDecorateStringGOOGLE %2 HlslSemanticGOOGLE "hello"
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsDifferentIdVariations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorateId %1 AlignmentId %2
OpDecorateId %3 AlignmentId %4
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%3      = OpVariable %u32 Uniform
%2      = OpSpecConstant %u32 0
%4      = OpSpecConstant %u32 0
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsDifferentStringVariations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "hello"
OpDecorateStringGOOGLE %2 HlslSemanticGOOGLE "world"
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsLeftSymmetry) {
  // Left being a subset of right is not enough.
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %1 Constant
OpDecorate %2 Constant
OpDecorate %2 Restrict
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsRightSymmetry) {
  // Right being a subset of left is not enough.
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %1 Restrict
OpDecorate %2 Constant
OpDecorate %2 Constant
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationIdsLeftSymmetry) {
  const std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorateId %1 AlignmentId %nine
OpDecorateId %1 AlignmentId %nine
OpDecorateId %2 AlignmentId %nine
OpDecorateId %2 MaxByteOffsetId %nine
%u32    = OpTypeInt 32 0
%nine   = OpConstant %u32 9
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationIdsRightSymmetry) {
  const std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorateId %1 AlignmentId %nine
OpDecorateId %1 MaxByteOffsetId %nine
OpDecorateId %2 AlignmentId %nine
OpDecorateId %2 AlignmentId %nine
%u32    = OpTypeInt 32 0
%nine   = OpConstant %u32 9
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationStringsLeftSymmetry) {
  const std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "hello"
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "hello"
OpDecorateStringGOOGLE %2 HlslSemanticGOOGLE "hello"
OpDecorateStringGOOGLE %2 HlslSemanticGOOGLE "world"
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationStringsRightSymmetry) {
  const std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_decorate_string"
OpMemoryModel Logical GLSL450
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "hello"
OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE "world"
OpDecorateStringGOOGLE %2 HlslSemanticGOOGLE "hello"
OpDecorateStringGOOGLE %2 HlslSemanticGOOGLE "hello"
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

}  // namespace
}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
