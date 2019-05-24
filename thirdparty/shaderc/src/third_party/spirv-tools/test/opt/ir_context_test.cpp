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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/ir_context.h"
#include "source/opt/pass.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using Analysis = IRContext::Analysis;
using ::testing::Each;
using ::testing::UnorderedElementsAre;

class DummyPassPreservesNothing : public Pass {
 public:
  DummyPassPreservesNothing(Status s) : Pass(), status_to_return_(s) {}

  const char* name() const override { return "dummy-pass"; }
  Status Process() override { return status_to_return_; }

 private:
  Status status_to_return_;
};

class DummyPassPreservesAll : public Pass {
 public:
  DummyPassPreservesAll(Status s) : Pass(), status_to_return_(s) {}

  const char* name() const override { return "dummy-pass"; }
  Status Process() override { return status_to_return_; }

  Analysis GetPreservedAnalyses() override {
    return Analysis(IRContext::kAnalysisEnd - 1);
  }

 private:
  Status status_to_return_;
};

class DummyPassPreservesFirst : public Pass {
 public:
  DummyPassPreservesFirst(Status s) : Pass(), status_to_return_(s) {}

  const char* name() const override { return "dummy-pass"; }
  Status Process() override { return status_to_return_; }

  Analysis GetPreservedAnalyses() override { return IRContext::kAnalysisBegin; }

 private:
  Status status_to_return_;
};

using IRContextTest = PassTest<::testing::Test>;

TEST_F(IRContextTest, IndividualValidAfterBuild) {
  std::unique_ptr<Module> module(new Module());
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         spvtools::MessageConsumer());

  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    localContext.BuildInvalidAnalyses(i);
    EXPECT_TRUE(localContext.AreAnalysesValid(i));
  }
}

TEST_F(IRContextTest, AllValidAfterBuild) {
  std::unique_ptr<Module> module = MakeUnique<Module>();
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         spvtools::MessageConsumer());

  Analysis built_analyses = IRContext::kAnalysisNone;
  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    localContext.BuildInvalidAnalyses(i);
    built_analyses |= i;
  }
  EXPECT_TRUE(localContext.AreAnalysesValid(built_analyses));
}

TEST_F(IRContextTest, AllValidAfterPassNoChange) {
  std::unique_ptr<Module> module = MakeUnique<Module>();
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         spvtools::MessageConsumer());

  Analysis built_analyses = IRContext::kAnalysisNone;
  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    localContext.BuildInvalidAnalyses(i);
    built_analyses |= i;
  }

  DummyPassPreservesNothing pass(Pass::Status::SuccessWithoutChange);
  Pass::Status s = pass.Run(&localContext);
  EXPECT_EQ(s, Pass::Status::SuccessWithoutChange);
  EXPECT_TRUE(localContext.AreAnalysesValid(built_analyses));
}

TEST_F(IRContextTest, NoneValidAfterPassWithChange) {
  std::unique_ptr<Module> module = MakeUnique<Module>();
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         spvtools::MessageConsumer());

  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    localContext.BuildInvalidAnalyses(i);
  }

  DummyPassPreservesNothing pass(Pass::Status::SuccessWithChange);
  Pass::Status s = pass.Run(&localContext);
  EXPECT_EQ(s, Pass::Status::SuccessWithChange);
  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    EXPECT_FALSE(localContext.AreAnalysesValid(i));
  }
}

TEST_F(IRContextTest, AllPreservedAfterPassWithChange) {
  std::unique_ptr<Module> module = MakeUnique<Module>();
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         spvtools::MessageConsumer());

  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    localContext.BuildInvalidAnalyses(i);
  }

  DummyPassPreservesAll pass(Pass::Status::SuccessWithChange);
  Pass::Status s = pass.Run(&localContext);
  EXPECT_EQ(s, Pass::Status::SuccessWithChange);
  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    EXPECT_TRUE(localContext.AreAnalysesValid(i));
  }
}

TEST_F(IRContextTest, PreserveFirstOnlyAfterPassWithChange) {
  std::unique_ptr<Module> module = MakeUnique<Module>();
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         spvtools::MessageConsumer());

  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    localContext.BuildInvalidAnalyses(i);
  }

  DummyPassPreservesFirst pass(Pass::Status::SuccessWithChange);
  Pass::Status s = pass.Run(&localContext);
  EXPECT_EQ(s, Pass::Status::SuccessWithChange);
  EXPECT_TRUE(localContext.AreAnalysesValid(IRContext::kAnalysisBegin));
  for (Analysis i = IRContext::kAnalysisBegin << 1; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    EXPECT_FALSE(localContext.AreAnalysesValid(i));
  }
}

TEST_F(IRContextTest, KillMemberName) {
  const std::string text = R"(
              OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "stuff"
               OpMemberName %3 0 "refZ"
               OpMemberDecorate %3 0 Offset 0
               OpDecorate %3 Block
          %4 = OpTypeFloat 32
          %3 = OpTypeStruct %4
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %2 = OpFunction %5 None %6
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);

  // Build the decoration manager.
  context->get_decoration_mgr();

  // Delete the OpTypeStruct.  Should delete the OpName, OpMemberName, and
  // OpMemberDecorate associated with it.
  context->KillDef(3);

  // Make sure all of the name are removed.
  for (auto& inst : context->debugs2()) {
    EXPECT_EQ(inst.opcode(), SpvOpNop);
  }

  // Make sure all of the decorations are removed.
  for (auto& inst : context->annotations()) {
    EXPECT_EQ(inst.opcode(), SpvOpNop);
  }
}

TEST_F(IRContextTest, KillGroupDecoration) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpDecorate %3 Restrict
          %3 = OpDecorationGroup
               OpGroupDecorate %3 %4 %5
          %6 = OpTypeFloat 32
          %7 = OpTypePointer Function %6
          %8 = OpTypeStruct %6
          %9 = OpTypeVoid
         %10 = OpTypeFunction %9
          %2 = OpFunction %9 None %10
         %11 = OpLabel
          %4 = OpVariable %7 Function
          %5 = OpVariable %7 Function
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);

  // Build the decoration manager.
  context->get_decoration_mgr();

  // Delete the second variable.
  context->KillDef(5);

  // The three decorations instructions should still be there.  The first two
  // should be the same, but the third should have %5 removed.

  // Check the OpDecorate instruction
  auto inst = context->annotation_begin();
  EXPECT_EQ(inst->opcode(), SpvOpDecorate);
  EXPECT_EQ(inst->GetSingleWordInOperand(0), 3);

  // Check the OpDecorationGroup Instruction
  ++inst;
  EXPECT_EQ(inst->opcode(), SpvOpDecorationGroup);
  EXPECT_EQ(inst->result_id(), 3);

  // Check that %5 is no longer part of the group.
  ++inst;
  EXPECT_EQ(inst->opcode(), SpvOpGroupDecorate);
  EXPECT_EQ(inst->NumInOperands(), 2);
  EXPECT_EQ(inst->GetSingleWordInOperand(0), 3);
  EXPECT_EQ(inst->GetSingleWordInOperand(1), 4);

  // Check that we are at the end.
  ++inst;
  EXPECT_EQ(inst, context->annotation_end());
}

TEST_F(IRContextTest, TakeNextUniqueIdIncrementing) {
  const uint32_t NUM_TESTS = 1000;
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, nullptr);
  for (uint32_t i = 1; i < NUM_TESTS; ++i)
    EXPECT_EQ(i, localContext.TakeNextUniqueId());
}

TEST_F(IRContextTest, KillGroupDecorationWitNoDecorations) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpDecorationGroup
               OpGroupDecorate %3 %4 %5
          %6 = OpTypeFloat 32
          %7 = OpTypePointer Function %6
          %8 = OpTypeStruct %6
          %9 = OpTypeVoid
         %10 = OpTypeFunction %9
          %2 = OpFunction %9 None %10
         %11 = OpLabel
          %4 = OpVariable %7 Function
          %5 = OpVariable %7 Function
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);

  // Build the decoration manager.
  context->get_decoration_mgr();

  // Delete the second variable.
  context->KillDef(5);

  // The two decoration instructions should still be there.  The first one
  // should be the same, but the second should have %5 removed.

  // Check the OpDecorationGroup Instruction
  auto inst = context->annotation_begin();
  EXPECT_EQ(inst->opcode(), SpvOpDecorationGroup);
  EXPECT_EQ(inst->result_id(), 3);

  // Check that %5 is no longer part of the group.
  ++inst;
  EXPECT_EQ(inst->opcode(), SpvOpGroupDecorate);
  EXPECT_EQ(inst->NumInOperands(), 2);
  EXPECT_EQ(inst->GetSingleWordInOperand(0), 3);
  EXPECT_EQ(inst->GetSingleWordInOperand(1), 4);

  // Check that we are at the end.
  ++inst;
  EXPECT_EQ(inst, context->annotation_end());
}

TEST_F(IRContextTest, KillDecorationGroup) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpDecorationGroup
               OpGroupDecorate %3 %4 %5
          %6 = OpTypeFloat 32
          %7 = OpTypePointer Function %6
          %8 = OpTypeStruct %6
          %9 = OpTypeVoid
         %10 = OpTypeFunction %9
          %2 = OpFunction %9 None %10
         %11 = OpLabel
          %4 = OpVariable %7 Function
          %5 = OpVariable %7 Function
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);

  // Build the decoration manager.
  context->get_decoration_mgr();

  // Delete the second variable.
  context->KillDef(3);

  // Check the OpDecorationGroup Instruction is still there.
  EXPECT_TRUE(context->annotations().empty());
}

TEST_F(IRContextTest, BasicVisitFromEntryPoint) {
  // Make sure we visit the entry point, and the function it calls.
  // Do not visit Dead or Exported.
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %10 "main"
               OpName %10 "main"
               OpName %Dead "Dead"
               OpName %11 "Constant"
               OpName %ExportedFunc "ExportedFunc"
               OpDecorate %ExportedFunc LinkageAttributes "ExportedFunc" Export
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
         %10 = OpFunction %void None %6
         %14 = OpLabel
         %15 = OpFunctionCall %void %11
         %16 = OpFunctionCall %void %11
               OpReturn
               OpFunctionEnd
         %11 = OpFunction %void None %6
         %18 = OpLabel
               OpReturn
               OpFunctionEnd
       %Dead = OpFunction %void None %6
         %19 = OpLabel
               OpReturn
               OpFunctionEnd
%ExportedFunc = OpFunction %void None %7
         %20 = OpLabel
         %21 = OpFunctionCall %void %11
               OpReturn
               OpFunctionEnd
)";
  // clang-format on

  std::unique_ptr<IRContext> localContext =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, localContext) << "Assembling failed for shader:\n"
                                   << text << std::endl;
  std::vector<uint32_t> processed;
  Pass::ProcessFunction mark_visited = [&processed](Function* fp) {
    processed.push_back(fp->result_id());
    return false;
  };
  localContext->ProcessEntryPointCallTree(mark_visited);
  EXPECT_THAT(processed, UnorderedElementsAre(10, 11));
}

TEST_F(IRContextTest, BasicVisitReachable) {
  // Make sure we visit the entry point, exported function, and the function
  // they call. Do not visit Dead.
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %10 "main"
               OpName %10 "main"
               OpName %Dead "Dead"
               OpName %11 "Constant"
               OpName %12 "ExportedFunc"
               OpName %13 "Constant2"
               OpDecorate %12 LinkageAttributes "ExportedFunc" Export
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
         %10 = OpFunction %void None %6
         %14 = OpLabel
         %15 = OpFunctionCall %void %11
         %16 = OpFunctionCall %void %11
               OpReturn
               OpFunctionEnd
         %11 = OpFunction %void None %6
         %18 = OpLabel
               OpReturn
               OpFunctionEnd
       %Dead = OpFunction %void None %6
         %19 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %void None %6
         %20 = OpLabel
         %21 = OpFunctionCall %void %13
               OpReturn
               OpFunctionEnd
         %13 = OpFunction %void None %6
         %22 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  // clang-format on

  std::unique_ptr<IRContext> localContext =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, localContext) << "Assembling failed for shader:\n"
                                   << text << std::endl;

  std::vector<uint32_t> processed;
  Pass::ProcessFunction mark_visited = [&processed](Function* fp) {
    processed.push_back(fp->result_id());
    return false;
  };
  localContext->ProcessReachableCallTree(mark_visited);
  EXPECT_THAT(processed, UnorderedElementsAre(10, 11, 12, 13));
}

TEST_F(IRContextTest, BasicVisitOnlyOnce) {
  // Make sure we visit %12 only once, even if it is called from two different
  // functions.
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %10 "main"
               OpName %10 "main"
               OpName %Dead "Dead"
               OpName %11 "Constant"
               OpName %12 "ExportedFunc"
               OpDecorate %12 LinkageAttributes "ExportedFunc" Export
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
         %10 = OpFunction %void None %6
         %14 = OpLabel
         %15 = OpFunctionCall %void %11
         %16 = OpFunctionCall %void %12
               OpReturn
               OpFunctionEnd
         %11 = OpFunction %void None %6
         %18 = OpLabel
         %19 = OpFunctionCall %void %12
               OpReturn
               OpFunctionEnd
       %Dead = OpFunction %void None %6
         %20 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %void None %6
         %21 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  // clang-format on

  std::unique_ptr<IRContext> localContext =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, localContext) << "Assembling failed for shader:\n"
                                   << text << std::endl;

  std::vector<uint32_t> processed;
  Pass::ProcessFunction mark_visited = [&processed](Function* fp) {
    processed.push_back(fp->result_id());
    return false;
  };
  localContext->ProcessReachableCallTree(mark_visited);
  EXPECT_THAT(processed, UnorderedElementsAre(10, 11, 12));
}

TEST_F(IRContextTest, BasicDontVisitExportedVariable) {
  // Make sure we only visit functions and not exported variables.
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %10 "main"
               OpExecutionMode %10 OriginUpperLeft
               OpSource GLSL 150
               OpName %10 "main"
               OpName %12 "export_var"
               OpDecorate %12 LinkageAttributes "export_var" Export
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
      %float = OpTypeFloat 32
  %float_1 = OpConstant %float 1
         %12 = OpVariable %float Output
         %10 = OpFunction %void None %6
         %14 = OpLabel
               OpStore %12 %float_1
               OpReturn
               OpFunctionEnd
)";
  // clang-format on

  std::unique_ptr<IRContext> localContext =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, localContext) << "Assembling failed for shader:\n"
                                   << text << std::endl;

  std::vector<uint32_t> processed;
  Pass::ProcessFunction mark_visited = [&processed](Function* fp) {
    processed.push_back(fp->result_id());
    return false;
  };
  localContext->ProcessReachableCallTree(mark_visited);
  EXPECT_THAT(processed, UnorderedElementsAre(10));
}

TEST_F(IRContextTest, IdBoundTestAtLimit) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturn
OpFunctionEnd)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  uint32_t current_bound = context->module()->id_bound();
  context->set_max_id_bound(current_bound);
  uint32_t next_id_bound = context->TakeNextId();
  EXPECT_EQ(next_id_bound, 0);
  EXPECT_EQ(current_bound, context->module()->id_bound());
  next_id_bound = context->TakeNextId();
  EXPECT_EQ(next_id_bound, 0);
}

TEST_F(IRContextTest, IdBoundTestBelowLimit) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturn
OpFunctionEnd)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  uint32_t current_bound = context->module()->id_bound();
  context->set_max_id_bound(current_bound + 100);
  uint32_t next_id_bound = context->TakeNextId();
  EXPECT_EQ(next_id_bound, current_bound);
  EXPECT_EQ(current_bound + 1, context->module()->id_bound());
  next_id_bound = context->TakeNextId();
  EXPECT_EQ(next_id_bound, current_bound + 1);
}

TEST_F(IRContextTest, IdBoundTestNearLimit) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturn
OpFunctionEnd)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  uint32_t current_bound = context->module()->id_bound();
  context->set_max_id_bound(current_bound + 1);
  uint32_t next_id_bound = context->TakeNextId();
  EXPECT_EQ(next_id_bound, current_bound);
  EXPECT_EQ(current_bound + 1, context->module()->id_bound());
  next_id_bound = context->TakeNextId();
  EXPECT_EQ(next_id_bound, 0);
}

TEST_F(IRContextTest, IdBoundTestUIntMax) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4294967294 = OpLabel ; ID is UINT_MAX-1
OpReturn
OpFunctionEnd)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  uint32_t current_bound = context->module()->id_bound();

  // Expecting |BuildModule| to preserve the numeric ids.
  EXPECT_EQ(current_bound, std::numeric_limits<uint32_t>::max());

  context->set_max_id_bound(current_bound);
  uint32_t next_id_bound = context->TakeNextId();
  EXPECT_EQ(next_id_bound, 0);
  EXPECT_EQ(current_bound, context->module()->id_bound());
}
}  // namespace
}  // namespace opt
}  // namespace spvtools
