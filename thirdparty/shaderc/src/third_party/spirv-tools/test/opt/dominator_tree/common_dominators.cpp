// Copyright (c) 2018 Google LLC
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace {

using CommonDominatorsTest = ::testing::Test;

const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %3 %4 None
OpBranch %5
%5 = OpLabel
OpBranchConditional %true %3 %4
%4 = OpLabel
OpBranch %2
%3 = OpLabel
OpSelectionMerge %6 None
OpBranchConditional %true %7 %8
%7 = OpLabel
OpBranch %6
%8 = OpLabel
OpBranch %9
%9 = OpLabel
OpBranch %6
%6 = OpLabel
OpBranch %10
%11 = OpLabel
OpBranch %10
%10 = OpLabel
OpReturn
OpFunctionEnd
)";

BasicBlock* GetBlock(uint32_t id, std::unique_ptr<IRContext>& context) {
  return context->get_instr_block(context->get_def_use_mgr()->GetDef(id));
}

TEST(CommonDominatorsTest, SameBlock) {
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, context);

  DominatorAnalysis* analysis =
      context->GetDominatorAnalysis(&*context->module()->begin());

  for (auto& block : *context->module()->begin()) {
    EXPECT_EQ(&block, analysis->CommonDominator(&block, &block));
  }
}

TEST(CommonDominatorsTest, ParentAndChild) {
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, context);

  DominatorAnalysis* analysis =
      context->GetDominatorAnalysis(&*context->module()->begin());

  EXPECT_EQ(
      GetBlock(1u, context),
      analysis->CommonDominator(GetBlock(1u, context), GetBlock(2u, context)));
  EXPECT_EQ(
      GetBlock(2u, context),
      analysis->CommonDominator(GetBlock(2u, context), GetBlock(5u, context)));
  EXPECT_EQ(
      GetBlock(1u, context),
      analysis->CommonDominator(GetBlock(1u, context), GetBlock(5u, context)));
}

TEST(CommonDominatorsTest, BranchSplit) {
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, context);

  DominatorAnalysis* analysis =
      context->GetDominatorAnalysis(&*context->module()->begin());

  EXPECT_EQ(
      GetBlock(3u, context),
      analysis->CommonDominator(GetBlock(7u, context), GetBlock(8u, context)));
  EXPECT_EQ(
      GetBlock(3u, context),
      analysis->CommonDominator(GetBlock(7u, context), GetBlock(9u, context)));
}

TEST(CommonDominatorsTest, LoopContinueAndMerge) {
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, context);

  DominatorAnalysis* analysis =
      context->GetDominatorAnalysis(&*context->module()->begin());

  EXPECT_EQ(
      GetBlock(5u, context),
      analysis->CommonDominator(GetBlock(3u, context), GetBlock(4u, context)));
}

TEST(CommonDominatorsTest, NoCommonDominator) {
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, context);

  DominatorAnalysis* analysis =
      context->GetDominatorAnalysis(&*context->module()->begin());

  EXPECT_EQ(nullptr, analysis->CommonDominator(GetBlock(10u, context),
                                               GetBlock(11u, context)));
  EXPECT_EQ(nullptr, analysis->CommonDominator(GetBlock(11u, context),
                                               GetBlock(6u, context)));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
