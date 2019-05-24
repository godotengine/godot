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

#include <string>

#include "gmock/gmock.h"
#include "source/opt/struct_cfg_analysis.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using StructCFGAnalysisTest = PassTest<::testing::Test>;

TEST_F(StructCFGAnalysisTest, BBInSelection) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%bool = OpTypeBool
%bool_undef = OpUndef %bool
%uint = OpTypeInt 32 0
%uint_undef = OpUndef %uint
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%1 = OpLabel
OpSelectionMerge %3 None
OpBranchConditional %undef_bool %2 %3
%2 = OpLabel
OpBranch %3
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  StructuredCFGAnalysis analysis(context.get());

  // The header is not in the construct.
  EXPECT_EQ(analysis.ContainingConstruct(1), 0);
  EXPECT_EQ(analysis.ContainingLoop(1), 0);
  EXPECT_EQ(analysis.MergeBlock(1), 0);
  EXPECT_EQ(analysis.LoopMergeBlock(1), 0);

  // BB2 is in the construct.
  EXPECT_EQ(analysis.ContainingConstruct(2), 1);
  EXPECT_EQ(analysis.ContainingLoop(2), 0);
  EXPECT_EQ(analysis.MergeBlock(2), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(2), 0);

  // The merge node is not in the construct.
  EXPECT_EQ(analysis.ContainingConstruct(3), 0);
  EXPECT_EQ(analysis.ContainingLoop(3), 0);
  EXPECT_EQ(analysis.MergeBlock(3), 0);
  EXPECT_EQ(analysis.LoopMergeBlock(3), 0);
}

TEST_F(StructCFGAnalysisTest, BBInLoop) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%bool = OpTypeBool
%bool_undef = OpUndef %bool
%uint = OpTypeInt 32 0
%uint_undef = OpUndef %uint
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%entry_lab = OpLabel
OpBranch %1
%1 = OpLabel
OpLoopMerge %3 %4 None
OpBranchConditional %undef_bool %2 %3
%2 = OpLabel
OpBranch %3
%4 = OpLabel
OpBranch %1
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  StructuredCFGAnalysis analysis(context.get());

  // The header is not in the construct.
  EXPECT_EQ(analysis.ContainingConstruct(1), 0);
  EXPECT_EQ(analysis.ContainingLoop(1), 0);
  EXPECT_EQ(analysis.MergeBlock(1), 0);
  EXPECT_EQ(analysis.LoopMergeBlock(1), 0);

  // BB2 is in the construct.
  EXPECT_EQ(analysis.ContainingConstruct(2), 1);
  EXPECT_EQ(analysis.ContainingLoop(2), 1);
  EXPECT_EQ(analysis.MergeBlock(2), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(2), 3);

  // The merge node is not in the construct.
  EXPECT_EQ(analysis.ContainingConstruct(3), 0);
  EXPECT_EQ(analysis.ContainingLoop(3), 0);
  EXPECT_EQ(analysis.MergeBlock(3), 0);
  EXPECT_EQ(analysis.LoopMergeBlock(3), 0);

  // The continue block is in the construct.
  EXPECT_EQ(analysis.ContainingConstruct(4), 1);
  EXPECT_EQ(analysis.ContainingLoop(4), 1);
  EXPECT_EQ(analysis.MergeBlock(4), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(4), 3);
}

TEST_F(StructCFGAnalysisTest, SelectionInLoop) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%bool = OpTypeBool
%bool_undef = OpUndef %bool
%uint = OpTypeInt 32 0
%uint_undef = OpUndef %uint
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%entry_lab = OpLabel
OpBranch %1
%1 = OpLabel
OpLoopMerge %3 %4 None
OpBranchConditional %undef_bool %2 %3
%2 = OpLabel
OpSelectionMerge %6 None
OpBranchConditional %undef_bool %5 %6
%5 = OpLabel
OpBranch %6
%6 = OpLabel
OpBranch %3
%4 = OpLabel
OpBranch %1
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  StructuredCFGAnalysis analysis(context.get());

  // The loop header is not in either construct.
  EXPECT_EQ(analysis.ContainingConstruct(1), 0);
  EXPECT_EQ(analysis.ContainingLoop(1), 0);
  EXPECT_EQ(analysis.MergeBlock(1), 0);
  EXPECT_EQ(analysis.LoopMergeBlock(1), 0);

  // Selection header is in the loop only.
  EXPECT_EQ(analysis.ContainingConstruct(2), 1);
  EXPECT_EQ(analysis.ContainingLoop(2), 1);
  EXPECT_EQ(analysis.MergeBlock(2), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(2), 3);

  // The loop merge node is not in either construct.
  EXPECT_EQ(analysis.ContainingConstruct(3), 0);
  EXPECT_EQ(analysis.ContainingLoop(3), 0);
  EXPECT_EQ(analysis.MergeBlock(3), 0);
  EXPECT_EQ(analysis.LoopMergeBlock(3), 0);

  // The continue block is in the loop only.
  EXPECT_EQ(analysis.ContainingConstruct(4), 1);
  EXPECT_EQ(analysis.ContainingLoop(4), 1);
  EXPECT_EQ(analysis.MergeBlock(4), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(4), 3);

  // BB5 is in the selection fist and the loop.
  EXPECT_EQ(analysis.ContainingConstruct(5), 2);
  EXPECT_EQ(analysis.ContainingLoop(5), 1);
  EXPECT_EQ(analysis.MergeBlock(5), 6);
  EXPECT_EQ(analysis.LoopMergeBlock(5), 3);

  // The selection merge is in the loop only.
  EXPECT_EQ(analysis.ContainingConstruct(6), 1);
  EXPECT_EQ(analysis.ContainingLoop(6), 1);
  EXPECT_EQ(analysis.MergeBlock(6), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(6), 3);
}

TEST_F(StructCFGAnalysisTest, LoopInSelection) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%bool = OpTypeBool
%bool_undef = OpUndef %bool
%uint = OpTypeInt 32 0
%uint_undef = OpUndef %uint
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%entry_lab = OpLabel
OpBranch %1
%1 = OpLabel
OpSelectionMerge %3 None
OpBranchConditional %undef_bool %2 %3
%2 = OpLabel
OpLoopMerge %4 %5 None
OpBranchConditional %undef_bool %4 %6
%5 = OpLabel
OpBranch %2
%6 = OpLabel
OpBranch %4
%4 = OpLabel
OpBranch %3
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  StructuredCFGAnalysis analysis(context.get());

  // The selection header is not in either construct.
  EXPECT_EQ(analysis.ContainingConstruct(1), 0);
  EXPECT_EQ(analysis.ContainingLoop(1), 0);
  EXPECT_EQ(analysis.MergeBlock(1), 0);
  EXPECT_EQ(analysis.LoopMergeBlock(1), 0);

  // Loop header is in the selection only.
  EXPECT_EQ(analysis.ContainingConstruct(2), 1);
  EXPECT_EQ(analysis.ContainingLoop(2), 0);
  EXPECT_EQ(analysis.MergeBlock(2), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(2), 0);

  // The selection merge node is not in either construct.
  EXPECT_EQ(analysis.ContainingConstruct(3), 0);
  EXPECT_EQ(analysis.ContainingLoop(3), 0);
  EXPECT_EQ(analysis.MergeBlock(3), 0);
  EXPECT_EQ(analysis.LoopMergeBlock(3), 0);

  // The loop merge is in the selection only.
  EXPECT_EQ(analysis.ContainingConstruct(4), 1);
  EXPECT_EQ(analysis.ContainingLoop(4), 0);
  EXPECT_EQ(analysis.MergeBlock(4), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(4), 0);

  // The loop continue target is in the loop.
  EXPECT_EQ(analysis.ContainingConstruct(5), 2);
  EXPECT_EQ(analysis.ContainingLoop(5), 2);
  EXPECT_EQ(analysis.MergeBlock(5), 4);
  EXPECT_EQ(analysis.LoopMergeBlock(5), 4);

  // BB6 is in the loop.
  EXPECT_EQ(analysis.ContainingConstruct(6), 2);
  EXPECT_EQ(analysis.ContainingLoop(6), 2);
  EXPECT_EQ(analysis.MergeBlock(6), 4);
  EXPECT_EQ(analysis.LoopMergeBlock(6), 4);
}

TEST_F(StructCFGAnalysisTest, SelectionInSelection) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%bool = OpTypeBool
%bool_undef = OpUndef %bool
%uint = OpTypeInt 32 0
%uint_undef = OpUndef %uint
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%entry_lab = OpLabel
OpBranch %1
%1 = OpLabel
OpSelectionMerge %3 None
OpBranchConditional %undef_bool %2 %3
%2 = OpLabel
OpSelectionMerge %4 None
OpBranchConditional %undef_bool %4 %5
%5 = OpLabel
OpBranch %4
%4 = OpLabel
OpBranch %3
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  StructuredCFGAnalysis analysis(context.get());

  // The outer selection header is not in either construct.
  EXPECT_EQ(analysis.ContainingConstruct(1), 0);
  EXPECT_EQ(analysis.ContainingLoop(1), 0);
  EXPECT_EQ(analysis.MergeBlock(1), 0);
  EXPECT_EQ(analysis.LoopMergeBlock(1), 0);

  // The inner header is in the outer selection.
  EXPECT_EQ(analysis.ContainingConstruct(2), 1);
  EXPECT_EQ(analysis.ContainingLoop(2), 0);
  EXPECT_EQ(analysis.MergeBlock(2), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(2), 0);

  // The outer merge node is not in either construct.
  EXPECT_EQ(analysis.ContainingConstruct(3), 0);
  EXPECT_EQ(analysis.ContainingLoop(3), 0);
  EXPECT_EQ(analysis.MergeBlock(3), 0);
  EXPECT_EQ(analysis.LoopMergeBlock(3), 0);

  // The inner merge is in the outer selection.
  EXPECT_EQ(analysis.ContainingConstruct(4), 1);
  EXPECT_EQ(analysis.ContainingLoop(4), 0);
  EXPECT_EQ(analysis.MergeBlock(4), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(4), 0);

  // BB5 is in the inner selection.
  EXPECT_EQ(analysis.ContainingConstruct(5), 2);
  EXPECT_EQ(analysis.ContainingLoop(5), 0);
  EXPECT_EQ(analysis.MergeBlock(5), 4);
  EXPECT_EQ(analysis.LoopMergeBlock(5), 0);
}

TEST_F(StructCFGAnalysisTest, LoopInLoop) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%bool = OpTypeBool
%bool_undef = OpUndef %bool
%uint = OpTypeInt 32 0
%uint_undef = OpUndef %uint
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%entry_lab = OpLabel
OpBranch %1
%1 = OpLabel
OpLoopMerge %3 %7 None
OpBranchConditional %undef_bool %2 %3
%2 = OpLabel
OpLoopMerge %4 %5 None
OpBranchConditional %undef_bool %4 %6
%5 = OpLabel
OpBranch %2
%6 = OpLabel
OpBranch %4
%4 = OpLabel
OpBranch %3
%7 = OpLabel
OpBranch %1
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  StructuredCFGAnalysis analysis(context.get());

  // The outer loop header is not in either construct.
  EXPECT_EQ(analysis.ContainingConstruct(1), 0);
  EXPECT_EQ(analysis.ContainingLoop(1), 0);
  EXPECT_EQ(analysis.MergeBlock(1), 0);
  EXPECT_EQ(analysis.LoopMergeBlock(1), 0);

  // The inner loop header is in the outer loop.
  EXPECT_EQ(analysis.ContainingConstruct(2), 1);
  EXPECT_EQ(analysis.ContainingLoop(2), 1);
  EXPECT_EQ(analysis.MergeBlock(2), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(2), 3);

  // The outer merge node is not in either construct.
  EXPECT_EQ(analysis.ContainingConstruct(3), 0);
  EXPECT_EQ(analysis.ContainingLoop(3), 0);
  EXPECT_EQ(analysis.MergeBlock(3), 0);
  EXPECT_EQ(analysis.LoopMergeBlock(3), 0);

  // The inner merge is in the outer loop.
  EXPECT_EQ(analysis.ContainingConstruct(4), 1);
  EXPECT_EQ(analysis.ContainingLoop(4), 1);
  EXPECT_EQ(analysis.MergeBlock(4), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(4), 3);

  // The inner continue target is in the inner loop.
  EXPECT_EQ(analysis.ContainingConstruct(5), 2);
  EXPECT_EQ(analysis.ContainingLoop(5), 2);
  EXPECT_EQ(analysis.MergeBlock(5), 4);
  EXPECT_EQ(analysis.LoopMergeBlock(5), 4);

  // BB6 is in the loop.
  EXPECT_EQ(analysis.ContainingConstruct(6), 2);
  EXPECT_EQ(analysis.ContainingLoop(6), 2);
  EXPECT_EQ(analysis.MergeBlock(6), 4);
  EXPECT_EQ(analysis.LoopMergeBlock(6), 4);

  // The outer continue target is in the outer loop.
  EXPECT_EQ(analysis.ContainingConstruct(7), 1);
  EXPECT_EQ(analysis.ContainingLoop(7), 1);
  EXPECT_EQ(analysis.MergeBlock(7), 3);
  EXPECT_EQ(analysis.LoopMergeBlock(7), 3);
}

TEST_F(StructCFGAnalysisTest, KernelTest) {
  const std::string text = R"(
OpCapability Kernel
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%bool = OpTypeBool
%bool_undef = OpUndef %bool
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%1 = OpLabel
OpBranchConditional %undef_bool %2 %3
%2 = OpLabel
OpBranch %3
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  StructuredCFGAnalysis analysis(context.get());

  // No structured control flow, so none of the basic block are in any
  // construct.
  for (uint32_t i = 1; i <= 3; i++) {
    EXPECT_EQ(analysis.ContainingConstruct(i), 0);
    EXPECT_EQ(analysis.ContainingLoop(i), 0);
    EXPECT_EQ(analysis.MergeBlock(i), 0);
    EXPECT_EQ(analysis.LoopMergeBlock(i), 0);
  }
}

TEST_F(StructCFGAnalysisTest, EmptyFunctionTest) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %func LinkageAttributes "x" Import
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%func = OpFunction %void None %void_fn
OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  // #2451: This segfaulted on empty functions.
  StructuredCFGAnalysis analysis(context.get());
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
