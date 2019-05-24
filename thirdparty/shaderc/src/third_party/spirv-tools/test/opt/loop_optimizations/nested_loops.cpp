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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "gmock/gmock.h"
#include "source/opt/iterator.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/pass.h"
#include "source/opt/tree_iterator.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/function_utils.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::UnorderedElementsAre;

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

using PassClassTest = PassTest<::testing::Test>;

/*
Generated from the following GLSL
#version 330 core
layout(location = 0) out vec4 c;
void main() {
  int i = 0;
  for (; i < 10; ++i) {
    int j = 0;
    int k = 0;
    for (; j < 11; ++j) {}
    for (; k < 12; ++k) {}
  }
}
*/
TEST_F(PassClassTest, BasicVisitFromEntryPoint) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %4 "i"
               OpName %5 "j"
               OpName %6 "k"
               OpName %3 "c"
               OpDecorate %3 Location 0
          %7 = OpTypeVoid
          %8 = OpTypeFunction %7
          %9 = OpTypeInt 32 1
         %10 = OpTypePointer Function %9
         %11 = OpConstant %9 0
         %12 = OpConstant %9 10
         %13 = OpTypeBool
         %14 = OpConstant %9 11
         %15 = OpConstant %9 1
         %16 = OpConstant %9 12
         %17 = OpTypeFloat 32
         %18 = OpTypeVector %17 4
         %19 = OpTypePointer Output %18
          %3 = OpVariable %19 Output
          %2 = OpFunction %7 None %8
         %20 = OpLabel
          %4 = OpVariable %10 Function
          %5 = OpVariable %10 Function
          %6 = OpVariable %10 Function
               OpStore %4 %11
               OpBranch %21
         %21 = OpLabel
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %25 = OpLoad %9 %4
         %26 = OpSLessThan %13 %25 %12
               OpBranchConditional %26 %27 %22
         %27 = OpLabel
               OpStore %5 %11
               OpStore %6 %11
               OpBranch %28
         %28 = OpLabel
               OpLoopMerge %29 %30 None
               OpBranch %31
         %31 = OpLabel
         %32 = OpLoad %9 %5
         %33 = OpSLessThan %13 %32 %14
               OpBranchConditional %33 %34 %29
         %34 = OpLabel
               OpBranch %30
         %30 = OpLabel
         %35 = OpLoad %9 %5
         %36 = OpIAdd %9 %35 %15
               OpStore %5 %36
               OpBranch %28
         %29 = OpLabel
               OpBranch %37
         %37 = OpLabel
               OpLoopMerge %38 %39 None
               OpBranch %40
         %40 = OpLabel
         %41 = OpLoad %9 %6
         %42 = OpSLessThan %13 %41 %16
               OpBranchConditional %42 %43 %38
         %43 = OpLabel
               OpBranch %39
         %39 = OpLabel
         %44 = OpLoad %9 %6
         %45 = OpIAdd %9 %44 %15
               OpStore %6 %45
               OpBranch %37
         %38 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %46 = OpLoad %9 %4
         %47 = OpIAdd %9 %46 %15
               OpStore %4 %47
               OpBranch %21
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
  const Function* f = spvtest::GetFunction(module, 2);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  EXPECT_EQ(ld.NumLoops(), 3u);

  // Invalid basic block id.
  EXPECT_EQ(ld[0u], nullptr);
  // Not a loop header.
  EXPECT_EQ(ld[20], nullptr);

  Loop& parent_loop = *ld[21];
  EXPECT_TRUE(parent_loop.HasNestedLoops());
  EXPECT_FALSE(parent_loop.IsNested());
  EXPECT_EQ(parent_loop.GetDepth(), 1u);
  EXPECT_EQ(std::distance(parent_loop.begin(), parent_loop.end()), 2u);
  EXPECT_EQ(parent_loop.GetHeaderBlock(), spvtest::GetBasicBlock(f, 21));
  EXPECT_EQ(parent_loop.GetLatchBlock(), spvtest::GetBasicBlock(f, 23));
  EXPECT_EQ(parent_loop.GetMergeBlock(), spvtest::GetBasicBlock(f, 22));

  Loop& child_loop_1 = *ld[28];
  EXPECT_FALSE(child_loop_1.HasNestedLoops());
  EXPECT_TRUE(child_loop_1.IsNested());
  EXPECT_EQ(child_loop_1.GetDepth(), 2u);
  EXPECT_EQ(std::distance(child_loop_1.begin(), child_loop_1.end()), 0u);
  EXPECT_EQ(child_loop_1.GetHeaderBlock(), spvtest::GetBasicBlock(f, 28));
  EXPECT_EQ(child_loop_1.GetLatchBlock(), spvtest::GetBasicBlock(f, 30));
  EXPECT_EQ(child_loop_1.GetMergeBlock(), spvtest::GetBasicBlock(f, 29));

  Loop& child_loop_2 = *ld[37];
  EXPECT_FALSE(child_loop_2.HasNestedLoops());
  EXPECT_TRUE(child_loop_2.IsNested());
  EXPECT_EQ(child_loop_2.GetDepth(), 2u);
  EXPECT_EQ(std::distance(child_loop_2.begin(), child_loop_2.end()), 0u);
  EXPECT_EQ(child_loop_2.GetHeaderBlock(), spvtest::GetBasicBlock(f, 37));
  EXPECT_EQ(child_loop_2.GetLatchBlock(), spvtest::GetBasicBlock(f, 39));
  EXPECT_EQ(child_loop_2.GetMergeBlock(), spvtest::GetBasicBlock(f, 38));
}

static void CheckLoopBlocks(Loop* loop,
                            std::unordered_set<uint32_t>* expected_ids) {
  SCOPED_TRACE("Check loop " + std::to_string(loop->GetHeaderBlock()->id()));
  for (uint32_t bb_id : loop->GetBlocks()) {
    EXPECT_EQ(expected_ids->count(bb_id), 1u);
    expected_ids->erase(bb_id);
  }
  EXPECT_FALSE(loop->IsInsideLoop(loop->GetMergeBlock()));
  EXPECT_EQ(expected_ids->size(), 0u);
}

/*
Generated from the following GLSL
#version 330 core
layout(location = 0) out vec4 c;
void main() {
  int i = 0;
  for (; i < 10; ++i) {
    for (int j = 0; j < 11; ++j) {
      if (j < 5) {
        for (int k = 0; k < 12; ++k) {}
      }
      else {}
      for (int k = 0; k < 12; ++k) {}
    }
  }
}*/
TEST_F(PassClassTest, TripleNestedLoop) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %4 "i"
               OpName %5 "j"
               OpName %6 "k"
               OpName %7 "k"
               OpName %3 "c"
               OpDecorate %3 Location 0
          %8 = OpTypeVoid
          %9 = OpTypeFunction %8
         %10 = OpTypeInt 32 1
         %11 = OpTypePointer Function %10
         %12 = OpConstant %10 0
         %13 = OpConstant %10 10
         %14 = OpTypeBool
         %15 = OpConstant %10 11
         %16 = OpConstant %10 5
         %17 = OpConstant %10 12
         %18 = OpConstant %10 1
         %19 = OpTypeFloat 32
         %20 = OpTypeVector %19 4
         %21 = OpTypePointer Output %20
          %3 = OpVariable %21 Output
          %2 = OpFunction %8 None %9
         %22 = OpLabel
          %4 = OpVariable %11 Function
          %5 = OpVariable %11 Function
          %6 = OpVariable %11 Function
          %7 = OpVariable %11 Function
               OpStore %4 %12
               OpBranch %23
         %23 = OpLabel
               OpLoopMerge %24 %25 None
               OpBranch %26
         %26 = OpLabel
         %27 = OpLoad %10 %4
         %28 = OpSLessThan %14 %27 %13
               OpBranchConditional %28 %29 %24
         %29 = OpLabel
               OpStore %5 %12
               OpBranch %30
         %30 = OpLabel
               OpLoopMerge %31 %32 None
               OpBranch %33
         %33 = OpLabel
         %34 = OpLoad %10 %5
         %35 = OpSLessThan %14 %34 %15
               OpBranchConditional %35 %36 %31
         %36 = OpLabel
         %37 = OpLoad %10 %5
         %38 = OpSLessThan %14 %37 %16
               OpSelectionMerge %39 None
               OpBranchConditional %38 %40 %39
         %40 = OpLabel
               OpStore %6 %12
               OpBranch %41
         %41 = OpLabel
               OpLoopMerge %42 %43 None
               OpBranch %44
         %44 = OpLabel
         %45 = OpLoad %10 %6
         %46 = OpSLessThan %14 %45 %17
               OpBranchConditional %46 %47 %42
         %47 = OpLabel
               OpBranch %43
         %43 = OpLabel
         %48 = OpLoad %10 %6
         %49 = OpIAdd %10 %48 %18
               OpStore %6 %49
               OpBranch %41
         %42 = OpLabel
               OpBranch %39
         %39 = OpLabel
               OpStore %7 %12
               OpBranch %50
         %50 = OpLabel
               OpLoopMerge %51 %52 None
               OpBranch %53
         %53 = OpLabel
         %54 = OpLoad %10 %7
         %55 = OpSLessThan %14 %54 %17
               OpBranchConditional %55 %56 %51
         %56 = OpLabel
               OpBranch %52
         %52 = OpLabel
         %57 = OpLoad %10 %7
         %58 = OpIAdd %10 %57 %18
               OpStore %7 %58
               OpBranch %50
         %51 = OpLabel
               OpBranch %32
         %32 = OpLabel
         %59 = OpLoad %10 %5
         %60 = OpIAdd %10 %59 %18
               OpStore %5 %60
               OpBranch %30
         %31 = OpLabel
               OpBranch %25
         %25 = OpLabel
         %61 = OpLoad %10 %4
         %62 = OpIAdd %10 %61 %18
               OpStore %4 %62
               OpBranch %23
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
  const Function* f = spvtest::GetFunction(module, 2);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  EXPECT_EQ(ld.NumLoops(), 4u);

  // Invalid basic block id.
  EXPECT_EQ(ld[0u], nullptr);
  // Not in a loop.
  EXPECT_EQ(ld[22], nullptr);

  // Check that we can map basic block to the correct loop.
  // The following block ids do not belong to a loop.
  for (uint32_t bb_id : {22, 24}) EXPECT_EQ(ld[bb_id], nullptr);

  {
    std::unordered_set<uint32_t> basic_block_in_loop = {
        {23, 26, 29, 30, 33, 36, 40, 41, 44, 47, 43,
         42, 39, 50, 53, 56, 52, 51, 32, 31, 25}};
    Loop* loop = ld[23];
    CheckLoopBlocks(loop, &basic_block_in_loop);

    EXPECT_TRUE(loop->HasNestedLoops());
    EXPECT_FALSE(loop->IsNested());
    EXPECT_EQ(loop->GetDepth(), 1u);
    EXPECT_EQ(std::distance(loop->begin(), loop->end()), 1u);
    EXPECT_EQ(loop->GetPreHeaderBlock(), spvtest::GetBasicBlock(f, 22));
    EXPECT_EQ(loop->GetHeaderBlock(), spvtest::GetBasicBlock(f, 23));
    EXPECT_EQ(loop->GetLatchBlock(), spvtest::GetBasicBlock(f, 25));
    EXPECT_EQ(loop->GetMergeBlock(), spvtest::GetBasicBlock(f, 24));
    EXPECT_FALSE(loop->IsInsideLoop(loop->GetMergeBlock()));
    EXPECT_FALSE(loop->IsInsideLoop(loop->GetPreHeaderBlock()));
  }

  {
    std::unordered_set<uint32_t> basic_block_in_loop = {
        {30, 33, 36, 40, 41, 44, 47, 43, 42, 39, 50, 53, 56, 52, 51, 32}};
    Loop* loop = ld[30];
    CheckLoopBlocks(loop, &basic_block_in_loop);

    EXPECT_TRUE(loop->HasNestedLoops());
    EXPECT_TRUE(loop->IsNested());
    EXPECT_EQ(loop->GetDepth(), 2u);
    EXPECT_EQ(std::distance(loop->begin(), loop->end()), 2u);
    EXPECT_EQ(loop->GetPreHeaderBlock(), spvtest::GetBasicBlock(f, 29));
    EXPECT_EQ(loop->GetHeaderBlock(), spvtest::GetBasicBlock(f, 30));
    EXPECT_EQ(loop->GetLatchBlock(), spvtest::GetBasicBlock(f, 32));
    EXPECT_EQ(loop->GetMergeBlock(), spvtest::GetBasicBlock(f, 31));
    EXPECT_FALSE(loop->IsInsideLoop(loop->GetMergeBlock()));
    EXPECT_FALSE(loop->IsInsideLoop(loop->GetPreHeaderBlock()));
  }

  {
    std::unordered_set<uint32_t> basic_block_in_loop = {{41, 44, 47, 43}};
    Loop* loop = ld[41];
    CheckLoopBlocks(loop, &basic_block_in_loop);

    EXPECT_FALSE(loop->HasNestedLoops());
    EXPECT_TRUE(loop->IsNested());
    EXPECT_EQ(loop->GetDepth(), 3u);
    EXPECT_EQ(std::distance(loop->begin(), loop->end()), 0u);
    EXPECT_EQ(loop->GetPreHeaderBlock(), spvtest::GetBasicBlock(f, 40));
    EXPECT_EQ(loop->GetHeaderBlock(), spvtest::GetBasicBlock(f, 41));
    EXPECT_EQ(loop->GetLatchBlock(), spvtest::GetBasicBlock(f, 43));
    EXPECT_EQ(loop->GetMergeBlock(), spvtest::GetBasicBlock(f, 42));
    EXPECT_FALSE(loop->IsInsideLoop(loop->GetMergeBlock()));
    EXPECT_FALSE(loop->IsInsideLoop(loop->GetPreHeaderBlock()));
  }

  {
    std::unordered_set<uint32_t> basic_block_in_loop = {{50, 53, 56, 52}};
    Loop* loop = ld[50];
    CheckLoopBlocks(loop, &basic_block_in_loop);

    EXPECT_FALSE(loop->HasNestedLoops());
    EXPECT_TRUE(loop->IsNested());
    EXPECT_EQ(loop->GetDepth(), 3u);
    EXPECT_EQ(std::distance(loop->begin(), loop->end()), 0u);
    EXPECT_EQ(loop->GetPreHeaderBlock(), spvtest::GetBasicBlock(f, 39));
    EXPECT_EQ(loop->GetHeaderBlock(), spvtest::GetBasicBlock(f, 50));
    EXPECT_EQ(loop->GetLatchBlock(), spvtest::GetBasicBlock(f, 52));
    EXPECT_EQ(loop->GetMergeBlock(), spvtest::GetBasicBlock(f, 51));
    EXPECT_FALSE(loop->IsInsideLoop(loop->GetMergeBlock()));
    EXPECT_FALSE(loop->IsInsideLoop(loop->GetPreHeaderBlock()));
  }

  // Make sure LoopDescriptor gives us the inner most loop when we query for
  // loops.
  for (const BasicBlock& bb : *f) {
    if (Loop* loop = ld[&bb]) {
      for (Loop& sub_loop :
           make_range(++TreeDFIterator<Loop>(loop), TreeDFIterator<Loop>())) {
        EXPECT_FALSE(sub_loop.IsInsideLoop(bb.id()));
      }
    }
  }
}

/*
Generated from the following GLSL
#version 330 core
layout(location = 0) out vec4 c;
void main() {
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 11; ++j) {
      for (int k = 0; k < 11; ++k) {}
    }
    for (int k = 0; k < 12; ++k) {}
  }
}
*/
TEST_F(PassClassTest, LoopParentTest) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 330
               OpName %2 "main"
               OpName %4 "i"
               OpName %5 "j"
               OpName %6 "k"
               OpName %7 "k"
               OpName %3 "c"
               OpDecorate %3 Location 0
          %8 = OpTypeVoid
          %9 = OpTypeFunction %8
         %10 = OpTypeInt 32 1
         %11 = OpTypePointer Function %10
         %12 = OpConstant %10 0
         %13 = OpConstant %10 10
         %14 = OpTypeBool
         %15 = OpConstant %10 11
         %16 = OpConstant %10 1
         %17 = OpConstant %10 12
         %18 = OpTypeFloat 32
         %19 = OpTypeVector %18 4
         %20 = OpTypePointer Output %19
          %3 = OpVariable %20 Output
          %2 = OpFunction %8 None %9
         %21 = OpLabel
          %4 = OpVariable %11 Function
          %5 = OpVariable %11 Function
          %6 = OpVariable %11 Function
          %7 = OpVariable %11 Function
               OpStore %4 %12
               OpBranch %22
         %22 = OpLabel
               OpLoopMerge %23 %24 None
               OpBranch %25
         %25 = OpLabel
         %26 = OpLoad %10 %4
         %27 = OpSLessThan %14 %26 %13
               OpBranchConditional %27 %28 %23
         %28 = OpLabel
               OpStore %5 %12
               OpBranch %29
         %29 = OpLabel
               OpLoopMerge %30 %31 None
               OpBranch %32
         %32 = OpLabel
         %33 = OpLoad %10 %5
         %34 = OpSLessThan %14 %33 %15
               OpBranchConditional %34 %35 %30
         %35 = OpLabel
               OpStore %6 %12
               OpBranch %36
         %36 = OpLabel
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %40 = OpLoad %10 %6
         %41 = OpSLessThan %14 %40 %15
               OpBranchConditional %41 %42 %37
         %42 = OpLabel
               OpBranch %38
         %38 = OpLabel
         %43 = OpLoad %10 %6
         %44 = OpIAdd %10 %43 %16
               OpStore %6 %44
               OpBranch %36
         %37 = OpLabel
               OpBranch %31
         %31 = OpLabel
         %45 = OpLoad %10 %5
         %46 = OpIAdd %10 %45 %16
               OpStore %5 %46
               OpBranch %29
         %30 = OpLabel
               OpStore %7 %12
               OpBranch %47
         %47 = OpLabel
               OpLoopMerge %48 %49 None
               OpBranch %50
         %50 = OpLabel
         %51 = OpLoad %10 %7
         %52 = OpSLessThan %14 %51 %17
               OpBranchConditional %52 %53 %48
         %53 = OpLabel
               OpBranch %49
         %49 = OpLabel
         %54 = OpLoad %10 %7
         %55 = OpIAdd %10 %54 %16
               OpStore %7 %55
               OpBranch %47
         %48 = OpLabel
               OpBranch %24
         %24 = OpLabel
         %56 = OpLoad %10 %4
         %57 = OpIAdd %10 %56 %16
               OpStore %4 %57
               OpBranch %22
         %23 = OpLabel
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
  const Function* f = spvtest::GetFunction(module, 2);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  EXPECT_EQ(ld.NumLoops(), 4u);

  {
    Loop& loop = *ld[22];
    EXPECT_TRUE(loop.HasNestedLoops());
    EXPECT_FALSE(loop.IsNested());
    EXPECT_EQ(loop.GetDepth(), 1u);
    EXPECT_EQ(loop.GetParent(), nullptr);
  }

  {
    Loop& loop = *ld[29];
    EXPECT_TRUE(loop.HasNestedLoops());
    EXPECT_TRUE(loop.IsNested());
    EXPECT_EQ(loop.GetDepth(), 2u);
    EXPECT_EQ(loop.GetParent(), ld[22]);
  }

  {
    Loop& loop = *ld[36];
    EXPECT_FALSE(loop.HasNestedLoops());
    EXPECT_TRUE(loop.IsNested());
    EXPECT_EQ(loop.GetDepth(), 3u);
    EXPECT_EQ(loop.GetParent(), ld[29]);
  }

  {
    Loop& loop = *ld[47];
    EXPECT_FALSE(loop.HasNestedLoops());
    EXPECT_TRUE(loop.IsNested());
    EXPECT_EQ(loop.GetDepth(), 2u);
    EXPECT_EQ(loop.GetParent(), ld[22]);
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store
The preheader of loop %33 and %41 were removed as well.

#version 330 core
void main() {
  int a = 0;
  for (int i = 0; i < 10; ++i) {
    if (i == 0) {
      a = 1;
    } else {
      a = 2;
    }
    for (int j = 0; j < 11; ++j) {
      a++;
    }
  }
  for (int k = 0; k < 12; ++k) {}
}
*/
TEST_F(PassClassTest, CreatePreheaderTest) {
  const std::string text = R"(
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
         %11 = OpConstant %5 2
         %12 = OpConstant %5 11
         %13 = OpConstant %5 12
         %14 = OpUndef %5
          %2 = OpFunction %3 None %4
         %15 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %17 = OpPhi %5 %7 %15 %18 %19
         %20 = OpPhi %5 %7 %15 %21 %19
         %22 = OpPhi %5 %14 %15 %23 %19
               OpLoopMerge %41 %19 None
               OpBranch %25
         %25 = OpLabel
         %26 = OpSLessThan %9 %20 %8
               OpBranchConditional %26 %27 %41
         %27 = OpLabel
         %28 = OpIEqual %9 %20 %7
               OpSelectionMerge %33 None
               OpBranchConditional %28 %30 %31
         %30 = OpLabel
               OpBranch %33
         %31 = OpLabel
               OpBranch %33
         %33 = OpLabel
         %18 = OpPhi %5 %10 %30 %11 %31 %34 %35
         %23 = OpPhi %5 %7 %30 %7 %31 %36 %35
               OpLoopMerge %37 %35 None
               OpBranch %38
         %38 = OpLabel
         %39 = OpSLessThan %9 %23 %12
               OpBranchConditional %39 %40 %37
         %40 = OpLabel
         %34 = OpIAdd %5 %18 %10
               OpBranch %35
         %35 = OpLabel
         %36 = OpIAdd %5 %23 %10
               OpBranch %33
         %37 = OpLabel
               OpBranch %19
         %19 = OpLabel
         %21 = OpIAdd %5 %20 %10
               OpBranch %16
         %41 = OpLabel
         %42 = OpPhi %5 %7 %25 %43 %44
               OpLoopMerge %45 %44 None
               OpBranch %46
         %46 = OpLabel
         %47 = OpSLessThan %9 %42 %13
               OpBranchConditional %47 %48 %45
         %48 = OpLabel
               OpBranch %44
         %44 = OpLabel
         %43 = OpIAdd %5 %42 %10
               OpBranch %41
         %45 = OpLabel
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
  const Function* f = spvtest::GetFunction(module, 2);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);
  // No invalidation of the cfg should occur during this test.
  CFG* cfg = context->cfg();

  EXPECT_EQ(ld.NumLoops(), 3u);

  {
    Loop& loop = *ld[16];
    EXPECT_TRUE(loop.HasNestedLoops());
    EXPECT_FALSE(loop.IsNested());
    EXPECT_EQ(loop.GetDepth(), 1u);
    EXPECT_EQ(loop.GetParent(), nullptr);
  }

  {
    Loop& loop = *ld[33];
    EXPECT_EQ(loop.GetPreHeaderBlock(), nullptr);
    EXPECT_NE(loop.GetOrCreatePreHeaderBlock(), nullptr);
    // Make sure the loop descriptor was properly updated.
    EXPECT_EQ(ld[loop.GetPreHeaderBlock()], ld[16]);
    {
      const std::vector<uint32_t>& preds =
          cfg->preds(loop.GetPreHeaderBlock()->id());
      std::unordered_set<uint32_t> pred_set(preds.begin(), preds.end());
      EXPECT_EQ(pred_set.size(), 2u);
      EXPECT_TRUE(pred_set.count(30));
      EXPECT_TRUE(pred_set.count(31));
      // Check the phi instructions.
      loop.GetPreHeaderBlock()->ForEachPhiInst([&pred_set](Instruction* phi) {
        for (uint32_t i = 1; i < phi->NumInOperands(); i += 2) {
          EXPECT_TRUE(pred_set.count(phi->GetSingleWordInOperand(i)));
        }
      });
    }
    {
      const std::vector<uint32_t>& preds =
          cfg->preds(loop.GetHeaderBlock()->id());
      std::unordered_set<uint32_t> pred_set(preds.begin(), preds.end());
      EXPECT_EQ(pred_set.size(), 2u);
      EXPECT_TRUE(pred_set.count(loop.GetPreHeaderBlock()->id()));
      EXPECT_TRUE(pred_set.count(35));
      // Check the phi instructions.
      loop.GetHeaderBlock()->ForEachPhiInst([&pred_set](Instruction* phi) {
        for (uint32_t i = 1; i < phi->NumInOperands(); i += 2) {
          EXPECT_TRUE(pred_set.count(phi->GetSingleWordInOperand(i)));
        }
      });
    }
  }

  {
    Loop& loop = *ld[41];
    EXPECT_EQ(loop.GetPreHeaderBlock(), nullptr);
    EXPECT_NE(loop.GetOrCreatePreHeaderBlock(), nullptr);
    EXPECT_EQ(ld[loop.GetPreHeaderBlock()], nullptr);
    EXPECT_EQ(cfg->preds(loop.GetPreHeaderBlock()->id()).size(), 1u);
    EXPECT_EQ(cfg->preds(loop.GetPreHeaderBlock()->id())[0], 25u);
    // Check the phi instructions.
    loop.GetPreHeaderBlock()->ForEachPhiInst([](Instruction* phi) {
      EXPECT_EQ(phi->NumInOperands(), 2u);
      EXPECT_EQ(phi->GetSingleWordInOperand(1), 25u);
    });
    {
      const std::vector<uint32_t>& preds =
          cfg->preds(loop.GetHeaderBlock()->id());
      std::unordered_set<uint32_t> pred_set(preds.begin(), preds.end());
      EXPECT_EQ(pred_set.size(), 2u);
      EXPECT_TRUE(pred_set.count(loop.GetPreHeaderBlock()->id()));
      EXPECT_TRUE(pred_set.count(44));
      // Check the phi instructions.
      loop.GetHeaderBlock()->ForEachPhiInst([&pred_set](Instruction* phi) {
        for (uint32_t i = 1; i < phi->NumInOperands(); i += 2) {
          EXPECT_TRUE(pred_set.count(phi->GetSingleWordInOperand(i)));
        }
      });
    }
  }

  // Make sure pre-header insertion leaves the module valid.
  std::vector<uint32_t> bin;
  context->module()->ToBinary(&bin, true);
  EXPECT_TRUE(Validate(bin));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
