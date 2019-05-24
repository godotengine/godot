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

#include <array>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/opt/dominator_analysis.h"
#include "source/opt/iterator.h"
#include "source/opt/pass.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/function_utils.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::UnorderedElementsAre;
using PassClassTest = PassTest<::testing::Test>;

// Check that x dominates y, and
//   if x != y then
//      x strictly dominates y and
//      y does not dominate x and
//      y does not strictly dominate x
//   if x == x then
//      x does not strictly dominate itself
void check_dominance(const DominatorAnalysisBase& dom_tree, const Function* fn,
                     uint32_t x, uint32_t y) {
  SCOPED_TRACE("Check dominance properties for Basic Block " +
               std::to_string(x) + " and " + std::to_string(y));
  EXPECT_TRUE(dom_tree.Dominates(spvtest::GetBasicBlock(fn, x),
                                 spvtest::GetBasicBlock(fn, y)));
  EXPECT_TRUE(dom_tree.Dominates(x, y));
  if (x == y) {
    EXPECT_FALSE(dom_tree.StrictlyDominates(x, x));
  } else {
    EXPECT_TRUE(dom_tree.StrictlyDominates(x, y));
    EXPECT_FALSE(dom_tree.Dominates(y, x));
    EXPECT_FALSE(dom_tree.StrictlyDominates(y, x));
  }
}

// Check that x does not dominates y and vise versa
void check_no_dominance(const DominatorAnalysisBase& dom_tree,
                        const Function* fn, uint32_t x, uint32_t y) {
  SCOPED_TRACE("Check no domination for Basic Block " + std::to_string(x) +
               " and " + std::to_string(y));
  EXPECT_FALSE(dom_tree.Dominates(spvtest::GetBasicBlock(fn, x),
                                  spvtest::GetBasicBlock(fn, y)));
  EXPECT_FALSE(dom_tree.Dominates(x, y));
  EXPECT_FALSE(dom_tree.StrictlyDominates(spvtest::GetBasicBlock(fn, x),
                                          spvtest::GetBasicBlock(fn, y)));
  EXPECT_FALSE(dom_tree.StrictlyDominates(x, y));

  EXPECT_FALSE(dom_tree.Dominates(spvtest::GetBasicBlock(fn, y),
                                  spvtest::GetBasicBlock(fn, x)));
  EXPECT_FALSE(dom_tree.Dominates(y, x));
  EXPECT_FALSE(dom_tree.StrictlyDominates(spvtest::GetBasicBlock(fn, y),
                                          spvtest::GetBasicBlock(fn, x)));
  EXPECT_FALSE(dom_tree.StrictlyDominates(y, x));
}

TEST_F(PassClassTest, DominatorSimpleCFG) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeBool
          %5 = OpTypeInt 32 0
          %6 = OpConstant %5 0
          %7 = OpConstantFalse %4
          %8 = OpConstantTrue %4
          %9 = OpConstant %5 1
          %1 = OpFunction %2 None %3
         %10 = OpLabel
               OpBranch %11
         %11 = OpLabel
               OpSwitch %6 %12 1 %13
         %12 = OpLabel
               OpBranch %14
         %13 = OpLabel
               OpBranch %14
         %14 = OpLabel
               OpBranchConditional %8 %11 %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_0, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const Function* fn = spvtest::GetFunction(module, 1);
  const BasicBlock* entry = spvtest::GetBasicBlock(fn, 10);
  EXPECT_EQ(entry, fn->entry().get())
      << "The entry node is not the expected one";

  // Test normal dominator tree
  {
    DominatorAnalysis dom_tree;
    const CFG& cfg = *context->cfg();
    dom_tree.InitializeTree(cfg, fn);

    // Inspect the actual tree
    DominatorTree& tree = dom_tree.GetDomTree();
    EXPECT_EQ(tree.GetRoot()->bb_, cfg.pseudo_entry_block());
    EXPECT_TRUE(
        dom_tree.Dominates(cfg.pseudo_entry_block()->id(), entry->id()));

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12, 13, 14, 15})
      check_dominance(dom_tree, fn, id, id);

    check_dominance(dom_tree, fn, 10, 11);
    check_dominance(dom_tree, fn, 10, 12);
    check_dominance(dom_tree, fn, 10, 13);
    check_dominance(dom_tree, fn, 10, 14);
    check_dominance(dom_tree, fn, 10, 15);

    check_dominance(dom_tree, fn, 11, 12);
    check_dominance(dom_tree, fn, 11, 13);
    check_dominance(dom_tree, fn, 11, 14);
    check_dominance(dom_tree, fn, 11, 15);

    check_dominance(dom_tree, fn, 14, 15);

    check_no_dominance(dom_tree, fn, 12, 13);
    check_no_dominance(dom_tree, fn, 12, 14);
    check_no_dominance(dom_tree, fn, 13, 14);

    // check with some invalid inputs
    EXPECT_FALSE(dom_tree.Dominates(nullptr, entry));
    EXPECT_FALSE(dom_tree.Dominates(entry, nullptr));
    EXPECT_FALSE(dom_tree.Dominates(static_cast<BasicBlock*>(nullptr),
                                    static_cast<BasicBlock*>(nullptr)));
    EXPECT_FALSE(dom_tree.Dominates(10, 1));
    EXPECT_FALSE(dom_tree.Dominates(1, 10));
    EXPECT_FALSE(dom_tree.Dominates(1, 1));

    EXPECT_FALSE(dom_tree.StrictlyDominates(nullptr, entry));
    EXPECT_FALSE(dom_tree.StrictlyDominates(entry, nullptr));
    EXPECT_FALSE(dom_tree.StrictlyDominates(nullptr, nullptr));
    EXPECT_FALSE(dom_tree.StrictlyDominates(10, 1));
    EXPECT_FALSE(dom_tree.StrictlyDominates(1, 10));
    EXPECT_FALSE(dom_tree.StrictlyDominates(1, 1));

    EXPECT_EQ(dom_tree.ImmediateDominator(cfg.pseudo_entry_block()), nullptr);
    EXPECT_EQ(dom_tree.ImmediateDominator(entry), cfg.pseudo_entry_block());
    EXPECT_EQ(dom_tree.ImmediateDominator(nullptr), nullptr);

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 11)),
              spvtest::GetBasicBlock(fn, 10));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 12)),
              spvtest::GetBasicBlock(fn, 11));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 13)),
              spvtest::GetBasicBlock(fn, 11));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 14)),
              spvtest::GetBasicBlock(fn, 11));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 15)),
              spvtest::GetBasicBlock(fn, 14));
  }

  // Test post dominator tree
  {
    PostDominatorAnalysis dom_tree;
    const CFG& cfg = *context->cfg();
    dom_tree.InitializeTree(cfg, fn);

    // Inspect the actual tree
    DominatorTree& tree = dom_tree.GetDomTree();
    EXPECT_EQ(tree.GetRoot()->bb_, cfg.pseudo_exit_block());
    EXPECT_TRUE(dom_tree.Dominates(cfg.pseudo_exit_block()->id(), 15));

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12, 13, 14, 15})
      check_dominance(dom_tree, fn, id, id);

    check_dominance(dom_tree, fn, 14, 10);
    check_dominance(dom_tree, fn, 14, 11);
    check_dominance(dom_tree, fn, 14, 12);
    check_dominance(dom_tree, fn, 14, 13);

    check_dominance(dom_tree, fn, 15, 10);
    check_dominance(dom_tree, fn, 15, 11);
    check_dominance(dom_tree, fn, 15, 12);
    check_dominance(dom_tree, fn, 15, 13);
    check_dominance(dom_tree, fn, 15, 14);

    check_no_dominance(dom_tree, fn, 13, 12);
    check_no_dominance(dom_tree, fn, 12, 11);
    check_no_dominance(dom_tree, fn, 13, 11);

    // check with some invalid inputs
    EXPECT_FALSE(dom_tree.Dominates(nullptr, entry));
    EXPECT_FALSE(dom_tree.Dominates(entry, nullptr));
    EXPECT_FALSE(dom_tree.Dominates(static_cast<BasicBlock*>(nullptr),
                                    static_cast<BasicBlock*>(nullptr)));
    EXPECT_FALSE(dom_tree.Dominates(10, 1));
    EXPECT_FALSE(dom_tree.Dominates(1, 10));
    EXPECT_FALSE(dom_tree.Dominates(1, 1));

    EXPECT_FALSE(dom_tree.StrictlyDominates(nullptr, entry));
    EXPECT_FALSE(dom_tree.StrictlyDominates(entry, nullptr));
    EXPECT_FALSE(dom_tree.StrictlyDominates(nullptr, nullptr));
    EXPECT_FALSE(dom_tree.StrictlyDominates(10, 1));
    EXPECT_FALSE(dom_tree.StrictlyDominates(1, 10));
    EXPECT_FALSE(dom_tree.StrictlyDominates(1, 1));

    EXPECT_EQ(dom_tree.ImmediateDominator(nullptr), nullptr);

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 11)),
              spvtest::GetBasicBlock(fn, 14));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 12)),
              spvtest::GetBasicBlock(fn, 14));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 13)),
              spvtest::GetBasicBlock(fn, 14));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 14)),
              spvtest::GetBasicBlock(fn, 15));

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 15)),
              cfg.pseudo_exit_block());

    EXPECT_EQ(dom_tree.ImmediateDominator(cfg.pseudo_exit_block()), nullptr);
  }
}

TEST_F(PassClassTest, DominatorIrreducibleCFG) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeBool
          %5 = OpTypeInt 32 0
          %6 = OpConstantFalse %4
          %7 = OpConstantTrue %4
          %1 = OpFunction %2 None %3
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpBranchConditional %7 %10 %11
         %10 = OpLabel
               OpBranch %11
         %11 = OpLabel
               OpBranchConditional %7 %10 %12
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_0, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const Function* fn = spvtest::GetFunction(module, 1);

  const BasicBlock* entry = spvtest::GetBasicBlock(fn, 8);
  EXPECT_EQ(entry, fn->entry().get())
      << "The entry node is not the expected one";

  // Check normal dominator tree
  {
    DominatorAnalysis dom_tree;
    const CFG& cfg = *context->cfg();
    dom_tree.InitializeTree(cfg, fn);

    // Inspect the actual tree
    DominatorTree& tree = dom_tree.GetDomTree();
    EXPECT_EQ(tree.GetRoot()->bb_, cfg.pseudo_entry_block());
    EXPECT_TRUE(
        dom_tree.Dominates(cfg.pseudo_entry_block()->id(), entry->id()));

    // (strict) dominance checks
    for (uint32_t id : {8, 9, 10, 11, 12})
      check_dominance(dom_tree, fn, id, id);

    check_dominance(dom_tree, fn, 8, 9);
    check_dominance(dom_tree, fn, 8, 10);
    check_dominance(dom_tree, fn, 8, 11);
    check_dominance(dom_tree, fn, 8, 12);

    check_dominance(dom_tree, fn, 9, 10);
    check_dominance(dom_tree, fn, 9, 11);
    check_dominance(dom_tree, fn, 9, 12);

    check_dominance(dom_tree, fn, 11, 12);

    check_no_dominance(dom_tree, fn, 10, 11);

    EXPECT_EQ(dom_tree.ImmediateDominator(cfg.pseudo_entry_block()), nullptr);
    EXPECT_EQ(dom_tree.ImmediateDominator(entry), cfg.pseudo_entry_block());

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 9)),
              spvtest::GetBasicBlock(fn, 8));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 10)),
              spvtest::GetBasicBlock(fn, 9));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 11)),
              spvtest::GetBasicBlock(fn, 9));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 12)),
              spvtest::GetBasicBlock(fn, 11));
  }

  // Check post dominator tree
  {
    PostDominatorAnalysis dom_tree;
    const CFG& cfg = *context->cfg();
    dom_tree.InitializeTree(cfg, fn);

    // Inspect the actual tree
    DominatorTree& tree = dom_tree.GetDomTree();
    EXPECT_EQ(tree.GetRoot()->bb_, cfg.pseudo_exit_block());
    EXPECT_TRUE(dom_tree.Dominates(cfg.pseudo_exit_block()->id(), 12));

    // (strict) dominance checks
    for (uint32_t id : {8, 9, 10, 11, 12})
      check_dominance(dom_tree, fn, id, id);

    check_dominance(dom_tree, fn, 12, 8);
    check_dominance(dom_tree, fn, 12, 10);
    check_dominance(dom_tree, fn, 12, 11);
    check_dominance(dom_tree, fn, 12, 12);

    check_dominance(dom_tree, fn, 11, 8);
    check_dominance(dom_tree, fn, 11, 9);
    check_dominance(dom_tree, fn, 11, 10);

    check_dominance(dom_tree, fn, 9, 8);

    EXPECT_EQ(dom_tree.ImmediateDominator(entry),
              spvtest::GetBasicBlock(fn, 9));

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 9)),
              spvtest::GetBasicBlock(fn, 11));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 10)),
              spvtest::GetBasicBlock(fn, 11));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 11)),
              spvtest::GetBasicBlock(fn, 12));

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 12)),
              cfg.pseudo_exit_block());

    EXPECT_EQ(dom_tree.ImmediateDominator(cfg.pseudo_exit_block()), nullptr);
  }
}

TEST_F(PassClassTest, DominatorLoopToSelf) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeBool
          %5 = OpTypeInt 32 0
          %6 = OpConstant %5 0
          %7 = OpConstantFalse %4
          %8 = OpConstantTrue %4
          %9 = OpConstant %5 1
          %1 = OpFunction %2 None %3
         %10 = OpLabel
               OpBranch %11
         %11 = OpLabel
               OpSwitch %6 %12 1 %11
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_0, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const Function* fn = spvtest::GetFunction(module, 1);

  const BasicBlock* entry = spvtest::GetBasicBlock(fn, 10);
  EXPECT_EQ(entry, fn->entry().get())
      << "The entry node is not the expected one";

  // Check normal dominator tree
  {
    DominatorAnalysis dom_tree;
    const CFG& cfg = *context->cfg();
    dom_tree.InitializeTree(cfg, fn);

    // Inspect the actual tree
    DominatorTree& tree = dom_tree.GetDomTree();
    EXPECT_EQ(tree.GetRoot()->bb_, cfg.pseudo_entry_block());
    EXPECT_TRUE(
        dom_tree.Dominates(cfg.pseudo_entry_block()->id(), entry->id()));

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12}) check_dominance(dom_tree, fn, id, id);

    check_dominance(dom_tree, fn, 10, 11);
    check_dominance(dom_tree, fn, 10, 12);
    check_dominance(dom_tree, fn, 11, 12);

    EXPECT_EQ(dom_tree.ImmediateDominator(cfg.pseudo_entry_block()), nullptr);
    EXPECT_EQ(dom_tree.ImmediateDominator(entry), cfg.pseudo_entry_block());

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 11)),
              spvtest::GetBasicBlock(fn, 10));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 12)),
              spvtest::GetBasicBlock(fn, 11));

    std::array<uint32_t, 3> node_order = {{10, 11, 12}};
    {
      // Test dominator tree iteration order.
      DominatorTree::iterator node_it = dom_tree.GetDomTree().begin();
      DominatorTree::iterator node_end = dom_tree.GetDomTree().end();
      for (uint32_t id : node_order) {
        EXPECT_NE(node_it, node_end);
        EXPECT_EQ(node_it->id(), id);
        node_it++;
      }
      EXPECT_EQ(node_it, node_end);
    }
    {
      // Same as above, but with const iterators.
      DominatorTree::const_iterator node_it = dom_tree.GetDomTree().cbegin();
      DominatorTree::const_iterator node_end = dom_tree.GetDomTree().cend();
      for (uint32_t id : node_order) {
        EXPECT_NE(node_it, node_end);
        EXPECT_EQ(node_it->id(), id);
        node_it++;
      }
      EXPECT_EQ(node_it, node_end);
    }
    {
      // Test dominator tree iteration order.
      DominatorTree::post_iterator node_it = dom_tree.GetDomTree().post_begin();
      DominatorTree::post_iterator node_end = dom_tree.GetDomTree().post_end();
      for (uint32_t id : make_range(node_order.rbegin(), node_order.rend())) {
        EXPECT_NE(node_it, node_end);
        EXPECT_EQ(node_it->id(), id);
        node_it++;
      }
      EXPECT_EQ(node_it, node_end);
    }
    {
      // Same as above, but with const iterators.
      DominatorTree::const_post_iterator node_it =
          dom_tree.GetDomTree().post_cbegin();
      DominatorTree::const_post_iterator node_end =
          dom_tree.GetDomTree().post_cend();
      for (uint32_t id : make_range(node_order.rbegin(), node_order.rend())) {
        EXPECT_NE(node_it, node_end);
        EXPECT_EQ(node_it->id(), id);
        node_it++;
      }
      EXPECT_EQ(node_it, node_end);
    }
  }

  // Check post dominator tree
  {
    PostDominatorAnalysis dom_tree;
    const CFG& cfg = *context->cfg();
    dom_tree.InitializeTree(cfg, fn);

    // Inspect the actual tree
    DominatorTree& tree = dom_tree.GetDomTree();
    EXPECT_EQ(tree.GetRoot()->bb_, cfg.pseudo_exit_block());
    EXPECT_TRUE(dom_tree.Dominates(cfg.pseudo_exit_block()->id(), 12));

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12}) check_dominance(dom_tree, fn, id, id);

    check_dominance(dom_tree, fn, 12, 10);
    check_dominance(dom_tree, fn, 12, 11);
    check_dominance(dom_tree, fn, 12, 12);

    EXPECT_EQ(dom_tree.ImmediateDominator(entry),
              spvtest::GetBasicBlock(fn, 11));

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 11)),
              spvtest::GetBasicBlock(fn, 12));

    EXPECT_EQ(dom_tree.ImmediateDominator(cfg.pseudo_exit_block()), nullptr);

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 12)),
              cfg.pseudo_exit_block());

    std::array<uint32_t, 3> node_order = {{12, 11, 10}};
    {
      // Test dominator tree iteration order.
      DominatorTree::iterator node_it = tree.begin();
      DominatorTree::iterator node_end = tree.end();
      for (uint32_t id : node_order) {
        EXPECT_NE(node_it, node_end);
        EXPECT_EQ(node_it->id(), id);
        node_it++;
      }
      EXPECT_EQ(node_it, node_end);
    }
    {
      // Same as above, but with const iterators.
      DominatorTree::const_iterator node_it = tree.cbegin();
      DominatorTree::const_iterator node_end = tree.cend();
      for (uint32_t id : node_order) {
        EXPECT_NE(node_it, node_end);
        EXPECT_EQ(node_it->id(), id);
        node_it++;
      }
      EXPECT_EQ(node_it, node_end);
    }
    {
      // Test dominator tree iteration order.
      DominatorTree::post_iterator node_it = dom_tree.GetDomTree().post_begin();
      DominatorTree::post_iterator node_end = dom_tree.GetDomTree().post_end();
      for (uint32_t id : make_range(node_order.rbegin(), node_order.rend())) {
        EXPECT_NE(node_it, node_end);
        EXPECT_EQ(node_it->id(), id);
        node_it++;
      }
      EXPECT_EQ(node_it, node_end);
    }
    {
      // Same as above, but with const iterators.
      DominatorTree::const_post_iterator node_it =
          dom_tree.GetDomTree().post_cbegin();
      DominatorTree::const_post_iterator node_end =
          dom_tree.GetDomTree().post_cend();
      for (uint32_t id : make_range(node_order.rbegin(), node_order.rend())) {
        EXPECT_NE(node_it, node_end);
        EXPECT_EQ(node_it->id(), id);
        node_it++;
      }
      EXPECT_EQ(node_it, node_end);
    }
  }
}

TEST_F(PassClassTest, DominatorUnreachableInLoop) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeBool
          %5 = OpTypeInt 32 0
          %6 = OpConstant %5 0
          %7 = OpConstantFalse %4
          %8 = OpConstantTrue %4
          %9 = OpConstant %5 1
          %1 = OpFunction %2 None %3
         %10 = OpLabel
               OpBranch %11
         %11 = OpLabel
               OpSwitch %6 %12 1 %13
         %12 = OpLabel
               OpBranch %14
         %13 = OpLabel
               OpUnreachable
         %14 = OpLabel
               OpBranchConditional %8 %11 %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_0, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const Function* fn = spvtest::GetFunction(module, 1);

  const BasicBlock* entry = spvtest::GetBasicBlock(fn, 10);
  EXPECT_EQ(entry, fn->entry().get())
      << "The entry node is not the expected one";

  // Check normal dominator tree
  {
    DominatorAnalysis dom_tree;
    const CFG& cfg = *context->cfg();
    dom_tree.InitializeTree(cfg, fn);

    // Inspect the actual tree
    DominatorTree& tree = dom_tree.GetDomTree();
    EXPECT_EQ(tree.GetRoot()->bb_, cfg.pseudo_entry_block());
    EXPECT_TRUE(
        dom_tree.Dominates(cfg.pseudo_entry_block()->id(), entry->id()));

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12, 13, 14, 15})
      check_dominance(dom_tree, fn, id, id);

    check_dominance(dom_tree, fn, 10, 11);
    check_dominance(dom_tree, fn, 10, 13);
    check_dominance(dom_tree, fn, 10, 12);
    check_dominance(dom_tree, fn, 10, 14);
    check_dominance(dom_tree, fn, 10, 15);

    check_dominance(dom_tree, fn, 11, 12);
    check_dominance(dom_tree, fn, 11, 13);
    check_dominance(dom_tree, fn, 11, 14);
    check_dominance(dom_tree, fn, 11, 15);

    check_dominance(dom_tree, fn, 12, 14);
    check_dominance(dom_tree, fn, 12, 15);

    check_dominance(dom_tree, fn, 14, 15);

    check_no_dominance(dom_tree, fn, 13, 12);
    check_no_dominance(dom_tree, fn, 13, 14);
    check_no_dominance(dom_tree, fn, 13, 15);

    EXPECT_EQ(dom_tree.ImmediateDominator(cfg.pseudo_entry_block()), nullptr);
    EXPECT_EQ(dom_tree.ImmediateDominator(entry), cfg.pseudo_entry_block());

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 11)),
              spvtest::GetBasicBlock(fn, 10));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 12)),
              spvtest::GetBasicBlock(fn, 11));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 13)),
              spvtest::GetBasicBlock(fn, 11));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 14)),
              spvtest::GetBasicBlock(fn, 12));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 15)),
              spvtest::GetBasicBlock(fn, 14));
  }

  // Check post dominator tree.
  {
    PostDominatorAnalysis dom_tree;
    const CFG& cfg = *context->cfg();
    dom_tree.InitializeTree(cfg, fn);

    // (strict) dominance checks.
    for (uint32_t id : {10, 11, 12, 13, 14, 15})
      check_dominance(dom_tree, fn, id, id);

    check_no_dominance(dom_tree, fn, 15, 10);
    check_no_dominance(dom_tree, fn, 15, 11);
    check_no_dominance(dom_tree, fn, 15, 12);
    check_no_dominance(dom_tree, fn, 15, 13);
    check_no_dominance(dom_tree, fn, 15, 14);

    check_dominance(dom_tree, fn, 14, 12);

    check_no_dominance(dom_tree, fn, 13, 10);
    check_no_dominance(dom_tree, fn, 13, 11);
    check_no_dominance(dom_tree, fn, 13, 12);
    check_no_dominance(dom_tree, fn, 13, 14);
    check_no_dominance(dom_tree, fn, 13, 15);

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 10)),
              spvtest::GetBasicBlock(fn, 11));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 12)),
              spvtest::GetBasicBlock(fn, 14));

    EXPECT_EQ(dom_tree.ImmediateDominator(cfg.pseudo_exit_block()), nullptr);

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 15)),
              cfg.pseudo_exit_block());
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 13)),
              cfg.pseudo_exit_block());
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 14)),
              cfg.pseudo_exit_block());
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 11)),
              cfg.pseudo_exit_block());
  }
}

TEST_F(PassClassTest, DominatorInfinitLoop) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeBool
          %5 = OpTypeInt 32 0
          %6 = OpConstant %5 0
          %7 = OpConstantFalse %4
          %8 = OpConstantTrue %4
          %9 = OpConstant %5 1
          %1 = OpFunction %2 None %3
         %10 = OpLabel
               OpBranch %11
         %11 = OpLabel
               OpSwitch %6 %12 1 %13
         %12 = OpLabel
               OpReturn
         %13 = OpLabel
               OpBranch %13
               OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_0, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const Function* fn = spvtest::GetFunction(module, 1);

  const BasicBlock* entry = spvtest::GetBasicBlock(fn, 10);
  EXPECT_EQ(entry, fn->entry().get())
      << "The entry node is not the expected one";
  // Check normal dominator tree
  {
    DominatorAnalysis dom_tree;
    const CFG& cfg = *context->cfg();
    dom_tree.InitializeTree(cfg, fn);

    // Inspect the actual tree
    DominatorTree& tree = dom_tree.GetDomTree();
    EXPECT_EQ(tree.GetRoot()->bb_, cfg.pseudo_entry_block());
    EXPECT_TRUE(
        dom_tree.Dominates(cfg.pseudo_entry_block()->id(), entry->id()));

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12, 13}) check_dominance(dom_tree, fn, id, id);

    check_dominance(dom_tree, fn, 10, 11);
    check_dominance(dom_tree, fn, 10, 12);
    check_dominance(dom_tree, fn, 10, 13);

    check_dominance(dom_tree, fn, 11, 12);
    check_dominance(dom_tree, fn, 11, 13);

    check_no_dominance(dom_tree, fn, 13, 12);

    EXPECT_EQ(dom_tree.ImmediateDominator(cfg.pseudo_entry_block()), nullptr);
    EXPECT_EQ(dom_tree.ImmediateDominator(entry), cfg.pseudo_entry_block());

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 11)),
              spvtest::GetBasicBlock(fn, 10));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 12)),
              spvtest::GetBasicBlock(fn, 11));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 13)),
              spvtest::GetBasicBlock(fn, 11));
  }

  // Check post dominator tree
  {
    PostDominatorAnalysis dom_tree;
    const CFG& cfg = *context->cfg();
    dom_tree.InitializeTree(cfg, fn);

    // Inspect the actual tree
    DominatorTree& tree = dom_tree.GetDomTree();
    EXPECT_EQ(tree.GetRoot()->bb_, cfg.pseudo_exit_block());
    EXPECT_TRUE(dom_tree.Dominates(cfg.pseudo_exit_block()->id(), 12));

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12}) check_dominance(dom_tree, fn, id, id);

    check_dominance(dom_tree, fn, 12, 11);
    check_dominance(dom_tree, fn, 12, 10);

    // 13 should be completely out of tree as it's unreachable from exit nodes
    check_no_dominance(dom_tree, fn, 12, 13);
    check_no_dominance(dom_tree, fn, 11, 13);
    check_no_dominance(dom_tree, fn, 10, 13);

    EXPECT_EQ(dom_tree.ImmediateDominator(cfg.pseudo_exit_block()), nullptr);

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 12)),
              cfg.pseudo_exit_block());

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 10)),
              spvtest::GetBasicBlock(fn, 11));

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 11)),
              spvtest::GetBasicBlock(fn, 12));
  }
}

TEST_F(PassClassTest, DominatorUnreachableFromEntry) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Addresses
               OpCapability Kernel
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeBool
          %5 = OpTypeInt 32 0
          %6 = OpConstantFalse %4
          %7 = OpConstantTrue %4
          %1 = OpFunction %2 None %3
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpReturn
         %10 = OpLabel
               OpBranch %9
               OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_0, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const Function* fn = spvtest::GetFunction(module, 1);

  const BasicBlock* entry = spvtest::GetBasicBlock(fn, 8);
  EXPECT_EQ(entry, fn->entry().get())
      << "The entry node is not the expected one";

  // Check dominator tree
  {
    DominatorAnalysis dom_tree;
    const CFG& cfg = *context->cfg();
    dom_tree.InitializeTree(cfg, fn);

    // Inspect the actual tree
    DominatorTree& tree = dom_tree.GetDomTree();
    EXPECT_EQ(tree.GetRoot()->bb_, cfg.pseudo_entry_block());
    EXPECT_TRUE(
        dom_tree.Dominates(cfg.pseudo_entry_block()->id(), entry->id()));

    // (strict) dominance checks
    for (uint32_t id : {8, 9}) check_dominance(dom_tree, fn, id, id);

    check_dominance(dom_tree, fn, 8, 9);

    check_no_dominance(dom_tree, fn, 10, 8);
    check_no_dominance(dom_tree, fn, 10, 9);

    EXPECT_EQ(dom_tree.ImmediateDominator(cfg.pseudo_entry_block()), nullptr);
    EXPECT_EQ(dom_tree.ImmediateDominator(entry), cfg.pseudo_entry_block());

    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 9)),
              spvtest::GetBasicBlock(fn, 8));
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 10)),
              nullptr);
  }

  // Check post dominator tree
  {
    PostDominatorAnalysis dom_tree;
    const CFG& cfg = *context->cfg();
    dom_tree.InitializeTree(cfg, fn);

    // Inspect the actual tree
    DominatorTree& tree = dom_tree.GetDomTree();
    EXPECT_EQ(tree.GetRoot()->bb_, cfg.pseudo_exit_block());
    EXPECT_TRUE(dom_tree.Dominates(cfg.pseudo_exit_block()->id(), 9));

    // (strict) dominance checks
    for (uint32_t id : {8, 9, 10}) check_dominance(dom_tree, fn, id, id);

    check_dominance(dom_tree, fn, 9, 8);
    check_dominance(dom_tree, fn, 9, 10);

    EXPECT_EQ(dom_tree.ImmediateDominator(entry),
              spvtest::GetBasicBlock(fn, 9));

    EXPECT_EQ(dom_tree.ImmediateDominator(cfg.pseudo_exit_block()), nullptr);
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 9)),
              cfg.pseudo_exit_block());
    EXPECT_EQ(dom_tree.ImmediateDominator(spvtest::GetBasicBlock(fn, 10)),
              spvtest::GetBasicBlock(fn, 9));
  }
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
