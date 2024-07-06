// Copyright (c) 2021 Google LLC.
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

#include "source/opt/control_dependence.h"

#include <cassert>
#include <tuple>

#include "source/opt/basic_block.h"
#include "source/opt/cfg.h"
#include "source/opt/dominator_analysis.h"
#include "source/opt/function.h"
#include "source/opt/instruction.h"

// Computes the control dependence graph (CDG) using the algorithm in Cytron
// 1991, "Efficiently Computing Static Single Assignment Form and the Control
// Dependence Graph." It relies on the fact that the control dependence sources
// (blocks on which a block is control dependent) are exactly the post-dominance
// frontier for that block. The explanation and proofs are given in Section 6 of
// that paper.
// Link: https://www.cs.utexas.edu/~pingali/CS380C/2010/papers/ssaCytron.pdf
//
// The algorithm in Section 4.2 of the same paper is used to construct the
// dominance frontier. It uses the post-dominance tree, which is available in
// the IR context.

namespace spvtools {
namespace opt {
constexpr uint32_t ControlDependenceAnalysis::kPseudoEntryBlock;

uint32_t ControlDependence::GetConditionID(const CFG& cfg) const {
  if (source_bb_id() == 0) {
    // Entry dependence; return 0.
    return 0;
  }
  const BasicBlock* source_bb = cfg.block(source_bb_id());
  const Instruction* branch = source_bb->terminator();
  assert((branch->opcode() == spv::Op::OpBranchConditional ||
          branch->opcode() == spv::Op::OpSwitch) &&
         "invalid control dependence; last instruction must be conditional "
         "branch or switch");
  return branch->GetSingleWordInOperand(0);
}

bool ControlDependence::operator<(const ControlDependence& other) const {
  return std::tie(source_bb_id_, target_bb_id_, branch_target_bb_id_) <
         std::tie(other.source_bb_id_, other.target_bb_id_,
                  other.branch_target_bb_id_);
}

bool ControlDependence::operator==(const ControlDependence& other) const {
  return std::tie(source_bb_id_, target_bb_id_, branch_target_bb_id_) ==
         std::tie(other.source_bb_id_, other.target_bb_id_,
                  other.branch_target_bb_id_);
}

std::ostream& operator<<(std::ostream& os, const ControlDependence& dep) {
  os << dep.source_bb_id() << "->" << dep.target_bb_id();
  if (dep.branch_target_bb_id() != dep.target_bb_id()) {
    os << " through " << dep.branch_target_bb_id();
  }
  return os;
}

void ControlDependenceAnalysis::ComputePostDominanceFrontiers(
    const CFG& cfg, const PostDominatorAnalysis& pdom) {
  // Compute post-dominance frontiers (reverse graph).
  // The dominance frontier for a block X is equal to (Equation 4)
  //   DF_local(X) U { B in DF_up(Z) | X = ipdom(Z) }
  //   (ipdom(Z) is the immediate post-dominator of Z.)
  // where
  //   DF_local(X) = { Y | X -> Y in CFG, X does not strictly post-dominate Y }
  //     represents the contribution of X's predecessors to the DF, and
  //   DF_up(Z) = { Y | Y in DF(Z), ipdom(Z) does not strictly post-dominate Y }
  //     (note: ipdom(Z) = X.)
  //     represents the contribution of a block to its immediate post-
  //     dominator's DF.
  // This is computed in one pass through a post-order traversal of the
  // post-dominator tree.

  // Assert that there is a block other than the pseudo exit in the pdom tree,
  // as we need one to get the function entry point (as the pseudo exit is not
  // actually part of the function.)
  assert(!cfg.IsPseudoExitBlock(pdom.GetDomTree().post_begin()->bb_));
  Function* function = pdom.GetDomTree().post_begin()->bb_->GetParent();
  uint32_t function_entry = function->entry()->id();
  // Explicitly initialize pseudo-entry block, as it doesn't depend on anything,
  // so it won't be initialized in the following loop.
  reverse_nodes_[kPseudoEntryBlock] = {};
  for (auto it = pdom.GetDomTree().post_cbegin();
       it != pdom.GetDomTree().post_cend(); ++it) {
    ComputePostDominanceFrontierForNode(cfg, pdom, function_entry, *it);
  }
}

void ControlDependenceAnalysis::ComputePostDominanceFrontierForNode(
    const CFG& cfg, const PostDominatorAnalysis& pdom, uint32_t function_entry,
    const DominatorTreeNode& pdom_node) {
  const uint32_t label = pdom_node.id();
  ControlDependenceList& edges = reverse_nodes_[label];
  for (uint32_t pred : cfg.preds(label)) {
    if (!pdom.StrictlyDominates(label, pred)) {
      edges.push_back(ControlDependence(pred, label));
    }
  }
  if (label == function_entry) {
    // Add edge from pseudo-entry to entry.
    // In CDG construction, an edge is added from entry to exit, so only the
    // exit node can post-dominate entry.
    edges.push_back(ControlDependence(kPseudoEntryBlock, label));
  }
  for (DominatorTreeNode* child : pdom_node) {
    // Note: iterate dependences by value, as we need a copy.
    for (const ControlDependence& dep : reverse_nodes_[child->id()]) {
      // Special-case pseudo-entry, as above.
      if (dep.source_bb_id() == kPseudoEntryBlock ||
          !pdom.StrictlyDominates(label, dep.source_bb_id())) {
        edges.push_back(ControlDependence(dep.source_bb_id(), label,
                                          dep.branch_target_bb_id()));
      }
    }
  }
}

void ControlDependenceAnalysis::ComputeControlDependenceGraph(
    const CFG& cfg, const PostDominatorAnalysis& pdom) {
  ComputePostDominanceFrontiers(cfg, pdom);
  ComputeForwardGraphFromReverse();
}

void ControlDependenceAnalysis::ComputeForwardGraphFromReverse() {
  for (const auto& entry : reverse_nodes_) {
    // Ensure an entry is created for each node.
    forward_nodes_[entry.first];
    for (const ControlDependence& dep : entry.second) {
      forward_nodes_[dep.source_bb_id()].push_back(dep);
    }
  }
}

}  // namespace opt
}  // namespace spvtools
