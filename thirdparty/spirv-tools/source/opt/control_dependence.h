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

#ifndef SOURCE_OPT_CONTROL_DEPENDENCE_H_
#define SOURCE_OPT_CONTROL_DEPENDENCE_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "source/opt/cfg.h"
#include "source/opt/dominator_analysis.h"

namespace spvtools {
namespace opt {

class ControlDependence {
 public:
  // The label of the source of this dependence, i.e. the block on which the
  // target is dependent on.
  // A |source_bb_id| of 0 represents an "entry" dependence, meaning that the
  // execution of |target_bb_id| is only dependent on entry to the function.
  uint32_t source_bb_id() const { return source_bb_id_; }
  // The label of the target of this dependence, i.e. the block which is
  // dependent on the source.
  uint32_t target_bb_id() const { return target_bb_id_; }
  // The label of the target of the *branch* for this dependence.
  // Equal to the ID of the entry block for entry dependences.
  //
  // For example, for the partial CFG pictured below:
  // 1 ---> 2 ---> 4 ---> 6
  //  \      \            ^
  //   \-> 3  \-> 5 -----/
  // Block 6 is control dependent on block 1, but this dependence comes from the
  // branch 1 -> 2, so in this case the branch target ID would be 2.
  uint32_t branch_target_bb_id() const { return branch_target_bb_id_; }

  // Create a direct control dependence from BB ID |source| to |target|.
  ControlDependence(uint32_t source, uint32_t target)
      : source_bb_id_(source),
        target_bb_id_(target),
        branch_target_bb_id_(target) {}
  // Create a control dependence from BB ID |source| to |target| through the
  // branch from |source| to |branch_target|.
  ControlDependence(uint32_t source, uint32_t target, uint32_t branch_target)
      : source_bb_id_(source),
        target_bb_id_(target),
        branch_target_bb_id_(branch_target) {}

  // Gets the ID of the conditional value for the branch corresponding to this
  // control dependence. This is the first input operand for both
  // OpConditionalBranch and OpSwitch.
  // Returns 0 for entry dependences.
  uint32_t GetConditionID(const CFG& cfg) const;

  bool operator==(const ControlDependence& other) const;
  bool operator!=(const ControlDependence& other) const {
    return !(*this == other);
  }

  // Comparison operators, ordered lexicographically. Total ordering.
  bool operator<(const ControlDependence& other) const;
  bool operator>(const ControlDependence& other) const { return other < *this; }
  bool operator<=(const ControlDependence& other) const {
    return !(*this > other);
  }
  bool operator>=(const ControlDependence& other) const {
    return !(*this < other);
  }

 private:
  uint32_t source_bb_id_;
  uint32_t target_bb_id_;
  uint32_t branch_target_bb_id_;
};

// Prints |dep| to |os| in a human-readable way. For example,
//   1->2           (target_bb_id = branch_target_bb_id = 2)
//   3->4 through 5 (target_bb_id = 4, branch_target_bb_id = 5)
std::ostream& operator<<(std::ostream& os, const ControlDependence& dep);

// Represents the control dependence graph. A basic block is control dependent
// on another if the result of that block (e.g. the condition of a conditional
// branch) influences whether it is executed or not. More formally, a block A is
// control dependent on B iff:
// 1. there exists a path from A to the exit node that does *not* go through B
//    (i.e., A does not postdominate B), and
// 2. there exists a path B -> b_1 -> ... -> b_n -> A such that A post-dominates
//    all nodes b_i.
class ControlDependenceAnalysis {
 public:
  // Map basic block labels to control dependencies/dependents.
  // Not guaranteed to be in any particular order.
  using ControlDependenceList = std::vector<ControlDependence>;
  using ControlDependenceListMap =
      std::unordered_map<uint32_t, ControlDependenceList>;

  // 0, the label number for the pseudo entry block.
  // All control dependences on the pseudo entry block are of type kEntry, and
  // vice versa.
  static constexpr uint32_t kPseudoEntryBlock = 0;

  // Build the control dependence graph for the given control flow graph |cfg|
  // and corresponding post-dominator analysis |pdom|.
  void ComputeControlDependenceGraph(const CFG& cfg,
                                     const PostDominatorAnalysis& pdom);

  // Get the list of the nodes that depend on a block.
  // Return value is not guaranteed to be in any particular order.
  const ControlDependenceList& GetDependenceTargets(uint32_t block) const {
    return forward_nodes_.at(block);
  }

  // Get the list of the nodes on which a block depends on.
  // Return value is not guaranteed to be in any particular order.
  const ControlDependenceList& GetDependenceSources(uint32_t block) const {
    return reverse_nodes_.at(block);
  }

  // Runs the function |f| on each block label in the CDG. If any iteration
  // returns false, immediately stops iteration and returns false. Otherwise
  // returns true. Nodes are iterated in some undefined order, including the
  // pseudo-entry block.
  bool WhileEachBlockLabel(std::function<bool(uint32_t)> f) const {
    for (const auto& entry : forward_nodes_) {
      if (!f(entry.first)) {
        return false;
      }
    }
    return true;
  }

  // Runs the function |f| on each block label in the CDG. Nodes are iterated in
  // some undefined order, including the pseudo-entry block.
  void ForEachBlockLabel(std::function<void(uint32_t)> f) const {
    WhileEachBlockLabel([&f](uint32_t label) {
      f(label);
      return true;
    });
  }

  // Returns true if the block |id| exists in the control dependence graph.
  // This can be false even if the block exists in the function when it is part
  // of an infinite loop, since it is not part of the post-dominator tree.
  bool HasBlock(uint32_t id) const { return forward_nodes_.count(id) > 0; }

  // Returns true if block |a| is dependent on block |b|.
  bool IsDependent(uint32_t a, uint32_t b) const {
    if (!HasBlock(a)) return false;
    // BBs tend to have more dependents (targets) than they are dependent on
    // (sources), so search sources.
    const ControlDependenceList& a_sources = GetDependenceSources(a);
    return std::find_if(a_sources.begin(), a_sources.end(),
                        [b](const ControlDependence& dep) {
                          return dep.source_bb_id() == b;
                        }) != a_sources.end();
  }

 private:
  // Computes the post-dominance frontiers (i.e. the reverse CDG) for each node
  // in the post-dominator tree. Only modifies reverse_nodes_; forward_nodes_ is
  // not modified.
  void ComputePostDominanceFrontiers(const CFG& cfg,
                                     const PostDominatorAnalysis& pdom);
  // Computes the post-dominance frontier for a specific node |pdom_node| in the
  // post-dominator tree. Result is placed in reverse_nodes_[pdom_node.id()].
  void ComputePostDominanceFrontierForNode(const CFG& cfg,
                                           const PostDominatorAnalysis& pdom,
                                           uint32_t function_entry,
                                           const DominatorTreeNode& pdom_node);

  // Computes the forward graph (forward_nodes_) from the reverse graph
  // (reverse_nodes_).
  void ComputeForwardGraphFromReverse();

  ControlDependenceListMap forward_nodes_;
  ControlDependenceListMap reverse_nodes_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_CONTROL_DEPENDENCE_H_
