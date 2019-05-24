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

#include <iostream>
#include <memory>
#include <set>

#include "source/cfa.h"
#include "source/opt/dominator_tree.h"
#include "source/opt/ir_context.h"

// Calculates the dominator or postdominator tree for a given function.
// 1 - Compute the successors and predecessors for each BasicBlock. We add a
// dummy node for the start node or for postdominators the exit. This node will
// point to all entry or all exit nodes.
// 2 - Using the CFA::DepthFirstTraversal get a depth first postordered list of
// all BasicBlocks. Using the successors (or for postdominator, predecessors)
// calculated in step 1 to traverse the tree.
// 3 - Pass the list calculated in step 2 to the CFA::CalculateDominators using
// the predecessors list (or for postdominator, successors). This will give us a
// vector of BB pairs. Each BB and its immediate dominator.
// 4 - Using the list from 3 use those edges to build a tree of
// DominatorTreeNodes. Each node containing a link to the parent dominator and
// children which are dominated.
// 5 - Using the tree from 4, perform a depth first traversal to calculate the
// preorder and postorder index of each node. We use these indexes to compare
// nodes against each other for domination checks.

namespace spvtools {
namespace opt {
namespace {

// Wrapper around CFA::DepthFirstTraversal to provide an interface to perform
// depth first search on generic BasicBlock types. Will call post and pre order
// user defined functions during traversal
//
// BBType - BasicBlock type. Will either be BasicBlock or DominatorTreeNode
// SuccessorLambda - Lamdba matching the signature of 'const
// std::vector<BBType>*(const BBType *A)'. Will return a vector of the nodes
// succeding BasicBlock A.
// PostLambda - Lamdba matching the signature of 'void (const BBType*)' will be
// called on each node traversed AFTER their children.
// PreLambda - Lamdba matching the signature of 'void (const BBType*)' will be
// called on each node traversed BEFORE their children.
template <typename BBType, typename SuccessorLambda, typename PreLambda,
          typename PostLambda>
static void DepthFirstSearch(const BBType* bb, SuccessorLambda successors,
                             PreLambda pre, PostLambda post) {
  // Ignore backedge operation.
  auto nop_backedge = [](const BBType*, const BBType*) {};
  CFA<BBType>::DepthFirstTraversal(bb, successors, pre, post, nop_backedge);
}

// Wrapper around CFA::DepthFirstTraversal to provide an interface to perform
// depth first search on generic BasicBlock types. This overload is for only
// performing user defined post order.
//
// BBType - BasicBlock type. Will either be BasicBlock or DominatorTreeNode
// SuccessorLambda - Lamdba matching the signature of 'const
// std::vector<BBType>*(const BBType *A)'. Will return a vector of the nodes
// succeding BasicBlock A.
// PostLambda - Lamdba matching the signature of 'void (const BBType*)' will be
// called on each node traversed after their children.
template <typename BBType, typename SuccessorLambda, typename PostLambda>
static void DepthFirstSearchPostOrder(const BBType* bb,
                                      SuccessorLambda successors,
                                      PostLambda post) {
  // Ignore preorder operation.
  auto nop_preorder = [](const BBType*) {};
  DepthFirstSearch(bb, successors, nop_preorder, post);
}

// Small type trait to get the function class type.
template <typename BBType>
struct GetFunctionClass {
  using FunctionType = Function;
};

// Helper class to compute predecessors and successors for each Basic Block in a
// function. Through GetPredFunctor and GetSuccessorFunctor it provides an
// interface to get the successor and predecessor lists for each basic
// block. This is required by the DepthFirstTraversal and ComputeDominator
// functions which take as parameter an std::function returning the successors
// and predecessors respectively.
//
// When computing the post-dominator tree, all edges are inverted. So successors
// returned by this class will be predecessors in the original CFG.
template <typename BBType>
class BasicBlockSuccessorHelper {
  // This should eventually become const BasicBlock.
  using BasicBlock = BBType;
  using Function = typename GetFunctionClass<BBType>::FunctionType;

  using BasicBlockListTy = std::vector<BasicBlock*>;
  using BasicBlockMapTy = std::map<const BasicBlock*, BasicBlockListTy>;

 public:
  // For compliance with the dominance tree computation, entry nodes are
  // connected to a single dummy node.
  BasicBlockSuccessorHelper(Function& func, const BasicBlock* dummy_start_node,
                            bool post);

  // CFA::CalculateDominators requires std::vector<BasicBlock*>.
  using GetBlocksFunction =
      std::function<const std::vector<BasicBlock*>*(const BasicBlock*)>;

  // Returns the list of predecessor functions.
  GetBlocksFunction GetPredFunctor() {
    return [this](const BasicBlock* bb) {
      BasicBlockListTy* v = &this->predecessors_[bb];
      return v;
    };
  }

  // Returns a vector of the list of successor nodes from a given node.
  GetBlocksFunction GetSuccessorFunctor() {
    return [this](const BasicBlock* bb) {
      BasicBlockListTy* v = &this->successors_[bb];
      return v;
    };
  }

 private:
  bool invert_graph_;
  BasicBlockMapTy successors_;
  BasicBlockMapTy predecessors_;

  // Build the successors and predecessors map for each basic blocks |f|.
  // If |invert_graph_| is true, all edges are reversed (successors becomes
  // predecessors and vice versa).
  // For convenience, the start of the graph is |dummy_start_node|.
  // The dominator tree construction requires a unique entry node, which cannot
  // be guaranteed for the postdominator graph. The |dummy_start_node| BB is
  // here to gather all entry nodes.
  void CreateSuccessorMap(Function& f, const BasicBlock* dummy_start_node);
};

template <typename BBType>
BasicBlockSuccessorHelper<BBType>::BasicBlockSuccessorHelper(
    Function& func, const BasicBlock* dummy_start_node, bool invert)
    : invert_graph_(invert) {
  CreateSuccessorMap(func, dummy_start_node);
}

template <typename BBType>
void BasicBlockSuccessorHelper<BBType>::CreateSuccessorMap(
    Function& f, const BasicBlock* dummy_start_node) {
  std::map<uint32_t, BasicBlock*> id_to_BB_map;
  auto GetSuccessorBasicBlock = [&f, &id_to_BB_map](uint32_t successor_id) {
    BasicBlock*& Succ = id_to_BB_map[successor_id];
    if (!Succ) {
      for (BasicBlock& BBIt : f) {
        if (successor_id == BBIt.id()) {
          Succ = &BBIt;
          break;
        }
      }
    }
    return Succ;
  };

  if (invert_graph_) {
    // For the post dominator tree, we see the inverted graph.
    // successors_ in the inverted graph are the predecessors in the CFG.
    // The tree construction requires 1 entry point, so we add a dummy node
    // that is connected to all function exiting basic blocks.
    // An exiting basic block is a block with an OpKill, OpUnreachable,
    // OpReturn or OpReturnValue as terminator instruction.
    for (BasicBlock& bb : f) {
      if (bb.hasSuccessor()) {
        BasicBlockListTy& pred_list = predecessors_[&bb];
        const auto& const_bb = bb;
        const_bb.ForEachSuccessorLabel(
            [this, &pred_list, &bb,
             &GetSuccessorBasicBlock](const uint32_t successor_id) {
              BasicBlock* succ = GetSuccessorBasicBlock(successor_id);
              // Inverted graph: our successors in the CFG
              // are our predecessors in the inverted graph.
              this->successors_[succ].push_back(&bb);
              pred_list.push_back(succ);
            });
      } else {
        successors_[dummy_start_node].push_back(&bb);
        predecessors_[&bb].push_back(const_cast<BasicBlock*>(dummy_start_node));
      }
    }
  } else {
    successors_[dummy_start_node].push_back(f.entry().get());
    predecessors_[f.entry().get()].push_back(
        const_cast<BasicBlock*>(dummy_start_node));
    for (BasicBlock& bb : f) {
      BasicBlockListTy& succ_list = successors_[&bb];

      const auto& const_bb = bb;
      const_bb.ForEachSuccessorLabel([&](const uint32_t successor_id) {
        BasicBlock* succ = GetSuccessorBasicBlock(successor_id);
        succ_list.push_back(succ);
        predecessors_[succ].push_back(&bb);
      });
    }
  }
}

}  // namespace

bool DominatorTree::StrictlyDominates(uint32_t a, uint32_t b) const {
  if (a == b) return false;
  return Dominates(a, b);
}

bool DominatorTree::StrictlyDominates(const BasicBlock* a,
                                      const BasicBlock* b) const {
  return DominatorTree::StrictlyDominates(a->id(), b->id());
}

bool DominatorTree::StrictlyDominates(const DominatorTreeNode* a,
                                      const DominatorTreeNode* b) const {
  if (a == b) return false;
  return Dominates(a, b);
}

bool DominatorTree::Dominates(uint32_t a, uint32_t b) const {
  // Check that both of the inputs are actual nodes.
  const DominatorTreeNode* a_node = GetTreeNode(a);
  const DominatorTreeNode* b_node = GetTreeNode(b);
  if (!a_node || !b_node) return false;

  return Dominates(a_node, b_node);
}

bool DominatorTree::Dominates(const DominatorTreeNode* a,
                              const DominatorTreeNode* b) const {
  // Node A dominates node B if they are the same.
  if (a == b) return true;

  return a->dfs_num_pre_ < b->dfs_num_pre_ &&
         a->dfs_num_post_ > b->dfs_num_post_;
}

bool DominatorTree::Dominates(const BasicBlock* A, const BasicBlock* B) const {
  return Dominates(A->id(), B->id());
}

BasicBlock* DominatorTree::ImmediateDominator(const BasicBlock* A) const {
  return ImmediateDominator(A->id());
}

BasicBlock* DominatorTree::ImmediateDominator(uint32_t a) const {
  // Check that A is a valid node in the tree.
  auto a_itr = nodes_.find(a);
  if (a_itr == nodes_.end()) return nullptr;

  const DominatorTreeNode* node = &a_itr->second;

  if (node->parent_ == nullptr) {
    return nullptr;
  }

  return node->parent_->bb_;
}

DominatorTreeNode* DominatorTree::GetOrInsertNode(BasicBlock* bb) {
  DominatorTreeNode* dtn = nullptr;

  std::map<uint32_t, DominatorTreeNode>::iterator node_iter =
      nodes_.find(bb->id());
  if (node_iter == nodes_.end()) {
    dtn = &nodes_.emplace(std::make_pair(bb->id(), DominatorTreeNode{bb}))
               .first->second;
  } else {
    dtn = &node_iter->second;
  }

  return dtn;
}

void DominatorTree::GetDominatorEdges(
    const Function* f, const BasicBlock* dummy_start_node,
    std::vector<std::pair<BasicBlock*, BasicBlock*>>* edges) {
  // Each time the depth first traversal calls the postorder callback
  // std::function we push that node into the postorder vector to create our
  // postorder list.
  std::vector<const BasicBlock*> postorder;
  auto postorder_function = [&](const BasicBlock* b) {
    postorder.push_back(b);
  };

  // CFA::CalculateDominators requires std::vector<BasicBlock*>
  // BB are derived from F, so we need to const cast it at some point
  // no modification is made on F.
  BasicBlockSuccessorHelper<BasicBlock> helper{
      *const_cast<Function*>(f), dummy_start_node, postdominator_};

  // The successor function tells DepthFirstTraversal how to move to successive
  // nodes by providing an interface to get a list of successor nodes from any
  // given node.
  auto successor_functor = helper.GetSuccessorFunctor();

  // The predecessor functor does the same as the successor functor
  // but for all nodes preceding a given node.
  auto predecessor_functor = helper.GetPredFunctor();

  // If we're building a post dominator tree we traverse the tree in reverse
  // using the predecessor function in place of the successor function and vice
  // versa.
  DepthFirstSearchPostOrder(dummy_start_node, successor_functor,
                            postorder_function);
  *edges = CFA<BasicBlock>::CalculateDominators(postorder, predecessor_functor);
}

void DominatorTree::InitializeTree(const CFG& cfg, const Function* f) {
  ClearTree();

  // Skip over empty functions.
  if (f->cbegin() == f->cend()) {
    return;
  }

  const BasicBlock* dummy_start_node =
      postdominator_ ? cfg.pseudo_exit_block() : cfg.pseudo_entry_block();

  // Get the immediate dominator for each node.
  std::vector<std::pair<BasicBlock*, BasicBlock*>> edges;
  GetDominatorEdges(f, dummy_start_node, &edges);

  // Transform the vector<pair> into the tree structure which we can use to
  // efficiently query dominance.
  for (auto edge : edges) {
    DominatorTreeNode* first = GetOrInsertNode(edge.first);

    if (edge.first == edge.second) {
      if (std::find(roots_.begin(), roots_.end(), first) == roots_.end())
        roots_.push_back(first);
      continue;
    }

    DominatorTreeNode* second = GetOrInsertNode(edge.second);

    first->parent_ = second;
    second->children_.push_back(first);
  }
  ResetDFNumbering();
}

void DominatorTree::ResetDFNumbering() {
  int index = 0;
  auto preFunc = [&index](const DominatorTreeNode* node) {
    const_cast<DominatorTreeNode*>(node)->dfs_num_pre_ = ++index;
  };

  auto postFunc = [&index](const DominatorTreeNode* node) {
    const_cast<DominatorTreeNode*>(node)->dfs_num_post_ = ++index;
  };

  auto getSucc = [](const DominatorTreeNode* node) { return &node->children_; };

  for (auto root : roots_) DepthFirstSearch(root, getSucc, preFunc, postFunc);
}

void DominatorTree::DumpTreeAsDot(std::ostream& out_stream) const {
  out_stream << "digraph {\n";
  Visit([&out_stream](const DominatorTreeNode* node) {
    // Print the node.
    if (node->bb_) {
      out_stream << node->bb_->id() << "[label=\"" << node->bb_->id()
                 << "\"];\n";
    }

    // Print the arrow from the parent to this node. Entry nodes will not have
    // parents so draw them as children from the dummy node.
    if (node->parent_) {
      out_stream << node->parent_->bb_->id() << " -> " << node->bb_->id()
                 << ";\n";
    }

    // Return true to continue the traversal.
    return true;
  });
  out_stream << "}\n";
}

}  // namespace opt
}  // namespace spvtools
