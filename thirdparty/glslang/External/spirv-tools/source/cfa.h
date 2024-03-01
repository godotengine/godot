// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#ifndef SOURCE_CFA_H_
#define SOURCE_CFA_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace spvtools {

// Control Flow Analysis of control flow graphs of basic block nodes |BB|.
template <class BB>
class CFA {
  using bb_ptr = BB*;
  using cbb_ptr = const BB*;
  using bb_iter = typename std::vector<BB*>::const_iterator;
  using get_blocks_func = std::function<const std::vector<BB*>*(const BB*)>;

  struct block_info {
    cbb_ptr block;  ///< pointer to the block
    bb_iter iter;   ///< Iterator to the current child node being processed
  };

  /// Returns true if a block with @p id is found in the @p work_list vector
  ///
  /// @param[in] work_list  Set of blocks visited in the depth first
  /// traversal
  ///                       of the CFG
  /// @param[in] id         The ID of the block being checked
  ///
  /// @return true if the edge work_list.back().block->id() => id is a back-edge
  static bool FindInWorkList(const std::vector<block_info>& work_list,
                             uint32_t id);

 public:
  /// @brief Depth first traversal starting from the \p entry BasicBlock
  ///
  /// This function performs a depth first traversal from the \p entry
  /// BasicBlock and calls the pre/postorder functions when it needs to process
  /// the node in pre order, post order.
  ///
  /// @param[in] entry      The root BasicBlock of a CFG
  /// @param[in] successor_func  A function which will return a pointer to the
  ///                            successor nodes
  /// @param[in] preorder   A function that will be called for every block in a
  ///                       CFG following preorder traversal semantics
  /// @param[in] postorder  A function that will be called for every block in a
  ///                       CFG following postorder traversal semantics
  /// @param[in] terminal   A function that will be called to determine if the
  ///                       search should stop at the given node.
  /// NOTE: The @p successor_func and predecessor_func each return a pointer to
  /// a collection such that iterators to that collection remain valid for the
  /// lifetime of the algorithm.
  static void DepthFirstTraversal(const BB* entry,
                                  get_blocks_func successor_func,
                                  std::function<void(cbb_ptr)> preorder,
                                  std::function<void(cbb_ptr)> postorder,
                                  std::function<bool(cbb_ptr)> terminal);

  /// @brief Depth first traversal starting from the \p entry BasicBlock
  ///
  /// This function performs a depth first traversal from the \p entry
  /// BasicBlock and calls the pre/postorder functions when it needs to process
  /// the node in pre order, post order. It also calls the backedge function
  /// when a back edge is encountered. The backedge function can be empty.  The
  /// runtime of the algorithm is improved if backedge is empty.
  ///
  /// @param[in] entry      The root BasicBlock of a CFG
  /// @param[in] successor_func  A function which will return a pointer to the
  ///                            successor nodes
  /// @param[in] preorder   A function that will be called for every block in a
  ///                       CFG following preorder traversal semantics
  /// @param[in] postorder  A function that will be called for every block in a
  ///                       CFG following postorder traversal semantics
  /// @param[in] backedge   A function that will be called when a backedge is
  ///                       encountered during a traversal.
  /// @param[in] terminal   A function that will be called to determine if the
  ///                       search should stop at the given node.
  /// NOTE: The @p successor_func and predecessor_func each return a pointer to
  /// a collection such that iterators to that collection remain valid for the
  /// lifetime of the algorithm.
  static void DepthFirstTraversal(
      const BB* entry, get_blocks_func successor_func,
      std::function<void(cbb_ptr)> preorder,
      std::function<void(cbb_ptr)> postorder,
      std::function<void(cbb_ptr, cbb_ptr)> backedge,
      std::function<bool(cbb_ptr)> terminal);

  /// @brief Calculates dominator edges for a set of blocks
  ///
  /// Computes dominators using the algorithm of Cooper, Harvey, and Kennedy
  /// "A Simple, Fast Dominance Algorithm", 2001.
  ///
  /// The algorithm assumes there is a unique root node (a node without
  /// predecessors), and it is therefore at the end of the postorder vector.
  ///
  /// This function calculates the dominator edges for a set of blocks in the
  /// CFG.
  /// Uses the dominator algorithm by Cooper et al.
  ///
  /// @param[in] postorder        A vector of blocks in post order traversal
  /// order
  ///                             in a CFG
  /// @param[in] predecessor_func Function used to get the predecessor nodes of
  /// a
  ///                             block
  ///
  /// @return the dominator tree of the graph, as a vector of pairs of nodes.
  /// The first node in the pair is a node in the graph. The second node in the
  /// pair is its immediate dominator in the sense of Cooper et.al., where a
  /// block
  /// without predecessors (such as the root node) is its own immediate
  /// dominator.
  static std::vector<std::pair<BB*, BB*>> CalculateDominators(
      const std::vector<cbb_ptr>& postorder, get_blocks_func predecessor_func);

  // Computes a minimal set of root nodes required to traverse, in the forward
  // direction, the CFG represented by the given vector of blocks, and successor
  // and predecessor functions.  When considering adding two nodes, each having
  // predecessors, favour using the one that appears earlier on the input blocks
  // list.
  static std::vector<BB*> TraversalRoots(const std::vector<BB*>& blocks,
                                         get_blocks_func succ_func,
                                         get_blocks_func pred_func);

  static void ComputeAugmentedCFG(
      std::vector<BB*>& ordered_blocks, BB* pseudo_entry_block,
      BB* pseudo_exit_block,
      std::unordered_map<const BB*, std::vector<BB*>>* augmented_successors_map,
      std::unordered_map<const BB*, std::vector<BB*>>*
          augmented_predecessors_map,
      get_blocks_func succ_func, get_blocks_func pred_func);
};

template <class BB>
bool CFA<BB>::FindInWorkList(const std::vector<block_info>& work_list,
                             uint32_t id) {
  for (const auto& b : work_list) {
    if (b.block->id() == id) return true;
  }
  return false;
}

template <class BB>
void CFA<BB>::DepthFirstTraversal(const BB* entry,
                                  get_blocks_func successor_func,
                                  std::function<void(cbb_ptr)> preorder,
                                  std::function<void(cbb_ptr)> postorder,
                                  std::function<bool(cbb_ptr)> terminal) {
  DepthFirstTraversal(entry, successor_func, preorder, postorder,
                      /* backedge = */ {}, terminal);
}

template <class BB>
void CFA<BB>::DepthFirstTraversal(
    const BB* entry, get_blocks_func successor_func,
    std::function<void(cbb_ptr)> preorder,
    std::function<void(cbb_ptr)> postorder,
    std::function<void(cbb_ptr, cbb_ptr)> backedge,
    std::function<bool(cbb_ptr)> terminal) {
  assert(successor_func && "The successor function cannot be empty.");
  assert(preorder && "The preorder function cannot be empty.");
  assert(postorder && "The postorder function cannot be empty.");
  assert(terminal && "The terminal function cannot be empty.");

  std::unordered_set<uint32_t> processed;

  /// NOTE: work_list is the sequence of nodes from the root node to the node
  /// being processed in the traversal
  std::vector<block_info> work_list;
  work_list.reserve(10);

  work_list.push_back({entry, std::begin(*successor_func(entry))});
  preorder(entry);
  processed.insert(entry->id());

  while (!work_list.empty()) {
    block_info& top = work_list.back();
    if (terminal(top.block) || top.iter == end(*successor_func(top.block))) {
      postorder(top.block);
      work_list.pop_back();
    } else {
      BB* child = *top.iter;
      top.iter++;
      if (backedge && FindInWorkList(work_list, child->id())) {
        backedge(top.block, child);
      }
      if (processed.count(child->id()) == 0) {
        preorder(child);
        work_list.emplace_back(
            block_info{child, std::begin(*successor_func(child))});
        processed.insert(child->id());
      }
    }
  }
}

template <class BB>
std::vector<std::pair<BB*, BB*>> CFA<BB>::CalculateDominators(
    const std::vector<cbb_ptr>& postorder, get_blocks_func predecessor_func) {
  struct block_detail {
    size_t dominator;  ///< The index of blocks's dominator in post order array
    size_t postorder_index;  ///< The index of the block in the post order array
  };
  const size_t undefined_dom = postorder.size();

  std::unordered_map<cbb_ptr, block_detail> idoms;
  for (size_t i = 0; i < postorder.size(); i++) {
    idoms[postorder[i]] = {undefined_dom, i};
  }
  idoms[postorder.back()].dominator = idoms[postorder.back()].postorder_index;

  bool changed = true;
  while (changed) {
    changed = false;
    for (auto b = postorder.rbegin() + 1; b != postorder.rend(); ++b) {
      const std::vector<BB*>& predecessors = *predecessor_func(*b);
      // Find the first processed/reachable predecessor that is reachable
      // in the forward traversal.
      auto res = std::find_if(std::begin(predecessors), std::end(predecessors),
                              [&idoms, undefined_dom](BB* pred) {
                                return idoms.count(pred) &&
                                       idoms[pred].dominator != undefined_dom;
                              });
      if (res == end(predecessors)) continue;
      const BB* idom = *res;
      size_t idom_idx = idoms[idom].postorder_index;

      // all other predecessors
      for (const auto* p : predecessors) {
        if (idom == p) continue;
        // Only consider nodes reachable in the forward traversal.
        // Otherwise the intersection doesn't make sense and will never
        // terminate.
        if (!idoms.count(p)) continue;
        if (idoms[p].dominator != undefined_dom) {
          size_t finger1 = idoms[p].postorder_index;
          size_t finger2 = idom_idx;
          while (finger1 != finger2) {
            while (finger1 < finger2) {
              finger1 = idoms[postorder[finger1]].dominator;
            }
            while (finger2 < finger1) {
              finger2 = idoms[postorder[finger2]].dominator;
            }
          }
          idom_idx = finger1;
        }
      }
      if (idoms[*b].dominator != idom_idx) {
        idoms[*b].dominator = idom_idx;
        changed = true;
      }
    }
  }

  std::vector<std::pair<bb_ptr, bb_ptr>> out;
  for (auto idom : idoms) {
    // At this point if there is no dominator for the node, just make it
    // reflexive.
    auto dominator = std::get<1>(idom).dominator;
    if (dominator == undefined_dom) {
      dominator = std::get<1>(idom).postorder_index;
    }
    // NOTE: performing a const cast for convenient usage with
    // UpdateImmediateDominators
    out.push_back({const_cast<BB*>(std::get<0>(idom)),
                   const_cast<BB*>(postorder[dominator])});
  }

  // Sort by postorder index to generate a deterministic ordering of edges.
  std::sort(
      out.begin(), out.end(),
      [&idoms](const std::pair<bb_ptr, bb_ptr>& lhs,
               const std::pair<bb_ptr, bb_ptr>& rhs) {
        assert(lhs.first);
        assert(lhs.second);
        assert(rhs.first);
        assert(rhs.second);
        auto lhs_indices = std::make_pair(idoms[lhs.first].postorder_index,
                                          idoms[lhs.second].postorder_index);
        auto rhs_indices = std::make_pair(idoms[rhs.first].postorder_index,
                                          idoms[rhs.second].postorder_index);
        return lhs_indices < rhs_indices;
      });
  return out;
}

template <class BB>
std::vector<BB*> CFA<BB>::TraversalRoots(const std::vector<BB*>& blocks,
                                         get_blocks_func succ_func,
                                         get_blocks_func pred_func) {
  // The set of nodes which have been visited from any of the roots so far.
  std::unordered_set<const BB*> visited;

  auto mark_visited = [&visited](const BB* b) { visited.insert(b); };
  auto ignore_block = [](const BB*) {};
  auto no_terminal_blocks = [](const BB*) { return false; };

  auto traverse_from_root = [&mark_visited, &succ_func, &ignore_block,
                             &no_terminal_blocks](const BB* entry) {
    DepthFirstTraversal(entry, succ_func, mark_visited, ignore_block,
                        no_terminal_blocks);
  };

  std::vector<BB*> result;

  // First collect nodes without predecessors.
  for (auto block : blocks) {
    if (pred_func(block)->empty()) {
      assert(visited.count(block) == 0 && "Malformed graph!");
      result.push_back(block);
      traverse_from_root(block);
    }
  }

  // Now collect other stranded nodes.  These must be in unreachable cycles.
  for (auto block : blocks) {
    if (visited.count(block) == 0) {
      result.push_back(block);
      traverse_from_root(block);
    }
  }

  return result;
}

template <class BB>
void CFA<BB>::ComputeAugmentedCFG(
    std::vector<BB*>& ordered_blocks, BB* pseudo_entry_block,
    BB* pseudo_exit_block,
    std::unordered_map<const BB*, std::vector<BB*>>* augmented_successors_map,
    std::unordered_map<const BB*, std::vector<BB*>>* augmented_predecessors_map,
    get_blocks_func succ_func, get_blocks_func pred_func) {
  // Compute the successors of the pseudo-entry block, and
  // the predecessors of the pseudo exit block.
  auto sources = TraversalRoots(ordered_blocks, succ_func, pred_func);

  // For the predecessor traversals, reverse the order of blocks.  This
  // will affect the post-dominance calculation as follows:
  //  - Suppose you have blocks A and B, with A appearing before B in
  //    the list of blocks.
  //  - Also, A branches only to B, and B branches only to A.
  //  - We want to compute A as dominating B, and B as post-dominating B.
  // By using reversed blocks for predecessor traversal roots discovery,
  // we'll add an edge from B to the pseudo-exit node, rather than from A.
  // All this is needed to correctly process the dominance/post-dominance
  // constraint when A is a loop header that points to itself as its
  // own continue target, and B is the latch block for the loop.
  std::vector<BB*> reversed_blocks(ordered_blocks.rbegin(),
                                   ordered_blocks.rend());
  auto sinks = TraversalRoots(reversed_blocks, pred_func, succ_func);

  // Wire up the pseudo entry block.
  (*augmented_successors_map)[pseudo_entry_block] = sources;
  for (auto block : sources) {
    auto& augmented_preds = (*augmented_predecessors_map)[block];
    const auto preds = pred_func(block);
    augmented_preds.reserve(1 + preds->size());
    augmented_preds.push_back(pseudo_entry_block);
    augmented_preds.insert(augmented_preds.end(), preds->begin(), preds->end());
  }

  // Wire up the pseudo exit block.
  (*augmented_predecessors_map)[pseudo_exit_block] = sinks;
  for (auto block : sinks) {
    auto& augmented_succ = (*augmented_successors_map)[block];
    const auto succ = succ_func(block);
    augmented_succ.reserve(1 + succ->size());
    augmented_succ.push_back(pseudo_exit_block);
    augmented_succ.insert(augmented_succ.end(), succ->begin(), succ->end());
  }
}

}  // namespace spvtools

#endif  // SOURCE_CFA_H_
