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

#ifndef SOURCE_OPT_CFG_H_
#define SOURCE_OPT_CFG_H_

#include <algorithm>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/opt/basic_block.h"

namespace spvtools {
namespace opt {

class CFG {
 public:
  explicit CFG(Module* module);

  // Return the list of predecessors for basic block with label |blkid|.
  // TODO(dnovillo): Move this to BasicBlock.
  const std::vector<uint32_t>& preds(uint32_t blk_id) const {
    assert(label2preds_.count(blk_id));
    return label2preds_.at(blk_id);
  }

  // Return a pointer to the basic block instance corresponding to the label
  // |blk_id|.
  BasicBlock* block(uint32_t blk_id) const { return id2block_.at(blk_id); }

  // Return the pseudo entry and exit blocks.
  const BasicBlock* pseudo_entry_block() const { return &pseudo_entry_block_; }
  BasicBlock* pseudo_entry_block() { return &pseudo_entry_block_; }

  const BasicBlock* pseudo_exit_block() const { return &pseudo_exit_block_; }
  BasicBlock* pseudo_exit_block() { return &pseudo_exit_block_; }

  // Return true if |block_ptr| is the pseudo-entry block.
  bool IsPseudoEntryBlock(BasicBlock* block_ptr) const {
    return block_ptr == &pseudo_entry_block_;
  }

  // Return true if |block_ptr| is the pseudo-exit block.
  bool IsPseudoExitBlock(BasicBlock* block_ptr) const {
    return block_ptr == &pseudo_exit_block_;
  }

  // Compute structured block order into |order| for |func| starting at |root|.
  // This order has the property that dominators come before all blocks they
  // dominate, merge blocks come after all blocks that are in the control
  // constructs of their header, and continue blocks come after all of the
  // blocks in the body of their loop.
  void ComputeStructuredOrder(Function* func, BasicBlock* root,
                              std::list<BasicBlock*>* order);

  // Compute structured block order into |order| for |func| starting at |root|
  // and ending at |end|. This order has the property that dominators come
  // before all blocks they dominate, merge blocks come after all blocks that
  // are in the control constructs of their header, and continue blocks come
  // after all the blocks in the body of their loop.
  void ComputeStructuredOrder(Function* func, BasicBlock* root, BasicBlock* end,
                              std::list<BasicBlock*>* order);

  // Applies |f| to all blocks that can be reach from |bb| in post order.
  void ForEachBlockInPostOrder(BasicBlock* bb,
                               const std::function<void(BasicBlock*)>& f);

  // Applies |f| to all blocks that can be reach from |bb| in reverse post
  // order.
  void ForEachBlockInReversePostOrder(
      BasicBlock* bb, const std::function<void(BasicBlock*)>& f);

  // Applies |f| to all blocks that can be reach from |bb| in reverse post
  // order.  Return false if |f| return false on any basic block, and stops
  // processing.
  bool WhileEachBlockInReversePostOrder(
      BasicBlock* bb, const std::function<bool(BasicBlock*)>& f);

  // Registers |blk| as a basic block in the cfg, this also updates the
  // predecessor lists of each successor of |blk|. |blk| must have a terminator
  // instruction at the end of the block.
  void RegisterBlock(BasicBlock* blk) {
    assert(blk->begin() != blk->end() &&
           "Basic blocks must have a terminator before registering.");
    assert(blk->tail()->IsBlockTerminator() &&
           "Basic blocks must have a terminator before registering.");
    uint32_t blk_id = blk->id();
    id2block_[blk_id] = blk;
    AddEdges(blk);
  }

  // Removes from the CFG any mapping for the basic block id |blk_id|.
  void ForgetBlock(const BasicBlock* blk) {
    id2block_.erase(blk->id());
    label2preds_.erase(blk->id());
    RemoveSuccessorEdges(blk);
  }

  void RemoveEdge(uint32_t pred_blk_id, uint32_t succ_blk_id) {
    auto pred_it = label2preds_.find(succ_blk_id);
    if (pred_it == label2preds_.end()) return;
    auto& preds_list = pred_it->second;
    auto it = std::find(preds_list.begin(), preds_list.end(), pred_blk_id);
    if (it != preds_list.end()) preds_list.erase(it);
  }

  // Registers |blk| to all of its successors.
  void AddEdges(BasicBlock* blk);

  // Registers the basic block id |pred_blk_id| as being a predecessor of the
  // basic block id |succ_blk_id|.
  void AddEdge(uint32_t pred_blk_id, uint32_t succ_blk_id) {
    label2preds_[succ_blk_id].push_back(pred_blk_id);
  }

  // Removes any edges that no longer exist from the predecessor mapping for
  // the basic block id |blk_id|.
  void RemoveNonExistingEdges(uint32_t blk_id);

  // Remove all edges that leave |bb|.
  void RemoveSuccessorEdges(const BasicBlock* bb) {
    bb->ForEachSuccessorLabel(
        [bb, this](uint32_t succ_id) { RemoveEdge(bb->id(), succ_id); });
  }

  // Divides |block| into two basic blocks.  The first block will have the same
  // id as |block| and will become a preheader for the loop.  The other block
  // is a new block that will be the new loop header.
  //
  // Returns a pointer to the new loop header.  Returns |nullptr| if the new
  // loop pointer could not be created.
  BasicBlock* SplitLoopHeader(BasicBlock* bb);

 private:
  // Compute structured successors for function |func|. A block's structured
  // successors are the blocks it branches to together with its declared merge
  // block and continue block if it has them. When order matters, the merge
  // block and continue block always appear first. This assures correct depth
  // first search in the presence of early returns and kills. If the successor
  // vector contain duplicates of the merge or continue blocks, they are safely
  // ignored by DFS.
  void ComputeStructuredSuccessors(Function* func);

  // Computes the post-order traversal of the cfg starting at |bb| skipping
  // nodes in |seen|.  The order of the traversal is appended to |order|, and
  // all nodes in the traversal are added to |seen|.
  void ComputePostOrderTraversal(BasicBlock* bb,
                                 std::vector<BasicBlock*>* order,
                                 std::unordered_set<BasicBlock*>* seen);

  // Module for this CFG.
  Module* module_;

  // Map from block to its structured successor blocks. See
  // ComputeStructuredSuccessors() for definition.
  std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>>
      block2structured_succs_;

  // Extra block whose successors are all blocks with no predecessors
  // in function.
  BasicBlock pseudo_entry_block_;

  // Augmented CFG Exit Block.
  BasicBlock pseudo_exit_block_;

  // Map from block's label id to its predecessor blocks ids
  std::unordered_map<uint32_t, std::vector<uint32_t>> label2preds_;

  // Map from block's label id to block.
  std::unordered_map<uint32_t, BasicBlock*> id2block_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_CFG_H_
