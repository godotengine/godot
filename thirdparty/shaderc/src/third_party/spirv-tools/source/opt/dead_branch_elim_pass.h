// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#ifndef SOURCE_OPT_DEAD_BRANCH_ELIM_PASS_H_
#define SOURCE_OPT_DEAD_BRANCH_ELIM_PASS_H_

#include <algorithm>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/mem_pass.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class DeadBranchElimPass : public MemPass {
  using cbb_ptr = const BasicBlock*;

 public:
  DeadBranchElimPass() = default;

  const char* name() const override { return "eliminate-dead-branches"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // If |condId| is boolean constant, return conditional value in |condVal| and
  // return true, otherwise return false.
  bool GetConstCondition(uint32_t condId, bool* condVal);

  // If |valId| is a 32-bit integer constant, return value via |value| and
  // return true, otherwise return false.
  bool GetConstInteger(uint32_t valId, uint32_t* value);

  // Add branch to |labelId| to end of block |bp|.
  void AddBranch(uint32_t labelId, BasicBlock* bp);

  // For function |func|, look for BranchConditionals with constant condition
  // and convert to a Branch to the indicated label. Delete resulting dead
  // blocks. Note some such branches and blocks may be left to avoid creating
  // invalid control flow.
  // TODO(greg-lunarg): Remove remaining constant conditional branches and dead
  // blocks.
  bool EliminateDeadBranches(Function* func);

  // Returns the basic block containing |id|.
  // Note: this pass only requires correct instruction block mappings for the
  // input. This pass does not preserve the block mapping, so it is not kept
  // up-to-date during processing.
  BasicBlock* GetParentBlock(uint32_t id);

  // Marks live blocks reachable from the entry of |func|. Simplifies constant
  // branches and switches as it proceeds, to limit the number of live blocks.
  // It is careful not to eliminate backedges even if they are dead, but the
  // header is live. Likewise, unreachable merge blocks named in live merge
  // instruction must be retained (though they may be clobbered).
  bool MarkLiveBlocks(Function* func,
                      std::unordered_set<BasicBlock*>* live_blocks);

  // Checks for unreachable merge and continue blocks with live headers; those
  // blocks must be retained. Continues are tracked separately so that a live
  // phi can be updated to take an undef value from any of its predecessors
  // that are unreachable continues.
  //
  // |unreachable_continues| maps the id of an unreachable continue target to
  // the header block that declares it.
  void MarkUnreachableStructuredTargets(
      const std::unordered_set<BasicBlock*>& live_blocks,
      std::unordered_set<BasicBlock*>* unreachable_merges,
      std::unordered_map<BasicBlock*, BasicBlock*>* unreachable_continues);

  // Fix phis in reachable blocks so that only live (or unremovable) incoming
  // edges are present. If the block now only has a single live incoming edge,
  // remove the phi and replace its uses with its data input. If the single
  // remaining incoming edge is from the phi itself, the the phi is in an
  // unreachable single block loop. Either the block is dead and will be
  // removed, or it's reachable from an unreachable continue target. In the
  // latter case that continue target block will be collapsed into a block that
  // only branches back to its header and we'll eliminate the block with the
  // phi.
  //
  // |unreachable_continues| maps continue targets that cannot be reached to
  // merge instruction that declares them.
  bool FixPhiNodesInLiveBlocks(
      Function* func, const std::unordered_set<BasicBlock*>& live_blocks,
      const std::unordered_map<BasicBlock*, BasicBlock*>&
          unreachable_continues);

  // Erases dead blocks. Any block captured in |unreachable_merges| or
  // |unreachable_continues| is a dead block that is required to remain due to
  // a live merge instruction in the corresponding header. These blocks will
  // have their instructions clobbered and will become a label and terminator.
  // Unreachable merge blocks are terminated by OpUnreachable, while
  // unreachable continue blocks are terminated by an unconditional branch to
  // the header. Otherwise, blocks are dead if not explicitly captured in
  // |live_blocks| and are totally removed.
  //
  // |unreachable_continues| maps continue targets that cannot be reached to
  // corresponding header block that declares them.
  bool EraseDeadBlocks(
      Function* func, const std::unordered_set<BasicBlock*>& live_blocks,
      const std::unordered_set<BasicBlock*>& unreachable_merges,
      const std::unordered_map<BasicBlock*, BasicBlock*>&
          unreachable_continues);

  // Reorders blocks in reachable functions so that they satisfy dominator
  // block ordering rules.
  void FixBlockOrder();

  // Return the first branch instruction that is a conditional branch to
  // |merge_block_id|. Returns |nullptr| if no such branch exists. If there are
  // multiple such branches, the first one is the one that would be executed
  // first when running the code.  That is, the one that dominates all of the
  // others.
  //
  // |start_block_id| must be a block whose innermost containing merge construct
  // has |merge_block_id| as the merge block.
  //
  // |loop_merge_id| and |loop_continue_id| are the merge and continue block ids
  // of the innermost loop containing |start_block_id|.
  Instruction* FindFirstExitFromSelectionMerge(uint32_t start_block_id,
                                               uint32_t merge_block_id,
                                               uint32_t loop_merge_id,
                                               uint32_t loop_continue_id);

  // Adds to |blocks_with_back_edges| all of the blocks on the path from the
  // basic block |cont_id| to |header_id| and |merge_id|.  The intention is that
  // |cond_id| is a the continue target of a loop, |header_id| is the header of
  // the loop, and |merge_id| is the merge block of the loop.
  void AddBlocksWithBackEdge(
      uint32_t cont_id, uint32_t header_id, uint32_t merge_id,
      std::unordered_set<BasicBlock*>* blocks_with_back_edges);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_DEAD_BRANCH_ELIM_PASS_H_
