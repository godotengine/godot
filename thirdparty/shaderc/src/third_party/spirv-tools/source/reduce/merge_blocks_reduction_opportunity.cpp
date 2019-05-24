// Copyright (c) 2019 Google LLC
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

#include "source/reduce/merge_blocks_reduction_opportunity.h"

#include "source/opt/block_merge_util.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace reduce {

using opt::BasicBlock;
using opt::Function;
using opt::IRContext;

MergeBlocksReductionOpportunity::MergeBlocksReductionOpportunity(
    IRContext* context, Function* function, BasicBlock* block) {
  // Precondition: the terminator has to be OpBranch.
  assert(block->terminator()->opcode() == SpvOpBranch);
  context_ = context;
  function_ = function;
  // Get the successor block associated with the OpBranch.
  successor_block_ =
      context->cfg()->block(block->terminator()->GetSingleWordInOperand(0));
}

bool MergeBlocksReductionOpportunity::PreconditionHolds() {
  // Merge block opportunities can disable each other.
  // Example: Given blocks: A->B->C.
  // A is a loop header; B and C are blocks in the loop; C ends with OpReturn.
  // There are two opportunities: B and C can be merged with their predecessors.
  // Merge C. B now ends with OpReturn. We now just have: A->B.
  // Merge B is now disabled, as this would lead to A, a loop header, ending
  // with an OpReturn, which is invalid.

  const auto predecessors = context_->cfg()->preds(successor_block_->id());
  assert(1 == predecessors.size() &&
         "For a successor to be merged into its predecessor, exactly one "
         "predecessor must be present.");
  const uint32_t predecessor_id = predecessors[0];
  BasicBlock* predecessor_block = context_->get_instr_block(predecessor_id);
  return opt::blockmergeutil::CanMergeWithSuccessor(context_,
                                                    predecessor_block);
}

void MergeBlocksReductionOpportunity::Apply() {
  // While the original block that targeted the successor may not exist anymore
  // (it might have been merged with another block), some block must exist that
  // targets the successor.  Find it.

  const auto predecessors = context_->cfg()->preds(successor_block_->id());
  assert(1 == predecessors.size() &&
         "For a successor to be merged into its predecessor, exactly one "
         "predecessor must be present.");
  const uint32_t predecessor_id = predecessors[0];

  // We need an iterator pointing to the predecessor, hence the loop.
  for (auto bi = function_->begin(); bi != function_->end(); ++bi) {
    if (bi->id() == predecessor_id) {
      opt::blockmergeutil::MergeWithSuccessor(context_, function_, bi);
      // Block merging changes the control flow graph, so invalidate it.
      context_->InvalidateAnalysesExceptFor(IRContext::Analysis::kAnalysisNone);
      return;
    }
  }

  assert(false &&
         "Unreachable: we should have found a block with the desired id.");
}

}  // namespace reduce
}  // namespace spvtools
