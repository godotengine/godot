// Copyright (c) 2019 Google LLC.
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

#include "source/reduce/remove_selection_reduction_opportunity_finder.h"

#include "source/reduce/remove_selection_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

namespace {
const uint32_t kMergeNodeIndex = 0;
const uint32_t kContinueNodeIndex = 1;
}  // namespace

std::string RemoveSelectionReductionOpportunityFinder::GetName() const {
  return "RemoveSelectionReductionOpportunityFinder";
}

std::vector<std::unique_ptr<ReductionOpportunity>>
RemoveSelectionReductionOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context, uint32_t target_function) const {
  // Get all loop merge and continue blocks so we can check for these later.
  std::unordered_set<uint32_t> merge_and_continue_blocks_from_loops;
  for (auto* function : GetTargetFunctions(context, target_function)) {
    for (auto& block : *function) {
      if (auto merge_instruction = block.GetMergeInst()) {
        if (merge_instruction->opcode() == spv::Op::OpLoopMerge) {
          uint32_t merge_block_id =
              merge_instruction->GetSingleWordOperand(kMergeNodeIndex);
          uint32_t continue_block_id =
              merge_instruction->GetSingleWordOperand(kContinueNodeIndex);
          merge_and_continue_blocks_from_loops.insert(merge_block_id);
          merge_and_continue_blocks_from_loops.insert(continue_block_id);
        }
      }
    }
  }

  // Return all selection headers where the OpSelectionMergeInstruction can be
  // removed.
  std::vector<std::unique_ptr<ReductionOpportunity>> result;
  for (auto& function : *context->module()) {
    for (auto& block : function) {
      if (auto merge_instruction = block.GetMergeInst()) {
        if (merge_instruction->opcode() == spv::Op::OpSelectionMerge) {
          if (CanOpSelectionMergeBeRemoved(
                  context, block, merge_instruction,
                  merge_and_continue_blocks_from_loops)) {
            result.push_back(
                MakeUnique<RemoveSelectionReductionOpportunity>(&block));
          }
        }
      }
    }
  }
  return result;
}

bool RemoveSelectionReductionOpportunityFinder::CanOpSelectionMergeBeRemoved(
    opt::IRContext* context, const opt::BasicBlock& header_block,
    opt::Instruction* merge_instruction,
    std::unordered_set<uint32_t> merge_and_continue_blocks_from_loops) {
  assert(header_block.GetMergeInst() == merge_instruction &&
         "CanOpSelectionMergeBeRemoved(...): header block and merge "
         "instruction mismatch");

  // The OpSelectionMerge instruction is needed if either of the following are
  // true.
  //
  // 1. The header block has at least two (unique) successors that are not
  // merge or continue blocks of a loop.
  //
  // 2. The predecessors of the merge block are "using" the merge block to avoid
  // divergence. In other words, there exists a predecessor of the merge block
  // that has a successor that is not the merge block of this construct and not
  // a merge or continue block of a loop.

  // 1.
  {
    uint32_t divergent_successor_count = 0;

    std::unordered_set<uint32_t> seen_successors;

    header_block.ForEachSuccessorLabel(
        [&seen_successors, &merge_and_continue_blocks_from_loops,
         &divergent_successor_count](uint32_t successor) {
          // Not already seen.
          if (seen_successors.find(successor) == seen_successors.end()) {
            seen_successors.insert(successor);
            // Not a loop continue or merge.
            if (merge_and_continue_blocks_from_loops.find(successor) ==
                merge_and_continue_blocks_from_loops.end()) {
              ++divergent_successor_count;
            }
          }
        });

    if (divergent_successor_count > 1) {
      return false;
    }
  }

  // 2.
  {
    uint32_t merge_block_id =
        merge_instruction->GetSingleWordOperand(kMergeNodeIndex);
    for (uint32_t predecessor_block_id :
         context->cfg()->preds(merge_block_id)) {
      const opt::BasicBlock* predecessor_block =
          context->cfg()->block(predecessor_block_id);
      assert(predecessor_block);
      bool found_divergent_successor = false;
      predecessor_block->ForEachSuccessorLabel(
          [&found_divergent_successor, merge_block_id,
           &merge_and_continue_blocks_from_loops](uint32_t successor_id) {
            // The successor is not the merge block, nor a loop merge or
            // continue.
            if (successor_id != merge_block_id &&
                merge_and_continue_blocks_from_loops.find(successor_id) ==
                    merge_and_continue_blocks_from_loops.end()) {
              found_divergent_successor = true;
            }
          });
      if (found_divergent_successor) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace reduce
}  // namespace spvtools
