// Copyright (c) 2018 Google LLC
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

#include "source/reduce/structured_loop_to_selection_reduction_opportunity_finder.h"

#include "source/reduce/structured_loop_to_selection_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

using opt::IRContext;

namespace {
const uint32_t kMergeNodeIndex = 0;
const uint32_t kContinueNodeIndex = 1;
}  // namespace

std::vector<std::unique_ptr<ReductionOpportunity>>
StructuredLoopToSelectionReductionOpportunityFinder::GetAvailableOpportunities(
    IRContext* context) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  std::set<uint32_t> merge_block_ids;
  for (auto& function : *context->module()) {
    for (auto& block : function) {
      auto merge_block_id = block.MergeBlockIdIfAny();
      if (merge_block_id) {
        merge_block_ids.insert(merge_block_id);
      }
    }
  }

  // Consider each loop construct header in the module.
  for (auto& function : *context->module()) {
    for (auto& block : function) {
      auto loop_merge_inst = block.GetLoopMergeInst();
      if (!loop_merge_inst) {
        // This is not a loop construct header.
        continue;
      }

      uint32_t continue_block_id =
          loop_merge_inst->GetSingleWordOperand(kContinueNodeIndex);

      // Check whether the loop construct's continue target is the merge block
      // of some structured control flow construct.  If it is, we cautiously do
      // not consider applying a transformation.
      if (merge_block_ids.find(continue_block_id) != merge_block_ids.end()) {
        continue;
      }

      // Check whether the loop header block is also the continue target. If it
      // is, we cautiously do not consider applying a transformation.
      if (block.id() == continue_block_id) {
        continue;
      }

      // Check whether the loop construct header dominates its merge block.
      // If not, the merge block must be unreachable in the control flow graph
      // so we cautiously do not consider applying a transformation.
      auto merge_block_id =
          loop_merge_inst->GetSingleWordInOperand(kMergeNodeIndex);
      if (!context->GetDominatorAnalysis(&function)->Dominates(
              block.id(), merge_block_id)) {
        continue;
      }

      // Check whether the loop construct merge block postdominates the loop
      // construct header.  If not (e.g. because the loop contains OpReturn,
      // OpKill or OpUnreachable), we cautiously do not consider applying
      // a transformation.
      if (!context->GetPostDominatorAnalysis(&function)->Dominates(
              merge_block_id, block.id())) {
        continue;
      }

      // We can turn this structured loop into a selection, so add the
      // opportunity to do so.
      result.push_back(
          MakeUnique<StructuredLoopToSelectionReductionOpportunity>(
              context, &block, &function));
    }
  }
  return result;
}

std::string StructuredLoopToSelectionReductionOpportunityFinder::GetName()
    const {
  return "StructuredLoopToSelectionReductionOpportunityFinder";
}

}  // namespace reduce
}  // namespace spvtools
