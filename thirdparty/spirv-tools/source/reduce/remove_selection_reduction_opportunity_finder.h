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

#ifndef SOURCE_REDUCE_REMOVE_SELECTION_REDUCTION_OPPORTUNITY_FINDER_H_
#define SOURCE_REDUCE_REMOVE_SELECTION_REDUCTION_OPPORTUNITY_FINDER_H_

#include "source/reduce/reduction_opportunity_finder.h"

namespace spvtools {
namespace reduce {

// A finder for opportunities for removing a selection construct by simply
// removing the OpSelectionMerge instruction; thus, the selections must have
// already been simplified to a point where they can be trivially removed.
class RemoveSelectionReductionOpportunityFinder
    : public ReductionOpportunityFinder {
 public:
  RemoveSelectionReductionOpportunityFinder() = default;

  ~RemoveSelectionReductionOpportunityFinder() override = default;

  std::string GetName() const final;

  std::vector<std::unique_ptr<ReductionOpportunity>> GetAvailableOpportunities(
      opt::IRContext* context, uint32_t target_function) const final;

  // Returns true if the OpSelectionMerge instruction |merge_instruction| in
  // block |header_block| can be removed.
  static bool CanOpSelectionMergeBeRemoved(
      opt::IRContext* context, const opt::BasicBlock& header_block,
      opt::Instruction* merge_instruction,
      std::unordered_set<uint32_t> merge_and_continue_blocks_from_loops);
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REMOVE_SELECTION_REDUCTION_OPPORTUNITY_FINDER_H_
