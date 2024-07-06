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

#include "source/reduce/merge_blocks_reduction_opportunity_finder.h"
#include "source/opt/block_merge_util.h"
#include "source/reduce/merge_blocks_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

std::string MergeBlocksReductionOpportunityFinder::GetName() const {
  return "MergeBlocksReductionOpportunityFinder";
}

std::vector<std::unique_ptr<ReductionOpportunity>>
MergeBlocksReductionOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context, uint32_t target_function) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  // Consider every block in every function.
  for (auto* function : GetTargetFunctions(context, target_function)) {
    for (auto& block : *function) {
      // See whether it is possible to merge this block with its successor.
      if (opt::blockmergeutil::CanMergeWithSuccessor(context, &block)) {
        // It is, so record an opportunity to do this.
        result.push_back(spvtools::MakeUnique<MergeBlocksReductionOpportunity>(
            context, function, &block));
      }
    }
  }
  return result;
}

}  // namespace reduce
}  // namespace spvtools
