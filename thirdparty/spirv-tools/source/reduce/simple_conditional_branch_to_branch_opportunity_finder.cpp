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

#include "source/reduce/simple_conditional_branch_to_branch_opportunity_finder.h"

#include "source/reduce/reduction_util.h"
#include "source/reduce/simple_conditional_branch_to_branch_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

std::vector<std::unique_ptr<ReductionOpportunity>>
SimpleConditionalBranchToBranchOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context, uint32_t target_function) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  // Consider every function.
  for (auto* function : GetTargetFunctions(context, target_function)) {
    // Consider every block in the function.
    for (auto& block : *function) {
      // The terminator must be spv::Op::OpBranchConditional.
      opt::Instruction* terminator = block.terminator();
      if (terminator->opcode() != spv::Op::OpBranchConditional) {
        continue;
      }
      // It must not be a selection header, as these cannot be followed by
      // OpBranch.
      if (block.GetMergeInst() &&
          block.GetMergeInst()->opcode() == spv::Op::OpSelectionMerge) {
        continue;
      }
      // The conditional branch must be simplified.
      if (terminator->GetSingleWordInOperand(kTrueBranchOperandIndex) !=
          terminator->GetSingleWordInOperand(kFalseBranchOperandIndex)) {
        continue;
      }

      result.push_back(
          MakeUnique<SimpleConditionalBranchToBranchReductionOpportunity>(
              block.terminator()));
    }
  }
  return result;
}

std::string SimpleConditionalBranchToBranchOpportunityFinder::GetName() const {
  return "SimpleConditionalBranchToBranchOpportunityFinder";
}

}  // namespace reduce
}  // namespace spvtools
