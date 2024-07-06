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

#include "source/reduce/conditional_branch_to_simple_conditional_branch_opportunity_finder.h"

#include "source/reduce/conditional_branch_to_simple_conditional_branch_reduction_opportunity.h"
#include "source/reduce/reduction_util.h"

namespace spvtools {
namespace reduce {

std::vector<std::unique_ptr<ReductionOpportunity>>
ConditionalBranchToSimpleConditionalBranchOpportunityFinder::
    GetAvailableOpportunities(opt::IRContext* context,
                              uint32_t target_function) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  // Find the opportunities for redirecting all false targets before the
  // opportunities for redirecting all true targets because the former
  // opportunities disable the latter, and vice versa, and the efficiency of the
  // reducer is improved by avoiding contiguous opportunities that disable one
  // another.
  for (bool redirect_to_true : {true, false}) {
    // Consider every relevant function.
    for (auto* function : GetTargetFunctions(context, target_function)) {
      // Consider every block in the function.
      for (auto& block : *function) {
        // The terminator must be spv::Op::OpBranchConditional.
        opt::Instruction* terminator = block.terminator();
        if (terminator->opcode() != spv::Op::OpBranchConditional) {
          continue;
        }

        uint32_t true_block_id =
            terminator->GetSingleWordInOperand(kTrueBranchOperandIndex);
        uint32_t false_block_id =
            terminator->GetSingleWordInOperand(kFalseBranchOperandIndex);

        // The conditional branch must not already be simplified.
        if (true_block_id == false_block_id) {
          continue;
        }

        // The redirected target must not be a back-edge to a structured loop
        // header.
        uint32_t redirected_block_id =
            redirect_to_true ? false_block_id : true_block_id;
        uint32_t containing_loop_header =
            context->GetStructuredCFGAnalysis()->ContainingLoop(block.id());
        // The structured CFG analysis does not include a loop header as part
        // of the loop construct, but we want to include it, so handle this
        // special case:
        if (block.GetLoopMergeInst() != nullptr) {
          containing_loop_header = block.id();
        }
        if (redirected_block_id == containing_loop_header) {
          continue;
        }

        result.push_back(
            MakeUnique<
                ConditionalBranchToSimpleConditionalBranchReductionOpportunity>(
                context, block.terminator(), redirect_to_true));
      }
    }
  }
  return result;
}

std::string
ConditionalBranchToSimpleConditionalBranchOpportunityFinder::GetName() const {
  return "ConditionalBranchToSimpleConditionalBranchOpportunityFinder";
}

}  // namespace reduce
}  // namespace spvtools
