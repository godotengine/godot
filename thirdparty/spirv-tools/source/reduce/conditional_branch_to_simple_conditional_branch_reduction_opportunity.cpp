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

#include "source/reduce/conditional_branch_to_simple_conditional_branch_reduction_opportunity.h"

#include "source/reduce/reduction_util.h"

namespace spvtools {
namespace reduce {

ConditionalBranchToSimpleConditionalBranchReductionOpportunity::
    ConditionalBranchToSimpleConditionalBranchReductionOpportunity(
        opt::IRContext* context,
        opt::Instruction* conditional_branch_instruction, bool redirect_to_true)
    : context_(context),
      conditional_branch_instruction_(conditional_branch_instruction),
      redirect_to_true_(redirect_to_true) {}

bool ConditionalBranchToSimpleConditionalBranchReductionOpportunity::
    PreconditionHolds() {
  // Another opportunity may have already simplified this conditional branch,
  // which should disable this opportunity.
  return conditional_branch_instruction_->GetSingleWordInOperand(
             kTrueBranchOperandIndex) !=
         conditional_branch_instruction_->GetSingleWordInOperand(
             kFalseBranchOperandIndex);
}

void ConditionalBranchToSimpleConditionalBranchReductionOpportunity::Apply() {
  uint32_t operand_to_modify =
      redirect_to_true_ ? kFalseBranchOperandIndex : kTrueBranchOperandIndex;
  uint32_t operand_to_copy =
      redirect_to_true_ ? kTrueBranchOperandIndex : kFalseBranchOperandIndex;

  auto old_successor_block_id =
      conditional_branch_instruction_->GetSingleWordInOperand(
          operand_to_modify);

  // Do the branch redirection.
  conditional_branch_instruction_->SetInOperand(
      operand_to_modify,
      {conditional_branch_instruction_->GetSingleWordInOperand(
          operand_to_copy)});

  // The old successor block may have phi instructions; these will need to
  // respect the change in edges.
  AdaptPhiInstructionsForRemovedEdge(
      context_->get_instr_block(conditional_branch_instruction_)->id(),
      context_->cfg()->block(old_successor_block_id));

  // We have changed the CFG.
  context_->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

}  // namespace reduce
}  // namespace spvtools
