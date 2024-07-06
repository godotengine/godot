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

#include "source/reduce/simple_conditional_branch_to_branch_reduction_opportunity.h"

#include "source/reduce/reduction_util.h"

namespace spvtools {
namespace reduce {

SimpleConditionalBranchToBranchReductionOpportunity::
    SimpleConditionalBranchToBranchReductionOpportunity(
        opt::Instruction* conditional_branch_instruction)
    : conditional_branch_instruction_(conditional_branch_instruction) {}

bool SimpleConditionalBranchToBranchReductionOpportunity::PreconditionHolds() {
  // We find at most one opportunity per conditional branch and simplifying
  // another branch cannot disable this opportunity.
  return true;
}

void SimpleConditionalBranchToBranchReductionOpportunity::Apply() {
  assert(conditional_branch_instruction_->opcode() ==
             spv::Op::OpBranchConditional &&
         "SimpleConditionalBranchToBranchReductionOpportunity: branch was not "
         "a conditional branch");

  assert(conditional_branch_instruction_->GetSingleWordInOperand(
             kTrueBranchOperandIndex) ==
             conditional_branch_instruction_->GetSingleWordInOperand(
                 kFalseBranchOperandIndex) &&
         "SimpleConditionalBranchToBranchReductionOpportunity: branch was not "
         "simple");

  // OpBranchConditional %condition %block_id %block_id ...
  // ->
  // OpBranch %block_id

  conditional_branch_instruction_->SetOpcode(spv::Op::OpBranch);
  conditional_branch_instruction_->ReplaceOperands(
      {{SPV_OPERAND_TYPE_ID,
        {conditional_branch_instruction_->GetSingleWordInOperand(
            kTrueBranchOperandIndex)}}});
  conditional_branch_instruction_->context()->InvalidateAnalysesExceptFor(
      opt::IRContext::kAnalysisNone);
}

}  // namespace reduce
}  // namespace spvtools
