// Copyright (c) 2021 Alastair F. Donaldson
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

#include "source/reduce/structured_construct_to_block_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

bool StructuredConstructToBlockReductionOpportunity::PreconditionHolds() {
  return context_->get_def_use_mgr()->GetDef(construct_header_) != nullptr;
}

void StructuredConstructToBlockReductionOpportunity::Apply() {
  auto header_block = context_->cfg()->block(construct_header_);
  auto merge_block = context_->cfg()->block(header_block->MergeBlockId());

  auto* enclosing_function = header_block->GetParent();

  // A region of blocks is defined in terms of dominators and post-dominators,
  // so we compute these for the enclosing function.
  auto* dominators = context_->GetDominatorAnalysis(enclosing_function);
  auto* postdominators = context_->GetPostDominatorAnalysis(enclosing_function);

  // For each block in the function, determine whether it is inside the region.
  // If it is, delete it.
  for (auto block_it = enclosing_function->begin();
       block_it != enclosing_function->end();) {
    if (header_block != &*block_it && merge_block != &*block_it &&
        dominators->Dominates(header_block, &*block_it) &&
        postdominators->Dominates(merge_block, &*block_it)) {
      block_it = block_it.Erase();
    } else {
      ++block_it;
    }
  }
  // Having removed some blocks from the module it is necessary to invalidate
  // analyses, since the remaining patch-up work depends on various analyses
  // which will otherwise reference blocks that have been deleted.
  context_->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

  // We demote the header of the region to a regular block by deleting its merge
  // instruction.
  context_->KillInst(header_block->GetMergeInst());

  // The terminator for the header block is changed to be an unconditional
  // branch to the merge block.
  header_block->terminator()->SetOpcode(spv::Op::OpBranch);
  header_block->terminator()->SetInOperands(
      {{SPV_OPERAND_TYPE_ID, {merge_block->id()}}});

  // This is an intrusive change, so we invalidate all analyses.
  context_->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

}  // namespace reduce
}  // namespace spvtools
