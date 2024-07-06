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

#include "source/reduce/remove_block_reduction_opportunity.h"

#include "source/opt/ir_context.h"

namespace spvtools {
namespace reduce {

RemoveBlockReductionOpportunity::RemoveBlockReductionOpportunity(
    opt::IRContext* context, opt::Function* function, opt::BasicBlock* block)
    : context_(context), function_(function), block_(block) {
  // precondition:
  assert(block_->begin() != block_->end() &&
         context_->get_def_use_mgr()->NumUsers(block_->id()) == 0 &&
         "RemoveBlockReductionOpportunity block must have 0 references");
}

bool RemoveBlockReductionOpportunity::PreconditionHolds() {
  // Removing other blocks cannot disable this opportunity.
  return true;
}

void RemoveBlockReductionOpportunity::Apply() {
  // We need an iterator pointing to the block, hence the loop.
  for (auto bi = function_->begin(); bi != function_->end(); ++bi) {
    if (bi->id() == block_->id()) {
      bi.Erase();
      context_->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
      return;
    }
  }

  assert(false &&
         "Unreachable: we should have found a block with the desired id.");
}

}  // namespace reduce
}  // namespace spvtools
