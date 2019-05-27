// Copyright (c) 2018 Google Inc.
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

#include "source/reduce/remove_unreferenced_instruction_reduction_opportunity_finder.h"

#include "source/opcode.h"
#include "source/opt/instruction.h"
#include "source/reduce/remove_instruction_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

using opt::IRContext;

std::vector<std::unique_ptr<ReductionOpportunity>>
RemoveUnreferencedInstructionReductionOpportunityFinder::
    GetAvailableOpportunities(IRContext* context) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  for (auto& function : *context->module()) {
    for (auto& block : function) {
      for (auto& inst : block) {
        if (context->get_def_use_mgr()->NumUses(&inst) > 0) {
          continue;
        }
        if (spvOpcodeIsBlockTerminator(inst.opcode()) ||
            inst.opcode() == SpvOpSelectionMerge ||
            inst.opcode() == SpvOpLoopMerge) {
          // In this reduction pass we do not want to affect static control
          // flow.
          continue;
        }
        // Given that we're in a block, we should only get here if the
        // instruction is not directly related to control flow; i.e., it's
        // some straightforward instruction with an unused result, like an
        // arithmetic operation or function call.
        result.push_back(
            MakeUnique<RemoveInstructionReductionOpportunity>(&inst));
      }
    }
  }
  return result;
}

std::string RemoveUnreferencedInstructionReductionOpportunityFinder::GetName()
    const {
  return "RemoveUnreferencedInstructionReductionOpportunityFinder";
}

}  // namespace reduce
}  // namespace spvtools
