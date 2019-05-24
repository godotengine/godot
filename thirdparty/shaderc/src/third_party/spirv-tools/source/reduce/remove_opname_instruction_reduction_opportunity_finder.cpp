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

#include "source/reduce/remove_opname_instruction_reduction_opportunity_finder.h"

#include "source/opcode.h"
#include "source/opt/instruction.h"
#include "source/reduce/remove_instruction_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

using opt::IRContext;

std::vector<std::unique_ptr<ReductionOpportunity>>
RemoveOpNameInstructionReductionOpportunityFinder::GetAvailableOpportunities(
    IRContext* context) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  for (auto& inst : context->module()->debugs2()) {
    if (inst.opcode() == SpvOpName || inst.opcode() == SpvOpMemberName) {
      result.push_back(
          MakeUnique<RemoveInstructionReductionOpportunity>(&inst));
    }
  }
  return result;
}

std::string RemoveOpNameInstructionReductionOpportunityFinder::GetName() const {
  return "RemoveOpNameInstructionReductionOpportunityFinder";
}

}  // namespace reduce
}  // namespace spvtools
