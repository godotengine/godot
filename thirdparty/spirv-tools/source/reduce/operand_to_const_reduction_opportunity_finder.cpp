// Copyright (c) 2018 Google LLC
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

#include "source/reduce/operand_to_const_reduction_opportunity_finder.h"

#include "source/opt/instruction.h"
#include "source/reduce/change_operand_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

std::vector<std::unique_ptr<ReductionOpportunity>>
OperandToConstReductionOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context, uint32_t target_function) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;
  assert(result.empty());

  // We first loop over all constants.  This means that all the reduction
  // opportunities to replace an operand with a particular constant will be
  // contiguous, and in particular it means that multiple, incompatible
  // reduction opportunities that try to replace the same operand with distinct
  // constants are likely to be discontiguous.  This is good because the
  // reducer works in the spirit of delta debugging and tries applying large
  // contiguous blocks of opportunities early on, and we want to avoid having a
  // large block of incompatible opportunities if possible.
  for (const auto& constant : context->GetConstants()) {
    for (auto* function : GetTargetFunctions(context, target_function)) {
      for (auto& block : *function) {
        for (auto& inst : block) {
          // We iterate through the operands using an explicit index (rather
          // than using a lambda) so that we use said index in the construction
          // of a ChangeOperandReductionOpportunity
          for (uint32_t index = 0; index < inst.NumOperands(); index++) {
            const auto& operand = inst.GetOperand(index);
            if (spvIsInIdType(operand.type)) {
              const auto id = operand.words[0];
              auto def = context->get_def_use_mgr()->GetDef(id);
              if (spvOpcodeIsConstant(def->opcode())) {
                // The argument is already a constant.
                continue;
              }
              if (def->opcode() == spv::Op::OpFunction) {
                // The argument refers to a function, e.g. the function called
                // by OpFunctionCall; avoid replacing this with a constant of
                // the function's return type.
                continue;
              }
              auto type_id = def->type_id();
              if (type_id) {
                if (constant->type_id() == type_id) {
                  result.push_back(
                      MakeUnique<ChangeOperandReductionOpportunity>(
                          &inst, index, constant->result_id()));
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}

std::string OperandToConstReductionOpportunityFinder::GetName() const {
  return "OperandToConstReductionOpportunityFinder";
}

}  // namespace reduce
}  // namespace spvtools
