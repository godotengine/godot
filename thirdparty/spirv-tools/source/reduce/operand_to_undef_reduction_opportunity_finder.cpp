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

#include "source/reduce/operand_to_undef_reduction_opportunity_finder.h"

#include "source/opt/instruction.h"
#include "source/reduce/change_operand_to_undef_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

std::vector<std::unique_ptr<ReductionOpportunity>>
OperandToUndefReductionOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context, uint32_t target_function) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  for (auto* function : GetTargetFunctions(context, target_function)) {
    for (auto& block : *function) {
      for (auto& inst : block) {
        // Skip instructions that result in a pointer type.
        auto type_id = inst.type_id();
        if (type_id) {
          auto type_id_def = context->get_def_use_mgr()->GetDef(type_id);
          if (type_id_def->opcode() == spv::Op::OpTypePointer) {
            continue;
          }
        }

        // We iterate through the operands using an explicit index (rather
        // than using a lambda) so that we use said index in the construction
        // of a ChangeOperandToUndefReductionOpportunity
        for (uint32_t index = 0; index < inst.NumOperands(); index++) {
          const auto& operand = inst.GetOperand(index);

          if (spvIsInIdType(operand.type)) {
            const auto operand_id = operand.words[0];
            auto operand_id_def =
                context->get_def_use_mgr()->GetDef(operand_id);

            // Skip constant and undef operands.
            // We always want the reducer to make the module "smaller", which
            // ensures termination.
            // Therefore, we assume: id > undef id > constant id.
            if (spvOpcodeIsConstantOrUndef(operand_id_def->opcode())) {
              continue;
            }

            // Don't replace function operands with undef.
            if (operand_id_def->opcode() == spv::Op::OpFunction) {
              continue;
            }

            // Only consider operands that have a type.
            auto operand_type_id = operand_id_def->type_id();
            if (operand_type_id) {
              auto operand_type_id_def =
                  context->get_def_use_mgr()->GetDef(operand_type_id);

              // Skip pointer operands.
              if (operand_type_id_def->opcode() == spv::Op::OpTypePointer) {
                continue;
              }

              result.push_back(
                  MakeUnique<ChangeOperandToUndefReductionOpportunity>(
                      context, &inst, index));
            }
          }
        }
      }
    }
  }
  return result;
}

std::string OperandToUndefReductionOpportunityFinder::GetName() const {
  return "OperandToUndefReductionOpportunityFinder";
}

}  // namespace reduce
}  // namespace spvtools
