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

#include "source/reduce/change_operand_to_undef_reduction_opportunity.h"

#include "source/opt/ir_context.h"
#include "source/reduce/reduction_util.h"

namespace spvtools {
namespace reduce {

bool ChangeOperandToUndefReductionOpportunity::PreconditionHolds() {
  // Check that the instruction still has the original operand.
  return inst_->NumOperands() > operand_index_ &&
         inst_->GetOperand(operand_index_).words[0] == original_id_;
}

void ChangeOperandToUndefReductionOpportunity::Apply() {
  auto operand = inst_->GetOperand(operand_index_);
  auto operand_id = operand.words[0];
  auto operand_id_def = context_->get_def_use_mgr()->GetDef(operand_id);
  auto operand_type_id = operand_id_def->type_id();
  // The opportunity should not exist unless this holds.
  assert(operand_type_id);
  auto undef_id = FindOrCreateGlobalUndef(context_, operand_type_id);
  inst_->SetOperand(operand_index_, {undef_id});
}

}  // namespace reduce
}  // namespace spvtools
