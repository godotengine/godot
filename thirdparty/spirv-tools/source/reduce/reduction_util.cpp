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

#include "source/reduce/reduction_util.h"

#include "source/opt/ir_context.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace reduce {

const uint32_t kTrueBranchOperandIndex = 1;
const uint32_t kFalseBranchOperandIndex = 2;

uint32_t FindOrCreateGlobalVariable(opt::IRContext* context,
                                    uint32_t pointer_type_id) {
  for (auto& inst : context->module()->types_values()) {
    if (inst.opcode() != spv::Op::OpVariable) {
      continue;
    }
    if (inst.type_id() == pointer_type_id) {
      return inst.result_id();
    }
  }
  const uint32_t variable_id = context->TakeNextId();
  auto variable_inst = MakeUnique<opt::Instruction>(
      context, spv::Op::OpVariable, pointer_type_id, variable_id,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_STORAGE_CLASS,
            {static_cast<uint32_t>(context->get_type_mgr()
                                       ->GetType(pointer_type_id)
                                       ->AsPointer()
                                       ->storage_class())}}}));
  context->module()->AddGlobalValue(std::move(variable_inst));
  return variable_id;
}

uint32_t FindOrCreateFunctionVariable(opt::IRContext* context,
                                      opt::Function* function,
                                      uint32_t pointer_type_id) {
  // The pointer type of a function variable must have Function storage class.
  assert(context->get_type_mgr()
             ->GetType(pointer_type_id)
             ->AsPointer()
             ->storage_class() == spv::StorageClass::Function);

  // Go through the instructions in the function's first block until we find a
  // suitable variable, or go past all the variables.
  opt::BasicBlock::iterator iter = function->begin()->begin();
  for (;; ++iter) {
    // We will either find a suitable variable, or find a non-variable
    // instruction; we won't exhaust all instructions.
    assert(iter != function->begin()->end());
    if (iter->opcode() != spv::Op::OpVariable) {
      // If we see a non-variable, we have gone through all the variables.
      break;
    }
    if (iter->type_id() == pointer_type_id) {
      return iter->result_id();
    }
  }
  // At this point, iter refers to the first non-function instruction of the
  // function's entry block.
  const uint32_t variable_id = context->TakeNextId();
  auto variable_inst = MakeUnique<opt::Instruction>(
      context, spv::Op::OpVariable, pointer_type_id, variable_id,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_STORAGE_CLASS,
            {uint32_t(spv::StorageClass::Function)}}}));
  iter->InsertBefore(std::move(variable_inst));
  return variable_id;
}

uint32_t FindOrCreateGlobalUndef(opt::IRContext* context, uint32_t type_id) {
  for (auto& inst : context->module()->types_values()) {
    if (inst.opcode() != spv::Op::OpUndef) {
      continue;
    }
    if (inst.type_id() == type_id) {
      return inst.result_id();
    }
  }
  const uint32_t undef_id = context->TakeNextId();
  auto undef_inst =
      MakeUnique<opt::Instruction>(context, spv::Op::OpUndef, type_id, undef_id,
                                   opt::Instruction::OperandList());
  assert(undef_id == undef_inst->result_id());
  context->module()->AddGlobalValue(std::move(undef_inst));
  return undef_id;
}

void AdaptPhiInstructionsForRemovedEdge(uint32_t from_id,
                                        opt::BasicBlock* to_block) {
  to_block->ForEachPhiInst([&from_id](opt::Instruction* phi_inst) {
    opt::Instruction::OperandList new_in_operands;
    // Go through the OpPhi's input operands in (variable, parent) pairs.
    for (uint32_t index = 0; index < phi_inst->NumInOperands(); index += 2) {
      // Keep all pairs where the parent is not the block from which the edge
      // is being removed.
      if (phi_inst->GetInOperand(index + 1).words[0] != from_id) {
        new_in_operands.push_back(phi_inst->GetInOperand(index));
        new_in_operands.push_back(phi_inst->GetInOperand(index + 1));
      }
    }
    phi_inst->SetInOperands(std::move(new_in_operands));
  });
}

}  // namespace reduce
}  // namespace spvtools
